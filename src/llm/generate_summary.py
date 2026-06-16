import argparse
import re
from typing import Dict, List, Optional

import pandas as pd
from sqlalchemy import text

from src.database import get_engine
from src.llm.prompts import build_executive_summary_prompt
from src.llm.providers import call_llm, get_active_model_name


def ensure_llm_insights_table() -> None:
    """
    Create llm_insights table if it does not already exist.
    """
    engine = get_engine()

    query = text("""
        CREATE TABLE IF NOT EXISTS llm_insights (
            insight_id SERIAL PRIMARY KEY,
            company_id INTEGER REFERENCES companies(company_id),
            year INTEGER NOT NULL,
            quarter VARCHAR(10) NOT NULL,
            insight_type VARCHAR(100) NOT NULL,
            insight_text TEXT NOT NULL,
            model_name VARCHAR(100),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    with engine.begin() as conn:
        conn.execute(query)


def fetch_target_periods(
    company_name: Optional[str],
    year: Optional[int],
    quarter: Optional[str],
    limit: Optional[int],
) -> List[Dict]:
    """
    Fetch company/year/quarter combinations to generate LLM insights for.
    """
    engine = get_engine()

    filters = []
    params = {}

    if company_name:
        filters.append("c.company_name = :company_name")
        params["company_name"] = company_name

    if year:
        filters.append("t.year = :year")
        params["year"] = year

    if quarter:
        filters.append("t.quarter = :quarter")
        params["quarter"] = quarter

    where_clause = ""
    if filters:
        where_clause = "WHERE " + " AND ".join(filters)

    limit_clause = ""
    if limit:
        limit_clause = "LIMIT :limit"
        params["limit"] = limit

    query = text(f"""
        SELECT
            c.company_id,
            c.company_name,
            t.year,
            t.quarter
        FROM transcripts t
        JOIN companies c
            ON t.company_id = c.company_id
        {where_clause}
        ORDER BY c.company_name, t.year DESC, t.quarter DESC
        {limit_clause};
    """)

    with engine.connect() as conn:
        rows = conn.execute(query, params).mappings().all()

    return [dict(row) for row in rows]


def fetch_sentiment_context(company_id: int, year: int, quarter: str) -> Dict:
    engine = get_engine()

    query = text("""
        SELECT
            sentiment_score,
            sentiment_label
        FROM sentiment_scores
        WHERE company_id = :company_id
          AND year = :year
          AND quarter = :quarter
        LIMIT 1;
    """)

    params = {
        "company_id": company_id,
        "year": year,
        "quarter": quarter,
    }

    with engine.connect() as conn:
        row = conn.execute(query, params).mappings().first()

    if not row:
        return {
            "sentiment_score": 0.0,
            "sentiment_label": "unknown",
        }

    return dict(row)


def fetch_top_topics(company_id: int, year: int, quarter: str, limit: int = 6) -> List[Dict]:
    engine = get_engine()

    query = text("""
        SELECT
            topic_name,
            topic_count,
            ROUND(topic_intensity * 10000, 2) AS mentions_per_10k_words
        FROM topic_scores
        WHERE company_id = :company_id
          AND year = :year
          AND quarter = :quarter
        ORDER BY topic_count DESC
        LIMIT :limit;
    """)

    params = {
        "company_id": company_id,
        "year": year,
        "quarter": quarter,
        "limit": limit,
    }

    with engine.connect() as conn:
        rows = conn.execute(query, params).mappings().all()

    return [dict(row) for row in rows]


def fetch_top_risks(company_id: int, year: int, quarter: str, limit: int = 6) -> List[Dict]:
    engine = get_engine()

    query = text("""
        SELECT
            risk_category,
            risk_keyword,
            frequency
        FROM risk_signals
        WHERE company_id = :company_id
          AND year = :year
          AND quarter = :quarter
        ORDER BY frequency DESC
        LIMIT :limit;
    """)

    params = {
        "company_id": company_id,
        "year": year,
        "quarter": quarter,
        "limit": limit,
    }

    with engine.connect() as conn:
        rows = conn.execute(query, params).mappings().all()

    return [dict(row) for row in rows]


def fetch_risk_quotes(company_id: int, year: int, quarter: str, limit: int = 4) -> List[Dict]:
    engine = get_engine()

    query = text("""
        SELECT
            risk_category,
            example_quote
        FROM risk_signals
        WHERE company_id = :company_id
          AND year = :year
          AND quarter = :quarter
          AND frequency > 0
          AND example_quote IS NOT NULL
        ORDER BY frequency DESC
        LIMIT :limit;
    """)

    params = {
        "company_id": company_id,
        "year": year,
        "quarter": quarter,
        "limit": limit,
    }

    with engine.connect() as conn:
        rows = conn.execute(query, params).mappings().all()

    return [dict(row) for row in rows]

def split_text_into_sentences(text_value: str) -> List[str]:
    """
    Split transcript text into sentence-like snippets.
    """
    if not text_value:
        return []

    sentence_candidates = re.split(r"(?<=[.!?])\s+", text_value)

    return [sentence.strip() for sentence in sentence_candidates if sentence.strip()]


def keyword_matches_sentence(sentence: str, keyword: str) -> bool:
    """
    Check whether a keyword appears in a sentence.
    Uses word boundaries for single-token keywords.
    """
    if not sentence or not keyword:
        return False

    normalized_sentence = sentence.lower()
    normalized_keyword = keyword.lower().strip()
    escaped_keyword = re.escape(normalized_keyword)

    if re.fullmatch(r"[a-z0-9]+", normalized_keyword):
        pattern = rf"\b{escaped_keyword}\b"
    else:
        pattern = escaped_keyword

    return re.search(pattern, normalized_sentence, flags=re.IGNORECASE) is not None


def fetch_transcript_text(company_id: int, year: int, quarter: str) -> str:
    """
    Fetch cleaned transcript text for one company/period.
    """
    engine = get_engine()

    query = text("""
        SELECT cleaned_text
        FROM transcripts
        WHERE company_id = :company_id
          AND year = :year
          AND quarter = :quarter
        LIMIT 1;
    """)

    params = {
        "company_id": company_id,
        "year": year,
        "quarter": quarter,
    }

    with engine.connect() as conn:
        row = conn.execute(query, params).mappings().first()

    if not row:
        return ""

    return row["cleaned_text"] or ""


def build_topic_keyword_map() -> Dict[str, List[str]]:
    """
    Lightweight topic-to-keyword map used only for evidence snippet extraction.
    This does not replace the main topic classifier.
    """
    return {
        "Membership / Enrollment": [
            "membership",
            "members",
            "enrollment",
            "covered lives",
            "lives served",
        ],
        "Medicare / Medicare Advantage": [
            "medicare advantage",
            "medicare",
            "ma plan",
            "star ratings",
            "cms",
        ],
        "Medicaid": [
            "medicaid",
            "redetermination",
            "redeterminations",
            "managed medicaid",
        ],
        "Commercial Insurance": [
            "commercial",
            "employer group",
            "self-insured",
            "fully insured",
        ],
        "PBM / Pharmacy": [
            "pbm",
            "pharmacy",
            "caremark",
            "prescription",
            "drug pricing",
        ],
        "Cost Pressure": [
            "cost pressure",
            "cost trend",
            "medical cost",
            "medical loss ratio",
            "mlr",
        ],
        "Operating Margin": [
            "operating margin",
            "margin",
            "operating income",
            "profitability",
        ],
        "Value-Based Care": [
            "value-based care",
            "value based care",
            "risk-based arrangement",
            "shared savings",
        ],
        "Care Delivery": [
            "care delivery",
            "primary care",
            "virtual care",
            "care management",
            "patient care",
        ],
        "AI / Digital Health": [
            "artificial intelligence",
            "ai",
            "digital",
            "automation",
            "data analytics",
        ],
    }


def get_risk_keywords_from_record(risk: Dict) -> List[str]:
    """
    Split stored risk_keyword field into individual keywords.
    """
    risk_keyword_text = risk.get("risk_keyword") or ""

    return [
        keyword.strip()
        for keyword in risk_keyword_text.split(",")
        if keyword.strip()
    ]


def extract_transcript_evidence_snippets(
    transcript_text: str,
    top_topics: List[Dict],
    top_risks: List[Dict],
    max_snippets: int = 8,
) -> List[Dict]:
    """
    Extract selected transcript snippets that support top topic and risk signals.

    This keeps LLM prompts grounded without sending the full transcript.
    """
    # Skip the transcript header / participant list area.
    # This helps avoid selecting snippets such as company title,
    # conference call participants, and speaker lists as evidence.
    analysis_text = transcript_text[2500:] if len(transcript_text) > 2500 else transcript_text
    sentences = split_text_into_sentences(analysis_text)
    topic_keyword_map = build_topic_keyword_map()

    snippets: List[Dict] = []
    seen_snippets = set()

    def add_snippet(evidence_type: str, matched_signal: str, sentence: str) -> None:
        cleaned_sentence = sentence.strip()
        lower_sentence = cleaned_sentence.lower()
        # Filter the header / speaker list
        header_markers = [
            "earnings conference call",
            "company participants",
            "conference call participants",
            "analyst",
            "operator",
            "chief executive officer",
            "chief financial officer",
        ]

        if any(marker in lower_sentence for marker in header_markers):
            return    
        if len(cleaned_sentence) < 80:
            return

        snippet = cleaned_sentence[:900]

        if snippet in seen_snippets:
            return

        seen_snippets.add(snippet)

        snippets.append(
            {
                "evidence_type": evidence_type,
                "matched_signal": matched_signal,
                "snippet": snippet,
            }
        )

    for topic in top_topics[:4]:
        topic_name = topic["topic_name"]
        keywords = topic_keyword_map.get(topic_name, [])

        for sentence in sentences:
            if any(keyword_matches_sentence(sentence, keyword) for keyword in keywords):
                add_snippet("Topic Evidence", topic_name, sentence)
                break

        if len(snippets) >= max_snippets:
            return snippets

    for risk in top_risks[:4]:
        risk_category = risk["risk_category"]
        keywords = get_risk_keywords_from_record(risk)

        for sentence in sentences:
            if any(keyword_matches_sentence(sentence, keyword) for keyword in keywords):
                add_snippet("Risk Evidence", risk_category, sentence)
                break

        if len(snippets) >= max_snippets:
            return snippets

    return snippets

def save_llm_insight(
    company_id: int,
    year: int,
    quarter: str,
    insight_type: str,
    insight_text: str,
    model_name: str,
) -> None:
    """
    Save one LLM insight into PostgreSQL.
    """
    engine = get_engine()

    delete_query = text("""
        DELETE FROM llm_insights
        WHERE company_id = :company_id
          AND year = :year
          AND quarter = :quarter
          AND insight_type = :insight_type;
    """)

    insert_query = text("""
        INSERT INTO llm_insights (
            company_id,
            year,
            quarter,
            insight_type,
            insight_text,
            model_name
        )
        VALUES (
            :company_id,
            :year,
            :quarter,
            :insight_type,
            :insight_text,
            :model_name
        );
    """)

    params = {
        "company_id": company_id,
        "year": year,
        "quarter": quarter,
        "insight_type": insight_type,
        "insight_text": insight_text,
        "model_name": model_name,
    }

    with engine.begin() as conn:
        conn.execute(delete_query, params)
        conn.execute(insert_query, params)


def generate_company_summary(target: Dict, dry_run: bool = False) -> None:
    company_id = target["company_id"]
    company_name = target["company_name"]
    year = target["year"]
    quarter = target["quarter"]

    sentiment_context = fetch_sentiment_context(company_id, year, quarter)
    top_topics = fetch_top_topics(company_id, year, quarter)
    top_risks = fetch_top_risks(company_id, year, quarter)
    risk_quotes = fetch_risk_quotes(company_id, year, quarter)
    transcript_text = fetch_transcript_text(company_id, year, quarter)
    transcript_snippets = extract_transcript_evidence_snippets(
        transcript_text=transcript_text,
        top_topics=top_topics,
        top_risks=top_risks,
    )
    

    prompt = build_executive_summary_prompt(
    company_name=company_name,
    year=year,
    quarter=quarter,
    sentiment_score=sentiment_context["sentiment_score"],
    sentiment_label=sentiment_context["sentiment_label"],
    top_topics=top_topics,
    top_risks=top_risks,
    risk_quotes=risk_quotes,
    transcript_snippets=transcript_snippets,
    )

    print("\n" + "=" * 80)
    print(f"Generating LLM executive summary: {company_name} {year} {quarter}")
    print("=" * 80)

    if dry_run:
        print(prompt)
        return

    insight_text = call_llm(prompt)
    model_name = get_active_model_name()

    save_llm_insight(
        company_id=company_id,
        year=year,
        quarter=quarter,
        insight_type="executive_summary_swot",
        insight_text=insight_text,
        model_name=model_name,
    )

    print(insight_text[:1200])
    print("\nSaved LLM insight to llm_insights table.")


def preview_llm_insights() -> None:
    engine = get_engine()

    query = text("""
        SELECT
            c.company_name,
            li.year,
            li.quarter,
            li.insight_type,
            li.model_name,
            LEFT(li.insight_text, 300) AS insight_preview,
            li.created_at
        FROM llm_insights li
        JOIN companies c
            ON li.company_id = c.company_id
        ORDER BY li.created_at DESC
        LIMIT 10;
    """)

    with engine.connect() as conn:
        rows = conn.execute(query).fetchall()

    print("\nLLM insights preview:")
    for row in rows:
        print(row)


def export_llm_insights_to_csv() -> None:
    engine = get_engine()

    query = """
        SELECT
            c.company_name,
            li.year,
            li.quarter,
            li.insight_type,
            li.insight_text,
            li.model_name,
            li.created_at
        FROM llm_insights li
        JOIN companies c
            ON li.company_id = c.company_id
        ORDER BY c.company_name, li.year, li.quarter;
    """

    df = pd.read_sql(query, engine)
    output_path = "data/processed/llm_insights.csv"
    df.to_csv(output_path, index=False)

    print(f"\nExported LLM insights to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate LLM executive summaries and SWOT insights."
    )

    parser.add_argument("--company", type=str, default=None)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--quarter", type=str, default="Q4")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ensure_llm_insights_table()

    targets = fetch_target_periods(
        company_name=args.company,
        year=args.year,
        quarter=args.quarter,
        limit=args.limit,
    )

    if not targets:
        print("No target company/year/quarter records found.")
        return

    print(f"Found {len(targets)} target records.")

    for target in targets:
        generate_company_summary(target, dry_run=args.dry_run)

    if not args.dry_run:
        preview_llm_insights()
        export_llm_insights_to_csv()


if __name__ == "__main__":
    main()