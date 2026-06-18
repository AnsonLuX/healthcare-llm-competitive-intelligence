import re
from datetime import datetime

import pandas as pd
from sqlalchemy import text

from src.database import get_engine


REQUIRED_SECTIONS = [
    "Executive Summary",
    "Key Business Themes",
    "Risk Signals",
    "SWOT Snapshot",
    "Analyst Note",
]

BUSINESS_KEYWORDS = [
    "medicare",
    "medicaid",
    "commercial",
    "membership",
    "enrollment",
    "utilization",
    "cost",
    "margin",
    "operating margin",
    "risk",
    "regulation",
    "policy",
    "pbm",
    "pharmacy",
    "care delivery",
    "value-based care",
    "digital",
    "ai",
    "growth",
    "pricing",
    "medical cost",
    "redetermination",
]


TOPIC_KEYWORD_MAP = {
    "Medicare / Medicare Advantage": ["medicare", "medicare advantage"],
    "Medicaid": ["medicaid", "redetermination"],
    "Commercial Insurance": ["commercial", "employer", "group insurance"],
    "PBM / Pharmacy": ["pbm", "pharmacy", "pharmaceutical", "prescription"],
    "Care Delivery": ["care delivery", "provider", "clinical", "care model"],
    "Regulation / Policy": ["regulation", "regulatory", "policy", "cms"],
    "Cost Pressure": ["cost", "medical cost", "expense", "pressure"],
    "Utilization": ["utilization", "care activity", "medical use"],
    "AI / Digital Health": ["ai", "digital", "automation", "technology"],
    "Membership / Enrollment": ["membership", "enrollment", "members"],
    "Operating Margin": ["margin", "operating margin", "earnings"],
    "Risk Adjustment": ["risk adjustment", "coding", "acuity"],
    "Value-Based Care": ["value-based", "value based", "risk-based care"],
}

RISK_KEYWORDS = [
    "risk",
    "pressure",
    "cost",
    "utilization",
    "regulatory",
    "policy",
    "margin",
    "redetermination",
    "headwind",
    "challenge",
    "uncertainty",
]


MIN_EVIDENCE_QUOTE_LENGTH = 40
MIN_INSIGHT_LENGTH = 800
EVALUATION_LIMITATION_NOTE = (
    "Rule-based audit proxy. This evaluation checks traceability, source-signal alignment, "
    "required structure, and obvious quality flags. It does not prove factual correctness."
)


def ensure_evaluation_table() -> None:
    """
    Create the llm_evaluations table if it does not already exist.
    """
    engine = get_engine()

    create_table_sql = text("""
        CREATE TABLE IF NOT EXISTS llm_evaluations (
            evaluation_id SERIAL PRIMARY KEY,
            insight_id INTEGER NOT NULL REFERENCES llm_insights(insight_id),
            company_name VARCHAR(255),
            year INTEGER,
            quarter VARCHAR(10),
            format_compliance_score NUMERIC(4, 2),
            evidence_grounded_score NUMERIC(4, 2),
            business_relevance_score NUMERIC(4, 2),
            topic_alignment_score NUMERIC(4, 2),
            risk_alignment_score NUMERIC(4, 2),
            source_traceability_score NUMERIC(4, 2),
            evidence_quote_count INTEGER,
            hallucination_risk_score NUMERIC(4, 2),
            overall_quality_score NUMERIC(4, 2),
            evaluation_notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    with engine.begin() as conn:
        conn.execute(create_table_sql)
        conn.execute(text("""
            ALTER TABLE llm_evaluations
            ADD COLUMN IF NOT EXISTS topic_alignment_score NUMERIC(4, 2),
            ADD COLUMN IF NOT EXISTS risk_alignment_score NUMERIC(4, 2),
            ADD COLUMN IF NOT EXISTS source_traceability_score NUMERIC(4, 2),
            ADD COLUMN IF NOT EXISTS evidence_quote_count INTEGER;
        """))


def load_llm_insights() -> pd.DataFrame:
    """
    Load all generated LLM insights with company metadata.
    """
    engine = get_engine()

    query = text("""
        SELECT
            li.insight_id,
            c.company_id,
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
        ORDER BY li.year, li.quarter, c.company_name;
    """)

    with engine.connect() as conn:
        return pd.read_sql(query, conn)
def load_source_signals(company_id: int, year: int, quarter: str) -> dict:
    """
    Load source-side NLP signals and transcript evidence for one company-period.

    This creates a traceability layer between the LLM output and the structured
    pipeline outputs that were used to generate the briefing.
    """
    engine = get_engine()

    topic_query = text("""
        SELECT
            ts.topic_name,
            ts.topic_count,
            ts.topic_intensity
        FROM topic_scores ts
        JOIN transcripts t
            ON ts.transcript_id = t.transcript_id
        WHERE t.company_id = :company_id
          AND t.year = :year
          AND t.quarter = :quarter
          AND ts.topic_count > 0
        ORDER BY ts.topic_count DESC, ts.topic_intensity DESC
        LIMIT 8;
    """)

    risk_query = text("""
        SELECT
            rs.risk_category,
            rs.risk_keyword,
            rs.frequency,
            rs.example_quote
        FROM risk_signals rs
        JOIN transcripts t
            ON rs.transcript_id = t.transcript_id
        WHERE t.company_id = :company_id
          AND t.year = :year
          AND t.quarter = :quarter
          AND rs.frequency > 0
        ORDER BY rs.frequency DESC
        LIMIT 8;
    """)

    params = {
        "company_id": company_id,
        "year": year,
        "quarter": quarter,
    }

    with engine.connect() as conn:
        topics_df = pd.read_sql(topic_query, conn, params=params)
        risks_df = pd.read_sql(risk_query, conn, params=params)

    return {
        "topics": topics_df.to_dict("records"),
        "risks": risks_df.to_dict("records"),
    }


def normalize_text(value: str) -> str:
    """
    Normalize text for lightweight matching.
    """
    if not value:
        return ""
    return re.sub(r"\s+", " ", str(value).lower()).strip()


def score_topic_alignment(insight_text: str, topics: list[dict]) -> tuple[float, list[str]]:
    """
    Score whether the LLM insight mentions the top structured topic signals.
    """
    if not topics:
        return 3.0, ["No source topic signals found for this company-period."]

    lower_text = normalize_text(insight_text)
    matched_topics = []
    unmatched_topics = []

    for topic in topics[:5]:
        topic_name = topic.get("topic_name", "")
        keywords = TOPIC_KEYWORD_MAP.get(topic_name, [topic_name.lower()])

        if any(keyword.lower() in lower_text for keyword in keywords):
            matched_topics.append(topic_name)
        else:
            unmatched_topics.append(topic_name)

    match_rate = len(matched_topics) / max(len(matched_topics) + len(unmatched_topics), 1)
    score = round(max(1.0, match_rate * 5), 2)

    notes = [
        f"Topic alignment matched {len(matched_topics)} of {len(matched_topics) + len(unmatched_topics)} top source topics."
    ]

    if unmatched_topics:
        notes.append(f"Unmatched source topics: {', '.join(unmatched_topics)}")

    return score, notes


def score_risk_alignment(insight_text: str, risks: list[dict]) -> tuple[float, list[str]]:
    """
    Score whether the LLM insight reflects the top structured risk signals.
    """
    if not risks:
        return 3.0, ["No source risk signals found for this company-period."]

    lower_text = normalize_text(insight_text)
    matched_risks = []
    unmatched_risks = []

    for risk in risks[:5]:
        risk_category = risk.get("risk_category", "")
        risk_keyword = risk.get("risk_keyword", "")

        category_terms = [term for term in re.split(r"[/()\-]", risk_category.lower()) if len(term.strip()) >= 4]
        candidate_terms = category_terms + [risk_keyword.lower()]

        if any(term.strip() and term.strip() in lower_text for term in candidate_terms):
            matched_risks.append(risk_category)
        else:
            unmatched_risks.append(risk_category)

    match_rate = len(matched_risks) / max(len(matched_risks) + len(unmatched_risks), 1)
    score = round(max(1.0, match_rate * 5), 2)

    notes = [
        f"Risk alignment matched {len(matched_risks)} of {len(matched_risks) + len(unmatched_risks)} top source risks."
    ]

    if unmatched_risks:
        notes.append(f"Unmatched source risks: {', '.join(unmatched_risks)}")

    return score, notes


def score_source_traceability(source_signals: dict) -> tuple[float, int, list[str]]:
    """
    Score whether source-side evidence exists for auditability.

    This does not judge whether every LLM claim is true. It checks whether the
    underlying pipeline has structured source signals and evidence quotes that
    can be audited for the same company-period.
    """
    topics = source_signals.get("topics", [])
    risks = source_signals.get("risks", [])

    evidence_quotes = [
        risk.get("example_quote", "")
        for risk in risks
        if len(str(risk.get("example_quote", "")).strip()) >= MIN_EVIDENCE_QUOTE_LENGTH
    ]

    score_components = 0
    notes = []

    if topics:
        score_components += 1
        notes.append(f"Source topic signals available: {len(topics)}.")
    else:
        notes.append("No source topic signals available.")

    if risks:
        score_components += 1
        notes.append(f"Source risk signals available: {len(risks)}.")
    else:
        notes.append("No source risk signals available.")

    if evidence_quotes:
        score_components += 1
        notes.append(f"Evidence quotes available: {len(evidence_quotes)}.")
    else:
        notes.append("No usable evidence quotes available.")

    score = round((score_components / 3) * 5, 2)

    return score, len(evidence_quotes), notes


def contains_section(text_value: str, section: str) -> bool:
    """
    Check whether a required section title appears in the LLM output.
    """
    pattern = re.escape(section.lower())
    return bool(re.search(pattern, text_value.lower()))


def count_keyword_hits(text_value: str, keywords: list[str]) -> int:
    """
    Count how many unique keywords appear in the insight text.
    """
    lower_text = text_value.lower()
    return sum(1 for keyword in keywords if keyword in lower_text)


def score_format_compliance(insight_text: str) -> tuple[float, list[str]]:
    """
    Score required section compliance on a 1-5 scale.
    """
    found_sections = [
        section for section in REQUIRED_SECTIONS
        if contains_section(insight_text, section)
    ]

    missing_sections = [
        section for section in REQUIRED_SECTIONS
        if section not in found_sections
    ]

    score = round((len(found_sections) / len(REQUIRED_SECTIONS)) * 5, 2)

    notes = []
    if missing_sections:
        notes.append(f"Missing sections: {', '.join(missing_sections)}")
    else:
        notes.append("All required sections present.")

    return score, notes


def score_business_relevance(insight_text: str) -> tuple[float, list[str]]:
    """
    Score business relevance based on healthcare and financial terminology.
    """
    keyword_hits = count_keyword_hits(insight_text, BUSINESS_KEYWORDS)

    if keyword_hits >= 12:
        score = 5.0
    elif keyword_hits >= 9:
        score = 4.0
    elif keyword_hits >= 6:
        score = 3.0
    elif keyword_hits >= 3:
        score = 2.0
    else:
        score = 1.0

    notes = [f"Business keyword hits: {keyword_hits}"]

    return score, notes


def score_evidence_grounding(
    topic_alignment_score: float,
    risk_alignment_score: float,
    source_traceability_score: float,
) -> tuple[float, list[str]]:
    """
    Score evidence grounding using source-signal alignment rather than only text keywords.

    This is still a rule-based proxy, but it is stricter than checking whether
    the LLM output simply mentions generic terms like risk, topic, or evidence.
    """
    score = round(
        (
            topic_alignment_score
            + risk_alignment_score
            + source_traceability_score
        ) / 3,
        2,
    )

    notes = [
        "Evidence grounding proxy combines topic alignment, risk alignment, and source traceability."
    ]

    return score, notes


def score_hallucination_risk(
    insight_text: str,
    source_traceability_score: float,
    topic_alignment_score: float,
    risk_alignment_score: float,
) -> tuple[float, list[str]]:
    """
    Score low apparent hallucination risk on a 1-5 scale.

    This is not a factual correctness score. It flags weak traceability,
    missing source alignment, short output, and overconfident language.
    """
    notes = []
    lower_text = normalize_text(insight_text)
    text_length = len(insight_text)
    score = 5.0

    if text_length < MIN_INSIGHT_LENGTH:
        score -= 1.5
        notes.append(f"Insight is short: {text_length} characters.")

    unsupported_phrases = [
        "guaranteed",
        "certainly",
        "without question",
        "definitely proves",
        "will definitely",
    ]

    phrase_hits = [
        phrase for phrase in unsupported_phrases
        if phrase in lower_text
    ]

    if phrase_hits:
        score -= 1.0
        notes.append(f"Potentially overconfident phrases: {', '.join(phrase_hits)}")

    if source_traceability_score < 4.0:
        score -= 1.0
        notes.append("Source traceability is below strong-audit threshold.")

    if topic_alignment_score < 3.0:
        score -= 0.75
        notes.append("Topic alignment is weak.")

    if risk_alignment_score < 3.0:
        score -= 0.75
        notes.append("Risk alignment is weak.")

    score = round(max(1.0, min(5.0, score)), 2)

    if not notes:
        notes.append("No obvious rule-based hallucination flags found.")

    notes.append(EVALUATION_LIMITATION_NOTE)

    return score, notes


def evaluate_single_insight(row: pd.Series) -> dict:
    """
    Evaluate a single LLM insight and return score fields.
    """
    insight_text = row["insight_text"] or ""

    source_signals = load_source_signals(
        company_id=int(row["company_id"]),
        year=int(row["year"]),
        quarter=row["quarter"],
    )

    format_score, format_notes = score_format_compliance(insight_text)
    relevance_score, relevance_notes = score_business_relevance(insight_text)
    topic_alignment_score, topic_alignment_notes = score_topic_alignment(
        insight_text,
        source_signals.get("topics", []),
    )
    risk_alignment_score, risk_alignment_notes = score_risk_alignment(
        insight_text,
        source_signals.get("risks", []),
    )
    source_traceability_score, evidence_quote_count, traceability_notes = score_source_traceability(
        source_signals
    )
    grounding_score, grounding_notes = score_evidence_grounding(
        topic_alignment_score=topic_alignment_score,
        risk_alignment_score=risk_alignment_score,
        source_traceability_score=source_traceability_score,
    )
    hallucination_score, hallucination_notes = score_hallucination_risk(
        insight_text=insight_text,
        source_traceability_score=source_traceability_score,
        topic_alignment_score=topic_alignment_score,
        risk_alignment_score=risk_alignment_score,
    )

    overall_score = round(
        (
            format_score * 0.20
            + relevance_score * 0.15
            + grounding_score * 0.25
            + topic_alignment_score * 0.15
            + risk_alignment_score * 0.15
            + hallucination_score * 0.10
        ),
        2,
    )

    notes = []
    notes.extend(format_notes)
    notes.extend(relevance_notes)
    notes.extend(topic_alignment_notes)
    notes.extend(risk_alignment_notes)
    notes.extend(traceability_notes)
    notes.extend(grounding_notes)
    notes.extend(hallucination_notes)

    return {
        "insight_id": int(row["insight_id"]),
        "company_name": row["company_name"],
        "year": int(row["year"]),
        "quarter": row["quarter"],
        "format_compliance_score": format_score,
        "evidence_grounded_score": grounding_score,
        "business_relevance_score": relevance_score,
        "topic_alignment_score": topic_alignment_score,
        "risk_alignment_score": risk_alignment_score,
        "source_traceability_score": source_traceability_score,
        "evidence_quote_count": evidence_quote_count,
        "hallucination_risk_score": hallucination_score,
        "overall_quality_score": overall_score,
        "evaluation_notes": " | ".join(notes),
    }


def save_evaluations(evaluations: list[dict]) -> None:
    """
    Replace old evaluations for the same insights and save new results.
    """
    if not evaluations:
        print("No evaluations to save.")
        return

    engine = get_engine()

    delete_query = text("""
        DELETE FROM llm_evaluations
        WHERE insight_id = :insight_id;
    """)

    insert_query = text("""
        INSERT INTO llm_evaluations (
            insight_id,
            company_name,
            year,
            quarter,
            format_compliance_score,
            evidence_grounded_score,
            business_relevance_score,
            topic_alignment_score,
            risk_alignment_score,
            source_traceability_score,
            evidence_quote_count,
            hallucination_risk_score,
            overall_quality_score,
            evaluation_notes,
            created_at
        )
        VALUES (
            :insight_id,
            :company_name,
            :year,
            :quarter,
            :format_compliance_score,
            :evidence_grounded_score,
            :business_relevance_score,
            :topic_alignment_score,
            :risk_alignment_score,
            :source_traceability_score,
            :evidence_quote_count,
            :hallucination_risk_score,
            :overall_quality_score,
            :evaluation_notes,
            :created_at
        );
    """)

    with engine.begin() as conn:
        for evaluation in evaluations:
            conn.execute(delete_query, {"insight_id": evaluation["insight_id"]})
            conn.execute(
                insert_query,
                {
                    **evaluation,
                    "created_at": datetime.now(),
                },
            )


def print_summary(evaluations: list[dict]) -> None:
    """
    Print evaluation summary to terminal.
    """
    if not evaluations:
        print("No evaluations generated.")
        return

    df = pd.DataFrame(evaluations)

    print("\nLLM Insight Evaluation Summary")
    print("=" * 40)
    print(f"Total evaluated insights: {len(df)}")
    print(f"Average format compliance: {df['format_compliance_score'].mean():.2f} / 5")
    print(f"Average evidence grounding proxy: {df['evidence_grounded_score'].mean():.2f} / 5")
    print(f"Average topic alignment: {df['topic_alignment_score'].mean():.2f} / 5")
    print(f"Average risk alignment: {df['risk_alignment_score'].mean():.2f} / 5")
    print(f"Average source traceability: {df['source_traceability_score'].mean():.2f} / 5")
    print(f"Average evidence quotes available: {df['evidence_quote_count'].mean():.2f}")
    print(f"Average business relevance: {df['business_relevance_score'].mean():.2f} / 5")
    print(f"Average low-hallucination flag score: {df['hallucination_risk_score'].mean():.2f} / 5")
    print(f"Average overall quality: {df['overall_quality_score'].mean():.2f} / 5")

    low_quality_df = df[
        (df["overall_quality_score"] < 3.5)
        | (df["topic_alignment_score"] < 3.0)
        | (df["risk_alignment_score"] < 3.0)
        | (df["source_traceability_score"] < 4.0)
    ]

    print(f"Low quality insights: {len(low_quality_df)}")

    if not low_quality_df.empty:
        print("\nLow quality records:")
        print(
            low_quality_df[
                [
                    "company_name",
                    "year",
                    "quarter",
                    "topic_alignment_score",
                    "risk_alignment_score",
                    "source_traceability_score",
                    "overall_quality_score",
                ]
            ].to_string(index=False)
        )


def main() -> None:
    ensure_evaluation_table()

    insights_df = load_llm_insights()

    if insights_df.empty:
        print("No LLM insights found. Run generate_summary first.")
        return

    evaluations = [
        evaluate_single_insight(row)
        for _, row in insights_df.iterrows()
    ]

    save_evaluations(evaluations)
    print_summary(evaluations)

    print("\nSaved evaluations to table: llm_evaluations")


if __name__ == "__main__":
    main()
