import re
from typing import Dict, List

import pandas as pd
from sqlalchemy import text

from src.database import get_engine


TOPIC_KEYWORDS: Dict[str, List[str]] = {
    "Medicare / Medicare Advantage": [
        "medicare advantage",
        "medicare",
        "ma plan",
        "ma plans",
        "seniors",
        "senior market",
        "cms star",
        "star ratings",
        "star rating",
        "cms rate notice",
        "cms",
    ],
    "Medicaid": [
        "medicaid",
        "redetermination",
        "redeterminations",
        "state program",
        "state programs",
        "managed medicaid",
        "dual eligible",
        "dual-eligible",
        "state-based",
        "state sponsored",
    ],
    "Commercial Insurance": [
        "commercial",
        "commercial business",
        "commercial membership",
        "commercial risk",
        "employer",
        "employer group",
        "group insurance",
        "self-insured",
        "fully insured",
        "large group",
        "small group",
    ],
    "PBM / Pharmacy": [
        "pbm",
        "pharmacy",
        "pharmacy benefit",
        "pharmacy benefits",
        "specialty pharmacy",
        "drug pricing",
        "prescription",
        "prescriptions",
        "caremark",
        "pharmaceutical",
        "pharma",
        "biosimilar",
    ],
    "Care Delivery": [
        "care delivery",
        "care model",
        "primary care",
        "virtual care",
        "home health",
        "home-based care",
        "care management",
        "clinical care",
        "patient care",
        "provider network",
        "provider networks",
        "optum health",
        "clinic",
        "clinics",
    ],
    "Regulation / Policy": [
        "regulation",
        "regulatory",
        "compliance",
        "cms",
        "policy",
        "federal",
        "state regulators",
        "legislation",
        "rate notice",
        "audit",
        "audits",
        "reimbursement",
        "reimbursements",
        "government rate",
        "regulatory environment",
    ],
    "Cost Pressure": [
        "cost pressure",
        "medical cost",
        "medical costs",
        "cost trend",
        "cost trends",
        "expense pressure",
        "inflation",
        "higher costs",
        "medical expense",
        "medical expenses",
        "cost of care",
        "benefit expense",
        "benefit expenses",
        "medical loss ratio",
        "mlr",
    ],
    "Utilization": [
        "utilization",
        "utilization trend",
        "utilization trends",
        "medical utilization",
        "higher utilization",
        "elevated utilization",
        "inpatient",
        "outpatient",
        "procedure volume",
        "claims volume",
        "claims activity",
        "care activity",
    ],
    "AI / Digital Health": [
        "artificial intelligence",
        "ai",
        "machine learning",
        "automation",
        "digital",
        "digital health",
        "technology platform",
        "data analytics",
        "analytics platform",
        "consumer experience",
        "digital experience",
        "virtual-first",
        "technology-enabled",
    ],
    "Membership / Enrollment": [
        "membership growth",
        "membership",
        "members",
        "member growth",
        "enrollment",
        "enrollments",
        "enrollees",
        "covered lives",
        "lives served",
        "member base",
    ],
    "Operating Margin": [
        "operating margin",
        "margin",
        "margins",
        "operating income",
        "operating earnings",
        "profitability",
        "earnings growth",
        "adjusted earnings",
        "operating gain",
    ],
    "Risk Adjustment": [
        "risk adjustment",
        "risk score",
        "risk scores",
        "risk coding",
        "coding intensity",
        "acuity",
        "risk pool",
        "risk model",
    ],
    "Value-Based Care": [
        "value-based care",
        "value based care",
        "value-based",
        "risk-based arrangement",
        "risk based arrangement",
        "capitation",
        "capitated",
        "accountable care",
        "quality outcomes",
        "quality measures",
        "shared savings",
    ],
}


def normalize_text(text_value: str) -> str:
    """
    Normalize text for keyword matching.
    """
    if not text_value:
        return ""

    normalized = text_value.lower()
    normalized = re.sub(r"\s+", " ", normalized)

    return normalized.strip()


def count_keyword_occurrences(text_value: str, keyword: str) -> int:
    """
    Count keyword occurrences using simple regex matching.

    For multi-word keywords, this checks phrase-level occurrence.
    For short keywords like 'ai', we use word boundaries to avoid false matches.
    """
    if not text_value or not keyword:
        return 0

    normalized_keyword = keyword.lower().strip()

    escaped_keyword = re.escape(normalized_keyword)

    if re.fullmatch(r"[a-z0-9]+", normalized_keyword):
        pattern = rf"\b{escaped_keyword}\b"
    else:
        pattern = escaped_keyword

    matches = re.findall(pattern, text_value, flags=re.IGNORECASE)

    return len(matches)


def count_words(text_value: str) -> int:
    """
    Count words in normalized text.
    """
    if not text_value:
        return 0

    return len(re.findall(r"\b\w+\b", text_value))


def classify_topics_for_text(text_value: str) -> List[Dict]:
    """
    Classify healthcare topics for one transcript using keyword counts.

    Returns one record per topic.
    """
    normalized_text = normalize_text(text_value)
    total_words = count_words(normalized_text)

    topic_results = []

    for topic_name, keywords in TOPIC_KEYWORDS.items():
        topic_count = 0

        for keyword in keywords:
            topic_count += count_keyword_occurrences(normalized_text, keyword)

        if total_words > 0:
            topic_intensity = topic_count / total_words
        else:
            topic_intensity = 0.0

        topic_results.append(
            {
                "topic_name": topic_name,
                "topic_count": topic_count,
                "topic_intensity": round(topic_intensity, 6),
                "topic_intensity_per_10k_words": round(topic_intensity * 10000, 2),
                "total_words": total_words,
            }
        )

    return topic_results


def fetch_transcripts() -> List[Dict]:
    """
    Fetch cleaned transcripts from PostgreSQL.
    """
    engine = get_engine()

    query = text("""
        SELECT
            t.transcript_id,
            t.company_id,
            c.company_name,
            t.year,
            t.quarter,
            t.cleaned_text
        FROM transcripts t
        JOIN companies c
            ON t.company_id = c.company_id
        ORDER BY c.company_name, t.year, t.quarter;
    """)

    with engine.connect() as conn:
        rows = conn.execute(query).mappings().all()

    return [dict(row) for row in rows]


def save_topic_result(
    transcript_id: int,
    company_id: int,
    year: int,
    quarter: str,
    topic_name: str,
    topic_count: int,
    topic_intensity: float,
) -> None:
    """
    Insert or update topic result into topic_scores table.
    """
    engine = get_engine()

    query = text("""
        INSERT INTO topic_scores (
            transcript_id,
            company_id,
            year,
            quarter,
            topic_name,
            topic_count,
            topic_intensity
        )
        VALUES (
            :transcript_id,
            :company_id,
            :year,
            :quarter,
            :topic_name,
            :topic_count,
            :topic_intensity
        )
        ON CONFLICT (transcript_id, topic_name)
        DO UPDATE SET
            topic_count = EXCLUDED.topic_count,
            topic_intensity = EXCLUDED.topic_intensity,
            created_at = CURRENT_TIMESTAMP;
    """)

    params = {
        "transcript_id": transcript_id,
        "company_id": company_id,
        "year": year,
        "quarter": quarter,
        "topic_name": topic_name,
        "topic_count": topic_count,
        "topic_intensity": topic_intensity,
    }

    with engine.begin() as conn:
        conn.execute(query, params)


def run_topic_pipeline() -> None:
    """
    Run topic classification for all transcripts.
    """
    transcripts = fetch_transcripts()

    if not transcripts:
        print("No transcripts found. Please run ingestion first.")
        return

    print(f"Running topic classification for {len(transcripts)} transcripts...")

    for transcript in transcripts:
        topic_results = classify_topics_for_text(transcript["cleaned_text"])

        for topic_result in topic_results:
            save_topic_result(
                transcript_id=transcript["transcript_id"],
                company_id=transcript["company_id"],
                year=transcript["year"],
                quarter=transcript["quarter"],
                topic_name=topic_result["topic_name"],
                topic_count=topic_result["topic_count"],
                topic_intensity=topic_result["topic_intensity"],
            )

        top_topics = sorted(
            topic_results,
            key=lambda item: item["topic_count"],
            reverse=True,
        )[:3]

        top_topic_text = ", ".join(
            [
                f"{topic['topic_name']}={topic['topic_count']} ({topic['topic_intensity_per_10k_words']}/10k words)"
                for topic in top_topics
            ]
        )

        print(
            f"Processed topics: {transcript['company_name']} "
            f"{transcript['year']} {transcript['quarter']} | "
            f"Top topics: {top_topic_text}"
        )

    print("\nTopic classification pipeline completed.")


def preview_topic_results() -> None:
    """
    Print a preview of topic classification results.
    """
    engine = get_engine()

    query = text("""
        SELECT
            c.company_name,
            ts.year,
            ts.quarter,
            ts.topic_name,
            ts.topic_count,
            ts.topic_intensity,
            ROUND(ts.topic_intensity * 10000, 2) AS topic_intensity_per_10k_words
        FROM topic_scores ts
        JOIN companies c
            ON ts.company_id = c.company_id
        ORDER BY c.company_name, ts.year, ts.quarter, ts.topic_count DESC
        LIMIT 50;
    """)

    with engine.connect() as conn:
        rows = conn.execute(query).fetchall()

    print("\nTopic results preview:")
    for row in rows:
        print(row)


def export_topic_scores_to_csv() -> None:
    """
    Export topic scores to data/processed/topic_scores.csv.
    """
    engine = get_engine()

    query = """
        SELECT
            c.company_name,
            ts.year,
            ts.quarter,
            ts.topic_name,
            ts.topic_count,
            ts.topic_intensity,
            ROUND(ts.topic_intensity * 10000, 2) AS topic_intensity_per_10k_words
        FROM topic_scores ts
        JOIN companies c
            ON ts.company_id = c.company_id
        ORDER BY c.company_name, ts.year, ts.quarter, ts.topic_name;
    """

    df = pd.read_sql(query, engine)
    output_path = "data/processed/topic_scores.csv"
    df.to_csv(output_path, index=False)

    print(f"\nExported topic scores to {output_path}")


if __name__ == "__main__":
    run_topic_pipeline()
    preview_topic_results()
    export_topic_scores_to_csv()