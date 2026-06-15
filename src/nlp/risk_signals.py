import re
from typing import Dict, List, Optional

import pandas as pd
from sqlalchemy import text

from src.database import get_engine


RISK_KEYWORDS: Dict[str, List[str]] = {
    "Cost Pressure Risk": [
        "cost pressure",
        "medical cost pressure",
        "higher medical costs",
        "medical cost trend",
        "medical cost trends",
        "cost trend",
        "cost trends",
        "higher costs",
        "elevated costs",
        "expense pressure",
        "medical expense",
        "medical expenses",
        "benefit expense",
        "benefit expenses",
        "cost of care",
        "inflationary pressure",
        "inflation",
        "mlr",
        "medical loss ratio",
    ],
    "Utilization Risk": [
        "utilization pressure",
        "higher utilization",
        "elevated utilization",
        "utilization trend",
        "utilization trends",
        "medical utilization",
        "inpatient utilization",
        "outpatient utilization",
        "procedure volume",
        "claims volume",
        "claims activity",
        "higher acuity",
        "care activity",
    ],
    "Regulatory / Policy Risk": [
        "regulatory pressure",
        "regulatory environment",
        "regulation",
        "regulatory",
        "compliance",
        "cms",
        "rate notice",
        "policy change",
        "policy changes",
        "legislation",
        "audit",
        "audits",
        "government rate",
        "reimbursement pressure",
        "reimbursement",
        "star ratings",
        "star rating",
    ],
    "Margin Pressure": [
        "margin pressure",
        "margin compression",
        "lower margin",
        "lower margins",
        "pressure on margins",
        "operating margin pressure",
        "profitability pressure",
        "earnings pressure",
        "operating income pressure",
        "headwind",
        "headwinds",
    ],
    "Membership / Enrollment Risk": [
        "membership decline",
        "membership declines",
        "membership loss",
        "membership losses",
        "lower membership",
        "enrollment decline",
        "enrollment declines",
        "member attrition",
        "attrition",
        "disenrollment",
        "redetermination",
        "redeterminations",
    ],
    "Medicaid Redetermination Risk": [
        "medicaid redetermination",
        "medicaid redeterminations",
        "redetermination",
        "redeterminations",
        "eligibility redetermination",
        "eligibility redeterminations",
        "medicaid disenrollment",
        "medicaid membership decline",
    ],
    "Medicare Advantage Risk": [
        "medicare advantage pressure",
        "medicare advantage margin",
        "medicare advantage risk",
        "ma pressure",
        "ma margin",
        "star ratings",
        "star rating",
        "cms rate notice",
        "risk adjustment",
        "risk score",
        "risk scores",
        "coding intensity",
    ],
    "PBM / Pharmacy Pressure": [
        "pbm pressure",
        "pharmacy pressure",
        "drug pricing pressure",
        "drug pricing",
        "pharmacy reimbursement",
        "specialty pharmacy pressure",
        "rebate pressure",
        "pharmaceutical pressure",
        "prescription drug cost",
        "prescription drug costs",
    ],
    "Operational Risk": [
        "operational challenge",
        "operational challenges",
        "execution risk",
        "implementation risk",
        "integration risk",
        "disruption",
        "disruptions",
        "cybersecurity",
        "cyberattack",
        "system outage",
        "technology issue",
        "technology issues",
        "labor pressure",
        "staffing pressure",
    ],
}


def normalize_text(text_value: str) -> str:
    """
    Normalize transcript text for keyword matching.
    """
    if not text_value:
        return ""

    normalized = text_value.lower()
    normalized = re.sub(r"\s+", " ", normalized)

    return normalized.strip()


def count_keyword_occurrences(text_value: str, keyword: str) -> int:
    """
    Count keyword occurrences with safer regex matching.
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


def split_into_sentences(text_value: str) -> List[str]:
    """
    Split text into simple sentence-like chunks.
    This is used to find example quotes for risk signals.
    """
    if not text_value:
        return []

    sentence_candidates = re.split(r"(?<=[.!?])\s+", text_value)

    return [sentence.strip() for sentence in sentence_candidates if sentence.strip()]


def find_example_quote(text_value: str, keywords: List[str]) -> Optional[str]:
    """
    Find one example sentence that contains any keyword from a risk category.
    """
    sentences = split_into_sentences(text_value)

    for sentence in sentences:
        normalized_sentence = normalize_text(sentence)

        for keyword in keywords:
            if count_keyword_occurrences(normalized_sentence, keyword) > 0:
                return sentence[:700]

    return None


def extract_risk_signals_for_text(text_value: str) -> List[Dict]:
    """
    Extract risk signals for one transcript.

    Returns one record per risk category.
    """
    normalized_text = normalize_text(text_value)

    risk_results = []

    for risk_category, keywords in RISK_KEYWORDS.items():
        category_frequency = 0
        matched_keywords = []

        for keyword in keywords:
            keyword_frequency = count_keyword_occurrences(normalized_text, keyword)

            if keyword_frequency > 0:
                category_frequency += keyword_frequency
                matched_keywords.append(keyword)

        example_quote = find_example_quote(text_value, matched_keywords)

        risk_results.append(
            {
                "risk_category": risk_category,
                "risk_keyword": ", ".join(matched_keywords[:10]),
                "frequency": category_frequency,
                "example_quote": example_quote,
            }
        )

    return risk_results


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


def delete_existing_risk_signals() -> None:
    """
    Clear existing risk signals before rerunning the pipeline.

    The risk_signals table does not currently have a unique constraint,
    so we truncate before inserting fresh results.
    """
    engine = get_engine()

    query = text("TRUNCATE TABLE risk_signals;")

    with engine.begin() as conn:
        conn.execute(query)


def save_risk_signal(
    transcript_id: int,
    company_id: int,
    year: int,
    quarter: str,
    risk_category: str,
    risk_keyword: str,
    frequency: int,
    example_quote: Optional[str],
) -> None:
    """
    Insert one risk signal result into risk_signals table.
    """
    engine = get_engine()

    query = text("""
        INSERT INTO risk_signals (
            transcript_id,
            company_id,
            year,
            quarter,
            risk_category,
            risk_keyword,
            frequency,
            example_quote
        )
        VALUES (
            :transcript_id,
            :company_id,
            :year,
            :quarter,
            :risk_category,
            :risk_keyword,
            :frequency,
            :example_quote
        );
    """)

    params = {
        "transcript_id": transcript_id,
        "company_id": company_id,
        "year": year,
        "quarter": quarter,
        "risk_category": risk_category,
        "risk_keyword": risk_keyword,
        "frequency": frequency,
        "example_quote": example_quote,
    }

    with engine.begin() as conn:
        conn.execute(query, params)


def run_risk_signal_pipeline() -> None:
    """
    Run risk signal extraction for all transcripts.
    """
    transcripts = fetch_transcripts()

    if not transcripts:
        print("No transcripts found. Please run ingestion first.")
        return

    print("Clearing existing risk signals...")
    delete_existing_risk_signals()

    print(f"Running risk signal extraction for {len(transcripts)} transcripts...")

    for transcript in transcripts:
        risk_results = extract_risk_signals_for_text(transcript["cleaned_text"])

        for risk_result in risk_results:
            save_risk_signal(
                transcript_id=transcript["transcript_id"],
                company_id=transcript["company_id"],
                year=transcript["year"],
                quarter=transcript["quarter"],
                risk_category=risk_result["risk_category"],
                risk_keyword=risk_result["risk_keyword"],
                frequency=risk_result["frequency"],
                example_quote=risk_result["example_quote"],
            )

        top_risks = sorted(
            risk_results,
            key=lambda item: item["frequency"],
            reverse=True,
        )[:3]

        top_risk_text = ", ".join(
            [
                f"{risk['risk_category']}={risk['frequency']}"
                for risk in top_risks
            ]
        )

        print(
            f"Processed risks: {transcript['company_name']} "
            f"{transcript['year']} {transcript['quarter']} | "
            f"Top risks: {top_risk_text}"
        )

    print("\nRisk signal extraction pipeline completed.")


def preview_risk_signal_results() -> None:
    """
    Print a preview of risk signal results.
    """
    engine = get_engine()

    query = text("""
        SELECT
            c.company_name,
            rs.year,
            rs.quarter,
            rs.risk_category,
            rs.frequency,
            rs.risk_keyword,
            LEFT(rs.example_quote, 250) AS example_quote_preview
        FROM risk_signals rs
        JOIN companies c
            ON rs.company_id = c.company_id
        ORDER BY c.company_name, rs.year, rs.quarter, rs.frequency DESC
        LIMIT 50;
    """)

    with engine.connect() as conn:
        rows = conn.execute(query).fetchall()

    print("\nRisk signal results preview:")
    for row in rows:
        print(row)


def export_risk_signals_to_csv() -> None:
    """
    Export risk signals to data/processed/risk_signals.csv.
    """
    engine = get_engine()

    query = """
        SELECT
            c.company_name,
            rs.year,
            rs.quarter,
            rs.risk_category,
            rs.risk_keyword,
            rs.frequency,
            rs.example_quote
        FROM risk_signals rs
        JOIN companies c
            ON rs.company_id = c.company_id
        ORDER BY c.company_name, rs.year, rs.quarter, rs.risk_category;
    """

    df = pd.read_sql(query, engine)
    output_path = "data/processed/risk_signals.csv"
    df.to_csv(output_path, index=False)

    print(f"\nExported risk signals to {output_path}")


if __name__ == "__main__":
    run_risk_signal_pipeline()
    preview_risk_signal_results()
    export_risk_signals_to_csv()