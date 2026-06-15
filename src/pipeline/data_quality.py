from typing import List, Tuple

from sqlalchemy import text

from src.database import get_engine
from src.nlp.topic_classifier import TOPIC_KEYWORDS
from src.nlp.risk_signals import RISK_KEYWORDS


QualityCheckResult = Tuple[str, bool, str]


def run_query_scalar(query: str):
    """
    Run a SQL query and return the first scalar result.
    """
    engine = get_engine()

    with engine.connect() as conn:
        result = conn.execute(text(query))
        return result.scalar()


def check_transcripts_exist() -> QualityCheckResult:
    transcript_count = run_query_scalar("SELECT COUNT(*) FROM transcripts;")

    passed = transcript_count > 0
    message = f"Transcript count: {transcript_count}"

    return ("Transcripts exist", passed, message)


def check_duplicate_transcripts() -> QualityCheckResult:
    duplicate_count = run_query_scalar("""
        SELECT COUNT(*)
        FROM (
            SELECT company_id, year, quarter, COUNT(*) AS record_count
            FROM transcripts
            GROUP BY company_id, year, quarter
            HAVING COUNT(*) > 1
        ) duplicates;
    """)

    passed = duplicate_count == 0
    message = f"Duplicate company/year/quarter records: {duplicate_count}"

    return ("No duplicate transcripts", passed, message)


def check_empty_cleaned_text() -> QualityCheckResult:
    empty_count = run_query_scalar("""
        SELECT COUNT(*)
        FROM transcripts
        WHERE cleaned_text IS NULL
           OR LENGTH(TRIM(cleaned_text)) = 0;
    """)

    passed = empty_count == 0
    message = f"Empty cleaned_text records: {empty_count}"

    return ("No empty cleaned transcript text", passed, message)


def check_short_cleaned_text() -> QualityCheckResult:
    short_count = run_query_scalar("""
        SELECT COUNT(*)
        FROM transcripts
        WHERE LENGTH(cleaned_text) < 1000;
    """)

    passed = short_count == 0
    message = f"Transcripts shorter than 1000 characters: {short_count}"

    return ("No unusually short transcripts", passed, message)


def check_sentiment_coverage() -> QualityCheckResult:
    transcript_count = run_query_scalar("SELECT COUNT(*) FROM transcripts;")
    sentiment_count = run_query_scalar("SELECT COUNT(*) FROM sentiment_scores;")

    passed = transcript_count == sentiment_count
    message = f"Transcript count: {transcript_count}, sentiment count: {sentiment_count}"

    return ("Sentiment coverage matches transcripts", passed, message)


def check_sentiment_score_range() -> QualityCheckResult:
    invalid_count = run_query_scalar("""
        SELECT COUNT(*)
        FROM sentiment_scores
        WHERE sentiment_score < -1
           OR sentiment_score > 1;
    """)

    passed = invalid_count == 0
    message = f"Invalid sentiment scores outside [-1, 1]: {invalid_count}"

    return ("Sentiment scores are within valid range", passed, message)


def check_topic_coverage() -> QualityCheckResult:
    transcript_count = run_query_scalar("SELECT COUNT(*) FROM transcripts;")
    topic_count = run_query_scalar("SELECT COUNT(*) FROM topic_scores;")

    expected_topic_count = transcript_count * len(TOPIC_KEYWORDS)

    passed = topic_count == expected_topic_count
    message = (
        f"Transcript count: {transcript_count}, "
        f"topic categories: {len(TOPIC_KEYWORDS)}, "
        f"expected topic rows: {expected_topic_count}, "
        f"actual topic rows: {topic_count}"
    )

    return ("Topic coverage matches transcripts × topic categories", passed, message)


def check_negative_topic_values() -> QualityCheckResult:
    invalid_count = run_query_scalar("""
        SELECT COUNT(*)
        FROM topic_scores
        WHERE topic_count < 0
           OR topic_intensity < 0;
    """)

    passed = invalid_count == 0
    message = f"Negative topic values: {invalid_count}"

    return ("Topic values are non-negative", passed, message)


def check_risk_signal_coverage() -> QualityCheckResult:
    transcript_count = run_query_scalar("SELECT COUNT(*) FROM transcripts;")
    risk_count = run_query_scalar("SELECT COUNT(*) FROM risk_signals;")

    expected_risk_count = transcript_count * len(RISK_KEYWORDS)

    passed = risk_count == expected_risk_count
    message = (
        f"Transcript count: {transcript_count}, "
        f"risk categories: {len(RISK_KEYWORDS)}, "
        f"expected risk rows: {expected_risk_count}, "
        f"actual risk rows: {risk_count}"
    )

    return ("Risk coverage matches transcripts × risk categories", passed, message)


def check_negative_risk_values() -> QualityCheckResult:
    invalid_count = run_query_scalar("""
        SELECT COUNT(*)
        FROM risk_signals
        WHERE frequency < 0;
    """)

    passed = invalid_count == 0
    message = f"Negative risk frequencies: {invalid_count}"

    return ("Risk frequencies are non-negative", passed, message)


def check_missing_quarters() -> QualityCheckResult:
    """
    This check does not fail the pipeline because some companies may have
    intentionally incomplete coverage. It prints missing company/year/quarter
    combinations for transparency.
    """
    engine = get_engine()

    query = text("""
        WITH expected_quarters AS (
            SELECT *
            FROM (VALUES
                (2020, 'Q1'), (2020, 'Q2'), (2020, 'Q3'), (2020, 'Q4'),
                (2021, 'Q1'), (2021, 'Q2'), (2021, 'Q3'), (2021, 'Q4'),
                (2022, 'Q1'), (2022, 'Q2'), (2022, 'Q3'), (2022, 'Q4'),
                (2023, 'Q1'), (2023, 'Q2'), (2023, 'Q3'), (2023, 'Q4'),
                (2024, 'Q1'), (2024, 'Q2'), (2024, 'Q3'), (2024, 'Q4')
            ) AS q(year, quarter)
        ),
        company_expected AS (
            SELECT
                c.company_id,
                c.company_name,
                e.year,
                e.quarter
            FROM companies c
            CROSS JOIN expected_quarters e
        )
        SELECT
            ce.company_name,
            ce.year,
            ce.quarter
        FROM company_expected ce
        LEFT JOIN transcripts t
            ON ce.company_id = t.company_id
            AND ce.year = t.year
            AND ce.quarter = t.quarter
        WHERE t.transcript_id IS NULL
        ORDER BY ce.company_name, ce.year, ce.quarter;
    """)

    with engine.connect() as conn:
        rows = conn.execute(query).fetchall()

    missing_count = len(rows)

    if missing_count > 0:
        missing_preview = ", ".join(
            [f"{row.company_name} {row.year} {row.quarter}" for row in rows[:10]]
        )
        message = f"Missing quarters: {missing_count}. Preview: {missing_preview}"
    else:
        message = "No missing quarters."

    # Missing quarters are allowed for MVP, so this always passes.
    return ("Missing quarter report", True, message)


def run_data_quality_checks() -> None:
    """
    Run all data quality checks and print a summary.
    """
    checks = [
        check_transcripts_exist,
        check_duplicate_transcripts,
        check_empty_cleaned_text,
        check_short_cleaned_text,
        check_sentiment_coverage,
        check_sentiment_score_range,
        check_topic_coverage,
        check_negative_topic_values,
        check_risk_signal_coverage,
        check_negative_risk_values,
        check_missing_quarters,
    ]

    results: List[QualityCheckResult] = []

    print("\nRunning data quality checks...")

    for check in checks:
        result = check()
        results.append(result)

        check_name, passed, message = result
        status = "PASS" if passed else "FAIL"

        print(f"[{status}] {check_name}: {message}")

    failed_checks = [result for result in results if not result[1]]

    if failed_checks:
        failed_names = ", ".join([result[0] for result in failed_checks])
        raise RuntimeError(f"Data quality checks failed: {failed_names}")

    print("\nAll required data quality checks passed.")


if __name__ == "__main__":
    run_data_quality_checks()