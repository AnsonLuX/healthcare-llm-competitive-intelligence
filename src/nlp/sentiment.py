from typing import Dict, List, Union

import pandas as pd
from sqlalchemy import text
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.database import get_engine


def get_sentiment_label(compound_score: float) -> str:
    """
    Convert VADER compound score into a simple sentiment label.
    """
    if compound_score >= 0.05:
        return "positive"

    if compound_score <= -0.05:
        return "negative"

    return "neutral"


def split_text_into_chunks(text_value: str, max_chars: int = 1200) -> List[str]:
    """
    Split a long transcript into smaller chunks for more stable sentiment scoring.
    VADER compound score can saturate on very long text, so chunk-level scoring
    gives a more useful average sentiment signal.
    """
    if not text_value:
        return []

    paragraphs = [paragraph.strip() for paragraph in text_value.split("\n") if paragraph.strip()]

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) <= max_chars:
            current_chunk += " " + paragraph
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = paragraph

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def analyze_text_sentiment(text_value: str) -> Dict[str, Union[float, str, int]]:
    """
    Analyze sentiment for one transcript using chunk-level VADER scoring.

    Instead of scoring the full transcript at once, this function splits the
    transcript into smaller chunks, calculates sentiment per chunk, and returns
    the average compound score.
    """
    analyzer = SentimentIntensityAnalyzer()

    if not text_value:
        return {
            "sentiment_score": 0.0,
            "sentiment_label": "neutral",
            "chunk_count": 0,
        }

    chunks = split_text_into_chunks(text_value)

    if not chunks:
        return {
            "sentiment_score": 0.0,
            "sentiment_label": "neutral",
            "chunk_count": 0,
        }

    chunk_scores = []

    for chunk in chunks:
        scores = analyzer.polarity_scores(chunk)
        chunk_scores.append(scores["compound"])

    average_score = sum(chunk_scores) / len(chunk_scores)

    return {
        "sentiment_score": round(average_score, 4),
        "sentiment_label": get_sentiment_label(average_score),
        "chunk_count": len(chunks),
    }


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


def save_sentiment_result(
    transcript_id: int,
    company_id: int,
    year: int,
    quarter: str,
    sentiment_score: float,
    sentiment_label: str,
) -> None:
    """
    Insert or update sentiment result into sentiment_scores table.
    """
    engine = get_engine()

    query = text("""
        INSERT INTO sentiment_scores (
            transcript_id,
            company_id,
            year,
            quarter,
            sentiment_score,
            sentiment_label
        )
        VALUES (
            :transcript_id,
            :company_id,
            :year,
            :quarter,
            :sentiment_score,
            :sentiment_label
        )
        ON CONFLICT (transcript_id)
        DO UPDATE SET
            sentiment_score = EXCLUDED.sentiment_score,
            sentiment_label = EXCLUDED.sentiment_label,
            created_at = CURRENT_TIMESTAMP;
    """)

    params = {
        "transcript_id": transcript_id,
        "company_id": company_id,
        "year": year,
        "quarter": quarter,
        "sentiment_score": sentiment_score,
        "sentiment_label": sentiment_label,
    }

    with engine.begin() as conn:
        conn.execute(query, params)


def run_sentiment_pipeline() -> None:
    """
    Run sentiment analysis for all transcripts.
    """
    transcripts = fetch_transcripts()

    if not transcripts:
        print("No transcripts found. Please run ingestion first.")
        return

    print(f"Running sentiment analysis for {len(transcripts)} transcripts...")

    for transcript in transcripts:
        result = analyze_text_sentiment(transcript["cleaned_text"])

        save_sentiment_result(
            transcript_id=transcript["transcript_id"],
            company_id=transcript["company_id"],
            year=transcript["year"],
            quarter=transcript["quarter"],
            sentiment_score=result["sentiment_score"],
            sentiment_label=result["sentiment_label"],
        )

        print(
            f"Processed sentiment: {transcript['company_name']} "
            f"{transcript['year']} {transcript['quarter']} "
            f"score={result['sentiment_score']} "
            f"label={result['sentiment_label']} "
            f"chunks={result['chunk_count']}"
        )

    print("\nSentiment pipeline completed.")


def preview_sentiment_results() -> None:
    """
    Print sentiment results for verification.
    """
    engine = get_engine()

    query = text("""
        SELECT
            c.company_name,
            s.year,
            s.quarter,
            s.sentiment_score,
            s.sentiment_label
        FROM sentiment_scores s
        JOIN companies c
            ON s.company_id = c.company_id
        ORDER BY c.company_name, s.year, s.quarter;
    """)

    with engine.connect() as conn:
        rows = conn.execute(query).fetchall()

    print("\nSentiment results:")
    for row in rows:
        print(row)


def export_sentiment_to_csv() -> None:
    """
    Export sentiment results to data/processed/sentiment_scores.csv.
    """
    engine = get_engine()

    query = """
        SELECT
            c.company_name,
            s.year,
            s.quarter,
            s.sentiment_score,
            s.sentiment_label
        FROM sentiment_scores s
        JOIN companies c
            ON s.company_id = c.company_id
        ORDER BY c.company_name, s.year, s.quarter;
    """

    df = pd.read_sql(query, engine)
    output_path = "data/processed/sentiment_scores.csv"
    df.to_csv(output_path, index=False)

    print(f"\nExported sentiment results to {output_path}")


if __name__ == "__main__":
    run_sentiment_pipeline()
    preview_sentiment_results()
    export_sentiment_to_csv()