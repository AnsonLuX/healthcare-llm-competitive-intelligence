

import argparse
import time
from typing import Callable, List, Tuple

from src.database import test_connection
from src.ingestion.load_transcripts import load_all_transcripts, preview_loaded_transcripts
from src.nlp.sentiment import (
    export_sentiment_to_csv,
    preview_sentiment_results,
    run_sentiment_pipeline,
)
from src.nlp.topic_classifier import (
    export_topic_scores_to_csv,
    preview_topic_results,
    run_topic_pipeline,
)
from src.nlp.risk_signals import (
    export_risk_signals_to_csv,
    preview_risk_signal_results,
    run_risk_signal_pipeline,
)


PipelineStep = Tuple[str, Callable[[], None]]


def run_step(step_name: str, step_function: Callable[[], None]) -> None:
    """
    Run one pipeline step and print basic timing information.
    """
    print("\n" + "=" * 80)
    print(f"Starting step: {step_name}")
    print("=" * 80)

    start_time = time.time()
    step_function()
    end_time = time.time()

    elapsed_seconds = round(end_time - start_time, 2)
    print(f"Completed step: {step_name} in {elapsed_seconds} seconds")


def check_database_connection() -> None:
    """
    Validate that the project can connect to PostgreSQL before running the pipeline.
    """
    is_connected = test_connection()

    if not is_connected:
        raise RuntimeError("Database connection failed. Please check your .env and PostgreSQL setup.")

    print("Database connection successful.")


def run_full_pipeline() -> None:
    """
    Run the full analytics pipeline.

    Current pipeline order:
    1. Check database connection
    2. Load transcripts into PostgreSQL
    3. Run chunk-level sentiment analysis
    4. Run healthcare topic classification
    5. Run risk signal extraction
    6. Export processed analytics outputs to CSV
    """
    steps: List[PipelineStep] = [
        ("Check database connection", check_database_connection),
        ("Ingest and clean transcripts", load_all_transcripts),
        ("Preview loaded transcripts", preview_loaded_transcripts),
        ("Run chunk-level sentiment analysis", run_sentiment_pipeline),
        ("Preview sentiment results", preview_sentiment_results),
        ("Export sentiment scores to CSV", export_sentiment_to_csv),
        ("Run healthcare topic classification", run_topic_pipeline),
        ("Preview topic results", preview_topic_results),
        ("Export topic scores to CSV", export_topic_scores_to_csv),
        ("Run risk signal extraction", run_risk_signal_pipeline),
        ("Preview risk signal results", preview_risk_signal_results),
        ("Export risk signals to CSV", export_risk_signals_to_csv),
    ]

    pipeline_start = time.time()

    print("\nRunning full Healthcare LLM Competitive Intelligence analytics pipeline...")

    for step_name, step_function in steps:
        run_step(step_name, step_function)

    pipeline_end = time.time()
    total_seconds = round(pipeline_end - pipeline_start, 2)

    print("\n" + "=" * 80)
    print(f"Full pipeline completed successfully in {total_seconds} seconds.")
    print("=" * 80)


def run_nlp_only_pipeline() -> None:
    """
    Run only NLP and analytics steps after transcripts have already been loaded.
    """
    steps: List[PipelineStep] = [
        ("Check database connection", check_database_connection),
        ("Run chunk-level sentiment analysis", run_sentiment_pipeline),
        ("Export sentiment scores to CSV", export_sentiment_to_csv),
        ("Run healthcare topic classification", run_topic_pipeline),
        ("Export topic scores to CSV", export_topic_scores_to_csv),
        ("Run risk signal extraction", run_risk_signal_pipeline),
        ("Export risk signals to CSV", export_risk_signals_to_csv),
    ]

    pipeline_start = time.time()

    print("\nRunning NLP-only analytics pipeline...")

    for step_name, step_function in steps:
        run_step(step_name, step_function)

    pipeline_end = time.time()
    total_seconds = round(pipeline_end - pipeline_start, 2)

    print("\n" + "=" * 80)
    print(f"NLP-only pipeline completed successfully in {total_seconds} seconds.")
    print("=" * 80)


def run_ingestion_only_pipeline() -> None:
    """
    Run only transcript ingestion and preview loaded records.
    """
    steps: List[PipelineStep] = [
        ("Check database connection", check_database_connection),
        ("Ingest and clean transcripts", load_all_transcripts),
        ("Preview loaded transcripts", preview_loaded_transcripts),
    ]

    pipeline_start = time.time()

    print("\nRunning ingestion-only pipeline...")

    for step_name, step_function in steps:
        run_step(step_name, step_function)

    pipeline_end = time.time()
    total_seconds = round(pipeline_end - pipeline_start, 2)

    print("\n" + "=" * 80)
    print(f"Ingestion-only pipeline completed successfully in {total_seconds} seconds.")
    print("=" * 80)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for pipeline execution mode.
    """
    parser = argparse.ArgumentParser(
        description="Run the Healthcare LLM Competitive Intelligence analytics pipeline."
    )

    parser.add_argument(
        "--mode",
        choices=["full", "nlp", "ingestion"],
        default="full",
        help="Pipeline mode to run. Default is full.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "full":
        run_full_pipeline()
    elif args.mode == "nlp":
        run_nlp_only_pipeline()
    elif args.mode == "ingestion":
        run_ingestion_only_pipeline()