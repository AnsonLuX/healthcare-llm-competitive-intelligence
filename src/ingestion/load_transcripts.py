import argparse
from pathlib import Path
from typing import Dict, List, Optional

from sqlalchemy import text

from src.database import get_engine
from src.ingestion.clean_text import clean_transcript_text


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


COMPANY_FOLDER_TO_DB_NAME: Dict[str, str] = {
    "elevance": "Elevance Health",
    "cvs_aetna": "CVS Health / Aetna",
    "unitedhealth": "UnitedHealth Group",
}


def parse_transcript_filename(file_path: Path) -> Optional[Dict[str, str]]:
    """
    Parse transcript metadata from filename.

    Supported filename examples:
    - elevance_2020_Q1.txt
    - Elevance 2020 Q1.txt
    - cvs_aetna_2021_Q4.txt
    - Aetna 2021 Q4.txt
    - unitedhealth_2024_Q2.txt
    - United 2024 Q2.txt
    """
    folder_name = file_path.parent.name
    file_stem = file_path.stem

    if folder_name not in COMPANY_FOLDER_TO_DB_NAME:
        print(f"Skipping file with unsupported company folder: {file_path}")
        return None

    company_name = COMPANY_FOLDER_TO_DB_NAME[folder_name]

    normalized_stem = file_stem.replace("-", "_").replace(" ", "_")
    parts = [part for part in normalized_stem.split("_") if part]

    if len(parts) < 3:
        print(f"Skipping file with invalid filename format: {file_path.name}")
        return None

    year = None
    quarter = None

    for part in parts:
        if part.isdigit() and len(part) == 4:
            year = int(part)

        normalized_part = part.upper()
        if normalized_part.startswith("Q") and len(normalized_part) == 2 and normalized_part[1].isdigit():
            quarter = normalized_part

    if year is None or quarter is None:
        print(f"Skipping file because year or quarter was not found: {file_path.name}")
        return None

    return {
        "company_name": company_name,
        "year": year,
        "quarter": quarter,
        "source_file": str(file_path.relative_to(PROJECT_ROOT)),
    }


def get_company_id(company_name: str) -> int:
    """
    Fetch company_id from companies table.
    """
    engine = get_engine()

    query = text("""
        SELECT company_id
        FROM companies
        WHERE company_name = :company_name;
    """)

    with engine.connect() as conn:
        result = conn.execute(query, {"company_name": company_name}).fetchone()

    if result is None:
        raise ValueError(f"Company not found in database: {company_name}")

    return result[0]


def load_single_transcript(file_path: Path) -> bool:
    """
    Load one transcript file into PostgreSQL.
    """
    metadata = parse_transcript_filename(file_path)

    if metadata is None:
        return False

    raw_text = file_path.read_text(encoding="utf-8", errors="ignore")
    cleaned_text = clean_transcript_text(raw_text)

    if not cleaned_text:
        print(f"Skipping empty transcript: {file_path}")
        return False

    company_id = get_company_id(metadata["company_name"])
    engine = get_engine()

    query = text("""
        INSERT INTO transcripts (
            company_id,
            year,
            quarter,
            raw_text,
            cleaned_text,
            source_file
        )
        VALUES (
            :company_id,
            :year,
            :quarter,
            :raw_text,
            :cleaned_text,
            :source_file
        )
        ON CONFLICT (company_id, year, quarter)
        DO UPDATE SET
            raw_text = EXCLUDED.raw_text,
            cleaned_text = EXCLUDED.cleaned_text,
            source_file = EXCLUDED.source_file;
    """)

    params = {
        "company_id": company_id,
        "year": metadata["year"],
        "quarter": metadata["quarter"],
        "raw_text": raw_text,
        "cleaned_text": cleaned_text,
        "source_file": metadata["source_file"],
    }

    with engine.begin() as conn:
        conn.execute(query, params)

    print(
        f"Loaded transcript: {metadata['company_name']} "
        f"{metadata['year']} {metadata['quarter']}"
    )

    return True


def find_transcript_files() -> List[Path]:
    """
    Find all .txt transcript files under data/raw.
    """
    return sorted(RAW_DATA_DIR.glob("*/*.txt"))


def load_all_transcripts() -> None:
    """
    Load all transcript files from data/raw into PostgreSQL.
    """
    transcript_files = find_transcript_files()

    if not transcript_files:
        print(f"No transcript files found under: {RAW_DATA_DIR}")
        return

    loaded_count = 0

    for file_path in transcript_files:
        was_loaded = load_single_transcript(file_path)

        if was_loaded:
            loaded_count += 1

    print(f"\nFinished loading transcripts. Total loaded: {loaded_count}")


def preview_loaded_transcripts() -> None:
    """
    Print loaded transcripts from PostgreSQL for verification.
    """
    engine = get_engine()

    query = text("""
        SELECT
            t.transcript_id,
            c.company_name,
            t.year,
            t.quarter,
            t.source_file,
            LENGTH(t.cleaned_text) AS cleaned_text_length
        FROM transcripts t
        JOIN companies c
            ON t.company_id = c.company_id
        ORDER BY c.company_name, t.year, t.quarter;
    """)

    with engine.connect() as conn:
        rows = conn.execute(query).fetchall()

    print("\nLoaded transcript records:")
    for row in rows:
        print(row)


def test_single_transcript(file_path: Path) -> None:
    """
    Test parsing, cleaning, and loading for one transcript file.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Transcript file not found: {file_path}")

    metadata = parse_transcript_filename(file_path)
    if metadata is None:
        print("Metadata parsing failed.")
        return

    raw_text = file_path.read_text(encoding="utf-8", errors="ignore")
    cleaned_text = clean_transcript_text(raw_text)

    print("\nSingle transcript test preview:")
    print(f"File: {file_path}")
    print(f"Company: {metadata['company_name']}")
    print(f"Year: {metadata['year']}")
    print(f"Quarter: {metadata['quarter']}")
    print(f"Raw text length: {len(raw_text)}")
    print(f"Cleaned text length: {len(cleaned_text)}")
    print("\nCleaned text preview:")
    print(cleaned_text[:1000])

    should_load = input("\nLoad this transcript into PostgreSQL? Type y to continue: ").strip().lower()
    if should_load == "y":
        load_single_transcript(file_path)
        preview_loaded_transcripts()
    else:
        print("Skipped database load.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load earnings call transcripts into PostgreSQL.")
    parser.add_argument(
        "--file",
        type=str,
        help="Optional path to a single transcript file for testing.",
    )

    args = parser.parse_args()

    if args.file:
        test_file_path = Path(args.file)
        if not test_file_path.is_absolute():
            test_file_path = PROJECT_ROOT / test_file_path
        test_single_transcript(test_file_path)
    else:
        load_all_transcripts()
        preview_loaded_transcripts()