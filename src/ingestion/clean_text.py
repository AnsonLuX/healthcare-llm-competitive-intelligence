import re


def clean_transcript_text(raw_text: str) -> str:
    """
    Clean raw earnings call transcript text.

    This function keeps the text readable for NLP and LLM workflows,
    while removing unnecessary spacing and common formatting noise.
    """
    if not raw_text:
        return ""

    text = raw_text

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove excessive whitespace on each line
    lines = [line.strip() for line in text.split("\n")]

    # Remove empty lines caused by formatting noise
    lines = [line for line in lines if line]

    # Join lines back with a single newline
    text = "\n".join(lines)

    # Replace multiple spaces/tabs with one space
    text = re.sub(r"[ \t]+", " ", text)

    # Replace 3+ newlines with 2 newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def normalize_text_for_nlp(text: str) -> str:
    """
    Normalize text for keyword-based NLP tasks.

    This version lowercases text and removes extra spacing.
    It should not replace the cleaned_text stored for human reading.
    """
    if not text:
        return ""

    text = text.lower()
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)

    return text.strip()