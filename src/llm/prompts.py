from typing import Dict, List


def format_topic_context(top_topics: List[Dict]) -> str:
    if not top_topics:
        return "No topic signals available."

    lines = []

    for topic in top_topics:
        lines.append(
            f"- {topic['topic_name']}: "
            f"{topic['topic_count']} mentions, "
            f"{topic['mentions_per_10k_words']} mentions per 10K words"
        )

    return "\n".join(lines)


def format_risk_context(top_risks: List[Dict]) -> str:
    if not top_risks:
        return "No risk signals available."

    lines = []

    for risk in top_risks:
        lines.append(
            f"- {risk['risk_category']}: "
            f"{risk['frequency']} mentions; "
            f"matched keywords: {risk['risk_keyword']}"
        )

    return "\n".join(lines)


def format_quote_context(risk_quotes: List[Dict]) -> str:
    if not risk_quotes:
        return "No risk evidence quotes available."

    lines = []

    for quote in risk_quotes:
        lines.append(
            f"- Risk Category: {quote['risk_category']}\n"
            f"  Quote: {quote['example_quote']}"
        )

    return "\n".join(lines)


def format_transcript_snippets(snippets: List[Dict]) -> str:
    if not snippets:
        return "No additional transcript evidence snippets available."

    lines = []

    for snippet in snippets:
        lines.append(
            f"- Evidence Type: {snippet['evidence_type']}\n"
            f"  Matched Signal: {snippet['matched_signal']}\n"
            f"  Snippet: {snippet['snippet']}"
        )

    return "\n".join(lines)


def build_executive_summary_prompt(
    company_name: str,
    year: int,
    quarter: str,
    sentiment_score: float,
    sentiment_label: str,
    top_topics: List[Dict],
    top_risks: List[Dict],
    risk_quotes: List[Dict],
    transcript_snippets: List[Dict],
) -> str:
    topic_context = format_topic_context(top_topics)
    risk_context = format_risk_context(top_risks)
    quote_context = format_quote_context(risk_quotes)
    transcript_snippet_context = format_transcript_snippets(transcript_snippets)

    return f"""
You are a healthcare competitive intelligence analyst.

Your task is to generate an evidence-grounded executive briefing based only on the structured analytics signals and transcript evidence below.
Do not invent facts outside the provided data.
Do not claim that keyword signals prove actual business performance.
Use careful language such as "the transcript signals suggest", "management emphasized", or "risk-related mentions increased".
When possible, connect interpretations to the provided evidence snippets.

Company: {company_name}
Period: {year} {quarter}

Management Tone:
- Sentiment score: {sentiment_score}
- Sentiment label: {sentiment_label}

Top Topic Signals:
{topic_context}

Top Risk Signals:
{risk_context}

Risk Evidence Quotes:
{quote_context}

Additional Transcript Evidence Snippets:
{transcript_snippet_context}

Return the briefing in this format:

## Executive Summary
Write 3-5 concise sentences summarizing the company's transcript signals for this period.

## Key Business Themes
Write 3 bullet points. Each bullet should connect topic signals to business interpretation and reference transcript evidence when useful.

## Risk Signals to Monitor
Write 3 bullet points. Each bullet should reference the risk categories and evidence when useful.

## SWOT Snapshot
Strengths:
- 2 bullets

Weaknesses:
- 2 bullets

Opportunities:
- 2 bullets

Threats:
- 2 bullets

## Analyst Note
Write 2 sentences explaining that this is a baseline LLM-generated briefing based on NLP topic/risk signals and selected transcript evidence snippets.
""".strip()