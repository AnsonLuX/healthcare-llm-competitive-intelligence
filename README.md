# Healthcare LLM Competitive Intelligence Analytics Platform

## 1. Project Overview

This project is an end-to-end AI analytics platform designed to analyze healthcare competitors’ earnings call transcripts and convert unstructured text into structured business insights.

The platform focuses on three major healthcare companies:

- Elevance Health
- CVS Health / Aetna
- UnitedHealth Group

The project uses earnings call transcripts from 2020 to 2024 to identify competitor strategy, business risks, sentiment trends, and healthcare market signals. It combines traditional analytics, NLP, LLM workflows, and interactive dashboards to support business-facing competitive intelligence analysis.

## 2. Project Goal

The main goal is to build a healthcare competitive intelligence platform that helps business stakeholders answer questions such as:

- What topics are competitors focusing on over time?
- How does sentiment change by company, year, and quarter?
- Which healthcare themes are becoming more important?
- What risk signals appear in competitor earnings calls?
- How do companies talk about Medicare Advantage, Medicaid, PBM, cost pressure, regulation, and AI/digital health?
- Can LLM workflows generate useful, evidence-based SWOT analysis and executive summaries?

## 3. Business Problem

Healthcare payers operate in a highly competitive and regulated market. Public earnings call transcripts contain valuable signals about company strategy, financial pressure, market risks, and executive priorities.

However, reading transcripts manually is time-consuming and difficult to scale.

This project automates part of that analyst workflow by transforming earnings call transcripts into structured data, extracting NLP signals, generating AI-assisted insights, and presenting the results in an interactive dashboard.

## 4. Target Users

The target users of this platform are:

- Healthcare strategy teams
- Business analysts
- Competitive intelligence analysts
- Market research teams
- Data analysts
- Executive reporting teams

## 5. Data Sources

### Primary Data

Earnings call transcripts from 2020 to 2024 for:

- Elevance Health
- CVS Health / Aetna
- UnitedHealth Group

### Optional Data Enhancement

Structured financial metrics can be added later, such as:

- Revenue
- Operating income
- EPS
- Medical cost ratio
- Membership growth
- Guidance changes

## 6. Core Project Scope

This project will be built in several stages.

The first version will focus on:

1. Transcript ingestion and cleaning
2. Data modeling in PostgreSQL
3. NLP feature extraction
4. Topic and sentiment analysis
5. LLM-based SWOT and executive summary generation
6. Interactive dashboard using Dash and Plotly

Future versions may include:

- RAG-based transcript search
- Embedding-based evidence retrieval
- Financial metric integration
- Docker deployment
- AWS deployment
- Data quality checks
- Scheduled pipeline automation

## 7. Recommended Tech Stack

### Data Processing

- Python
- Pandas
- NumPy
- Regex
- SQLAlchemy

### Database

- PostgreSQL

### NLP

- VADER or TextBlob for baseline sentiment analysis
- spaCy or NLTK for text preprocessing
- Optional: sentence-transformers for embeddings

### LLM Workflow

- Ollama for local LLM experimentation
- Gemini API or OpenAI API as optional alternatives
- Prompt templates for SWOT and executive summaries
- Optional: ChromaDB or FAISS for vector search

### Dashboard

- Dash
- Plotly
- Dash Bootstrap Components

### Deployment

Initial version:

- Local development

Future version:

- Docker
- AWS EC2
- Render
- Railway
- Hugging Face Spaces

## 8. High-Level Architecture

```text
Raw Transcript Files
        |
        v
Transcript Ingestion
        |
        v
Text Cleaning and Metadata Extraction
        |
        v
PostgreSQL Database
        |
        v
NLP Feature Extraction
        |
        v
Topic, Sentiment, and Risk Signal Tables
        |
        v
LLM Insight Generation
        |
        v
Dash + Plotly Dashboard