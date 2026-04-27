DROP TABLE IF EXISTS llm_insights CASCADE;
DROP TABLE IF EXISTS risk_signals CASCADE;
DROP TABLE IF EXISTS topic_scores CASCADE;
DROP TABLE IF EXISTS sentiment_scores CASCADE;
DROP TABLE IF EXISTS transcript_sections CASCADE;
DROP TABLE IF EXISTS transcripts CASCADE;
DROP TABLE IF EXISTS companies CASCADE;

CREATE TABLE companies (
    company_id SERIAL PRIMARY KEY,
    company_name VARCHAR(255) NOT NULL UNIQUE,
    ticker VARCHAR(20),
    industry VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE transcripts (
    transcript_id SERIAL PRIMARY KEY,
    company_id INTEGER NOT NULL REFERENCES companies(company_id) ON DELETE CASCADE,
    year INTEGER NOT NULL,
    quarter VARCHAR(10) NOT NULL,
    transcript_date DATE,
    raw_text TEXT NOT NULL,
    cleaned_text TEXT,
    source_file VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(company_id, year, quarter)
);

CREATE TABLE transcript_sections (
    section_id SERIAL PRIMARY KEY,
    transcript_id INTEGER NOT NULL REFERENCES transcripts(transcript_id) ON DELETE CASCADE,
    section_type VARCHAR(100),
    speaker_name VARCHAR(255),
    section_text TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE sentiment_scores (
    sentiment_id SERIAL PRIMARY KEY,
    transcript_id INTEGER NOT NULL REFERENCES transcripts(transcript_id) ON DELETE CASCADE,
    company_id INTEGER NOT NULL REFERENCES companies(company_id) ON DELETE CASCADE,
    year INTEGER NOT NULL,
    quarter VARCHAR(10) NOT NULL,
    sentiment_score NUMERIC(10, 4),
    sentiment_label VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(transcript_id)
);

CREATE TABLE topic_scores (
    topic_score_id SERIAL PRIMARY KEY,
    transcript_id INTEGER NOT NULL REFERENCES transcripts(transcript_id) ON DELETE CASCADE,
    company_id INTEGER NOT NULL REFERENCES companies(company_id) ON DELETE CASCADE,
    year INTEGER NOT NULL,
    quarter VARCHAR(10) NOT NULL,
    topic_name VARCHAR(255) NOT NULL,
    topic_count INTEGER DEFAULT 0,
    topic_intensity NUMERIC(10, 4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(transcript_id, topic_name)
);

CREATE TABLE risk_signals (
    risk_signal_id SERIAL PRIMARY KEY,
    transcript_id INTEGER NOT NULL REFERENCES transcripts(transcript_id) ON DELETE CASCADE,
    company_id INTEGER NOT NULL REFERENCES companies(company_id) ON DELETE CASCADE,
    year INTEGER NOT NULL,
    quarter VARCHAR(10) NOT NULL,
    risk_category VARCHAR(255),
    risk_keyword VARCHAR(255),
    frequency INTEGER DEFAULT 0,
    example_quote TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE llm_insights (
    insight_id SERIAL PRIMARY KEY,
    transcript_id INTEGER NOT NULL REFERENCES transcripts(transcript_id) ON DELETE CASCADE,
    company_id INTEGER NOT NULL REFERENCES companies(company_id) ON DELETE CASCADE,
    year INTEGER NOT NULL,
    quarter VARCHAR(10) NOT NULL,
    summary TEXT,
    strengths TEXT,
    weaknesses TEXT,
    opportunities TEXT,
    threats TEXT,
    supporting_quotes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(transcript_id)
);

INSERT INTO companies (company_name, ticker, industry)
VALUES
    ('Elevance Health', 'ELV', 'Healthcare Insurance'),
    ('CVS Health / Aetna', 'CVS', 'Healthcare Insurance and Pharmacy'),
    ('UnitedHealth Group', 'UNH', 'Healthcare Insurance and Services')
ON CONFLICT (company_name) DO NOTHING;