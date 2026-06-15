import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, dash_table
import dash_bootstrap_components as dbc
from sqlalchemy import text

from src.database import get_engine


def load_summary_metrics() -> pd.DataFrame:
    engine = get_engine()

    query = """
        SELECT
            c.company_name,
            COUNT(t.transcript_id) AS transcript_count,
            ROUND(AVG(s.sentiment_score), 4) AS avg_management_tone
        FROM companies c
        LEFT JOIN transcripts t
            ON c.company_id = t.company_id
        LEFT JOIN sentiment_scores s
            ON t.transcript_id = s.transcript_id
        GROUP BY c.company_name
        ORDER BY c.company_name;
    """

    return pd.read_sql(query, engine)


def load_sentiment_data() -> pd.DataFrame:
    engine = get_engine()

    query = """
        SELECT
            c.company_name,
            s.year,
            s.quarter,
            CONCAT(s.year, ' ', s.quarter) AS period,
            s.sentiment_score,
            s.sentiment_label
        FROM sentiment_scores s
        JOIN companies c
            ON s.company_id = c.company_id
        ORDER BY c.company_name, s.year, s.quarter;
    """

    return pd.read_sql(query, engine)


def load_topic_data() -> pd.DataFrame:
    engine = get_engine()

    query = """
        SELECT
            c.company_name,
            ts.year,
            ts.quarter,
            CONCAT(ts.year, ' ', ts.quarter) AS period,
            ts.topic_name,
            ts.topic_count,
            ts.topic_intensity,
            ROUND(ts.topic_intensity * 10000, 2) AS mentions_per_10k_words
        FROM topic_scores ts
        JOIN companies c
            ON ts.company_id = c.company_id
        ORDER BY c.company_name, ts.year, ts.quarter, ts.topic_name;
    """

    return pd.read_sql(query, engine)


def load_risk_data() -> pd.DataFrame:
    engine = get_engine()

    query = """
        SELECT
            c.company_name,
            rs.year,
            rs.quarter,
            CONCAT(rs.year, ' ', rs.quarter) AS period,
            rs.risk_category,
            rs.frequency,
            rs.risk_keyword,
            rs.example_quote
        FROM risk_signals rs
        JOIN companies c
            ON rs.company_id = c.company_id
        ORDER BY c.company_name, rs.year, rs.quarter, rs.risk_category;
    """

    return pd.read_sql(query, engine)


def create_metric_card(title: str, value: str, subtitle: str = ""):
    return dbc.Card(
        dbc.CardBody(
            [
                html.H6(title, className="text-muted"),
                html.H3(value),
                html.P(subtitle, className="text-muted", style={"fontSize": "0.9rem"}),
            ]
        ),
        className="shadow-sm",
    )


def build_dashboard():
    summary_df = load_summary_metrics()
    sentiment_df = load_sentiment_data()
    topic_df = load_topic_data()
    risk_df = load_risk_data()

    total_transcripts = int(summary_df["transcript_count"].sum())
    company_count = summary_df["company_name"].nunique()
    avg_tone = round(summary_df["avg_management_tone"].mean(), 4)

    top_topic_df = (
        topic_df.groupby(["company_name", "topic_name"], as_index=False)
        .agg(
            total_topic_count=("topic_count", "sum"),
            avg_mentions_per_10k_words=("mentions_per_10k_words", "mean"),
        )
        .sort_values(["company_name", "total_topic_count"], ascending=[True, False])
    )

    top_risk_df = (
        risk_df.groupby(["company_name", "risk_category"], as_index=False)
        .agg(total_risk_mentions=("frequency", "sum"))
        .sort_values(["company_name", "total_risk_mentions"], ascending=[True, False])
    )

    latest_risk_quotes_df = risk_df[
        (risk_df["frequency"] > 0) & (risk_df["example_quote"].notna())
    ].copy()

    latest_risk_quotes_df = latest_risk_quotes_df.sort_values(
        ["year", "quarter", "frequency"],
        ascending=[False, False, False],
    ).head(15)

    sentiment_fig = px.line(
        sentiment_df,
        x="period",
        y="sentiment_score",
        line_group="company_name",
        markers=True,
        title="Management Tone Score by Company and Quarter",
        labels={
            "period": "Quarter",
            "sentiment_score": "Management Tone Score",
            "company_name": "Company",
        },
    )

    topic_bar_fig = px.bar(
        top_topic_df,
        x="topic_name",
        y="avg_mentions_per_10k_words",
        facet_col="company_name",
        title="Average Topic Mentions per 10K Words by Company",
        labels={
            "topic_name": "Topic",
            "avg_mentions_per_10k_words": "Avg Mentions per 10K Words",
            "company_name": "Company",
        },
    )

    risk_bar_fig = px.bar(
        top_risk_df,
        x="risk_category",
        y="total_risk_mentions",
        facet_col="company_name",
        title="Total Risk Signal Mentions by Company",
        labels={
            "risk_category": "Risk Category",
            "total_risk_mentions": "Total Mentions",
            "company_name": "Company",
        },
    )

    topic_heatmap_df = (
        topic_df.groupby(["company_name", "topic_name"], as_index=False)
        .agg(avg_mentions_per_10k_words=("mentions_per_10k_words", "mean"))
    )

    topic_heatmap_fig = px.density_heatmap(
        topic_heatmap_df,
        x="topic_name",
        y="company_name",
        z="avg_mentions_per_10k_words",
        title="Topic Intensity Heatmap",
        labels={
            "topic_name": "Topic",
            "company_name": "Company",
            "avg_mentions_per_10k_words": "Avg Mentions per 10K Words",
        },
    )

    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        title="Healthcare LLM Competitive Intelligence Dashboard",
    )

    app.layout = dbc.Container(
        [
            html.Br(),

            html.H2("Healthcare LLM Competitive Intelligence Dashboard"),
            html.P(
                "AI-powered analytics platform for healthcare earnings call transcripts. "
                "This dashboard summarizes management tone, healthcare topic trends, "
                "and risk signals across Elevance Health, CVS/Aetna, and UnitedHealth Group.",
                className="text-muted",
            ),

            html.Hr(),

            dbc.Row(
                [
                    dbc.Col(
                        create_metric_card(
                            "Total Transcripts",
                            str(total_transcripts),
                            "Loaded and processed earnings call transcripts",
                        ),
                        md=4,
                    ),
                    dbc.Col(
                        create_metric_card(
                            "Companies",
                            str(company_count),
                            "Healthcare competitors included",
                        ),
                        md=4,
                    ),
                    dbc.Col(
                        create_metric_card(
                            "Avg Management Tone",
                            str(avg_tone),
                            "Chunk-level VADER baseline score",
                        ),
                        md=4,
                    ),
                ],
                className="mb-4",
            ),

            dcc.Tabs(
                [
                    dcc.Tab(
                        label="Executive Overview",
                        children=[
                            html.Br(),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dcc.Graph(figure=sentiment_fig),
                                        md=12,
                                    ),
                                ]
                            ),
                            html.Br(),
                            html.H5("Company Summary"),
                            dash_table.DataTable(
                                data=summary_df.to_dict("records"),
                                columns=[
                                    {"name": "Company", "id": "company_name"},
                                    {"name": "Transcript Count", "id": "transcript_count"},
                                    {"name": "Avg Management Tone", "id": "avg_management_tone"},
                                ],
                                page_size=10,
                                style_table={"overflowX": "auto"},
                                style_cell={
                                    "textAlign": "left",
                                    "padding": "8px",
                                    "fontFamily": "Arial",
                                },
                                style_header={
                                    "fontWeight": "bold",
                                    "backgroundColor": "#f8f9fa",
                                },
                            ),
                        ],
                    ),

                    dcc.Tab(
                        label="Topic Comparison",
                        children=[
                            html.Br(),
                            dcc.Graph(figure=topic_bar_fig),
                            html.Br(),
                            dcc.Graph(figure=topic_heatmap_fig),
                            html.Br(),
                            html.H5("Top Topic Summary"),
                            dash_table.DataTable(
                                data=top_topic_df.to_dict("records"),
                                columns=[
                                    {"name": "Company", "id": "company_name"},
                                    {"name": "Topic", "id": "topic_name"},
                                    {"name": "Total Topic Count", "id": "total_topic_count"},
                                    {
                                        "name": "Avg Mentions per 10K Words",
                                        "id": "avg_mentions_per_10k_words",
                                    },
                                ],
                                page_size=15,
                                style_table={"overflowX": "auto"},
                                style_cell={
                                    "textAlign": "left",
                                    "padding": "8px",
                                    "fontFamily": "Arial",
                                },
                                style_header={
                                    "fontWeight": "bold",
                                    "backgroundColor": "#f8f9fa",
                                },
                            ),
                        ],
                    ),

                    dcc.Tab(
                        label="Risk Signal Monitor",
                        children=[
                            html.Br(),
                            dcc.Graph(figure=risk_bar_fig),
                            html.Br(),
                            html.H5("Risk Signal Evidence Quotes"),
                            dash_table.DataTable(
                                data=latest_risk_quotes_df[
                                    [
                                        "company_name",
                                        "year",
                                        "quarter",
                                        "risk_category",
                                        "frequency",
                                        "risk_keyword",
                                        "example_quote",
                                    ]
                                ].to_dict("records"),
                                columns=[
                                    {"name": "Company", "id": "company_name"},
                                    {"name": "Year", "id": "year"},
                                    {"name": "Quarter", "id": "quarter"},
                                    {"name": "Risk Category", "id": "risk_category"},
                                    {"name": "Frequency", "id": "frequency"},
                                    {"name": "Matched Keywords", "id": "risk_keyword"},
                                    {"name": "Example Quote", "id": "example_quote"},
                                ],
                                page_size=10,
                                style_table={"overflowX": "auto"},
                                style_cell={
                                    "textAlign": "left",
                                    "padding": "8px",
                                    "fontFamily": "Arial",
                                    "whiteSpace": "normal",
                                    "height": "auto",
                                    "maxWidth": "360px",
                                },
                                style_header={
                                    "fontWeight": "bold",
                                    "backgroundColor": "#f8f9fa",
                                },
                            ),
                        ],
                    ),
                ]
            ),

            html.Br(),
            html.Hr(),
            html.P(
                "Note: Sentiment, topic, and risk signals are baseline NLP outputs. "
                "They are designed to support business analysis, not to directly measure company performance.",
                className="text-muted",
            ),
        ],
        fluid=True,
    )

    return app


app = build_dashboard()


if __name__ == "__main__":
    app.run(debug=True)