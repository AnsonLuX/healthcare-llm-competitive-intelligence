import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, dash_table, Input, Output
import dash_bootstrap_components as dbc

from src.database import get_engine


GRAPH_CONFIG = {"displayModeBar": False, "responsive": True}


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


def load_llm_insights_data() -> pd.DataFrame:
    engine = get_engine()

    query = """
        SELECT
            c.company_name,
            li.year,
            li.quarter,
            CONCAT(li.year, ' ', li.quarter) AS period,
            li.insight_type,
            li.insight_text,
            li.model_name,
            li.created_at
        FROM llm_insights li
        JOIN companies c
            ON li.company_id = c.company_id
        ORDER BY li.year DESC, li.quarter DESC, c.company_name;
    """

    return pd.read_sql(query, engine)


def load_llm_evaluation_summary() -> dict:
    """
    Load aggregate LLM evaluation metrics for dashboard monitoring.
    """
    engine = get_engine()

    query = """
        SELECT
            COUNT(*) AS total_evaluated_insights,
            ROUND(AVG(format_compliance_score), 2) AS avg_format_compliance,
            ROUND(AVG(evidence_grounded_score), 2) AS avg_evidence_grounding,
            ROUND(AVG(topic_alignment_score), 2) AS avg_topic_alignment,
            ROUND(AVG(risk_alignment_score), 2) AS avg_risk_alignment,
            ROUND(AVG(source_traceability_score), 2) AS avg_source_traceability,
            ROUND(AVG(evidence_quote_count), 2) AS avg_evidence_quotes,
            ROUND(AVG(business_relevance_score), 2) AS avg_business_relevance,
            ROUND(AVG(hallucination_risk_score), 2) AS avg_low_hallucination_flags,
            ROUND(AVG(overall_quality_score), 2) AS avg_overall_quality,
            COALESCE(
                SUM(
                    CASE
                        WHEN overall_quality_score < 3.5
                          OR topic_alignment_score < 3.0
                          OR risk_alignment_score < 3.0
                          OR source_traceability_score < 4.0
                        THEN 1
                        ELSE 0
                    END
                ),
                0
            ) AS manual_review_count
        FROM llm_evaluations;
    """

    default_summary = {
        "total_evaluated_insights": 0,
        "avg_format_compliance": 0,
        "avg_evidence_grounding": 0,
        "avg_topic_alignment": 0,
        "avg_risk_alignment": 0,
        "avg_source_traceability": 0,
        "avg_evidence_quotes": 0,
        "avg_business_relevance": 0,
        "avg_low_hallucination_flags": 0,
        "avg_overall_quality": 0,
        "manual_review_count": 0,
    }

    try:
        df = pd.read_sql(query, engine)
    except Exception:
        return default_summary

    if df.empty:
        return default_summary

    summary = df.iloc[0].fillna(0).to_dict()
    return {**default_summary, **summary}


def load_low_quality_insights() -> pd.DataFrame:
    """
    Load LLM evaluation records that need manual review.
    """
    engine = get_engine()

    query = """
        SELECT
            company_name,
            year,
            quarter,
            topic_alignment_score,
            risk_alignment_score,
            source_traceability_score,
            evidence_quote_count,
            overall_quality_score,
            CASE
                WHEN source_traceability_score < 4.0
                    THEN 'Weak source traceability: missing topic, risk, or evidence quote support'
                WHEN risk_alignment_score < 3.0
                    THEN 'Weak risk alignment: LLM briefing does not fully reflect top source risk signals'
                WHEN topic_alignment_score < 3.0
                    THEN 'Weak topic alignment: LLM briefing does not fully reflect top source topic signals'
                WHEN overall_quality_score < 3.5
                    THEN 'Low overall quality score: review briefing structure and source alignment'
                ELSE 'Manual review recommended'
            END AS review_reason
        FROM llm_evaluations
        WHERE overall_quality_score < 3.5
           OR topic_alignment_score < 3.0
           OR risk_alignment_score < 3.0
           OR source_traceability_score < 4.0
        ORDER BY overall_quality_score ASC, risk_alignment_score ASC;
    """

    try:
        return pd.read_sql(query, engine)
    except Exception:
        return pd.DataFrame(
            columns=[
                "company_name",
                "year",
                "quarter",
                "topic_alignment_score",
                "risk_alignment_score",
                "source_traceability_score",
                "evidence_quote_count",
                "overall_quality_score",
                "review_reason",
            ]
        )


def create_metric_card(title: str, value: str, subtitle: str = ""):
    return dbc.Card(
        dbc.CardBody(
            [
                html.H6(title, className="text-muted"),
                html.H3(value),
                html.P(subtitle, className="text-muted", style={"fontSize": "0.9rem"}),
            ]
        ),
        className="shadow-sm h-100",
    )


def style_figure(fig):
    """
    Apply consistent dashboard styling to Plotly figures.
    """
    fig.update_layout(
        template="plotly_white",
        margin={"l": 40, "r": 20, "t": 70, "b": 40},
        title={"x": 0.01, "xanchor": "left"},
        legend_title_text="",
        font={"family": "Arial"},
    )
    return fig


def create_ai_briefing_cards(filtered_df: pd.DataFrame):
    """
    Create AI briefing cards from filtered LLM insight records.
    """
    if filtered_df.empty:
        return [
            dbc.Alert(
                "No AI briefing records match the selected filters.",
                color="warning",
            )
        ]

    cards = []

    for insight in filtered_df.to_dict("records"):
        cards.append(
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div(
                            [
                                html.H5(
                                    f"{insight['company_name']} - {insight['period']}",
                                    className="mb-1",
                                ),
                                html.P(
                                    f"Model: {insight['model_name']} | Generated: {insight['created_at']}",
                                    className="text-muted mb-3",
                                    style={"fontSize": "0.9rem"},
                                ),
                            ]
                        ),
                        dcc.Markdown(
                            insight["insight_text"],
                            style={
                                "whiteSpace": "pre-wrap",
                                "lineHeight": "1.6",
                                "fontSize": "0.95rem",
                            },
                        ),
                    ]
                ),
                className="shadow-sm mb-4",
            )
        )

    return cards


def build_dashboard():
    summary_df = load_summary_metrics()
    sentiment_df = load_sentiment_data()
    topic_df = load_topic_data()
    risk_df = load_risk_data()
    llm_insights_df = load_llm_insights_data()
    llm_eval_summary = load_llm_evaluation_summary()
    low_quality_df = load_low_quality_insights()

    total_transcripts = int(summary_df["transcript_count"].sum())
    company_count = summary_df["company_name"].nunique()
    avg_tone = round(summary_df["avg_management_tone"].mean(), 4)

    summary_df["avg_management_tone"] = summary_df["avg_management_tone"].round(4)

    top_topic_df = (
        topic_df.groupby(["company_name", "topic_name"], as_index=False)
        .agg(
            total_topic_count=("topic_count", "sum"),
            avg_mentions_per_10k_words=("mentions_per_10k_words", "mean"),
        )
        .sort_values(["company_name", "total_topic_count"], ascending=[True, False])
    )
    top_topic_df["avg_mentions_per_10k_words"] = top_topic_df[
        "avg_mentions_per_10k_words"
    ].round(2)

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

    ai_company_options = [
        {"label": "All Companies", "value": "all"}
    ] + [
        {"label": company, "value": company}
        for company in sorted(llm_insights_df["company_name"].dropna().unique())
    ]

    ai_period_options = [
        {"label": "All Periods", "value": "all"}
    ] + [
        {"label": period, "value": period}
        for period in sorted(llm_insights_df["period"].dropna().unique(), reverse=True)
    ]

    default_ai_company = "all"
    default_ai_period = "all"

    sentiment_fig = px.line(
        sentiment_df,
        x="period",
        y="sentiment_score",
        color="company_name",
        markers=True,
        title="Management Tone Score by Company and Quarter",
        labels={
            "period": "Quarter",
            "sentiment_score": "Management Tone Score",
            "company_name": "Company",
        },
    )
    sentiment_fig.update_yaxes(range=[0.75, 1.0])
    sentiment_fig = style_figure(sentiment_fig)

    topic_bar_fig = px.bar(
        top_topic_df,
        x="avg_mentions_per_10k_words",
        y="topic_name",
        color="company_name",
        facet_col="company_name",
        orientation="h",
        title="Average Topic Mentions per 10K Words by Company",
        labels={
            "topic_name": "Topic",
            "avg_mentions_per_10k_words": "Avg Mentions per 10K Words",
            "company_name": "Company",
        },
    )
    topic_bar_fig.update_yaxes(matches=None, autorange="reversed")
    topic_bar_fig.for_each_annotation(lambda annotation: annotation.update(text=annotation.text.split("=")[-1]))
    topic_bar_fig = style_figure(topic_bar_fig)

    risk_bar_fig = px.bar(
        top_risk_df,
        x="total_risk_mentions",
        y="risk_category",
        color="company_name",
        facet_col="company_name",
        orientation="h",
        title="Total Risk Signal Mentions by Company",
        labels={
            "risk_category": "Risk Category",
            "total_risk_mentions": "Total Mentions",
            "company_name": "Company",
        },
    )
    risk_bar_fig.update_yaxes(matches=None, autorange="reversed")
    risk_bar_fig.for_each_annotation(lambda annotation: annotation.update(text=annotation.text.split("=")[-1]))
    risk_bar_fig = style_figure(risk_bar_fig)

    topic_heatmap_df = (
        topic_df.groupby(["company_name", "topic_name"], as_index=False)
        .agg(avg_mentions_per_10k_words=("mentions_per_10k_words", "mean"))
    )
    topic_heatmap_df["avg_mentions_per_10k_words"] = topic_heatmap_df[
        "avg_mentions_per_10k_words"
    ].round(2)

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
    topic_heatmap_fig = style_figure(topic_heatmap_fig)

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
                                        dcc.Graph(figure=sentiment_fig, config=GRAPH_CONFIG),
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
                                    {
                                        "name": "Avg Management Tone",
                                        "id": "avg_management_tone",
                                        "type": "numeric",
                                        "format": {"specifier": ".4f"},
                                    },
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
                            dcc.Graph(figure=topic_bar_fig, config=GRAPH_CONFIG),
                            html.Br(),
                            dcc.Graph(figure=topic_heatmap_fig, config=GRAPH_CONFIG),
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
                                        "type": "numeric",
                                        "format": {"specifier": ".2f"},
                                    },
                                ],
                                page_size=13,
                                sort_action="native",
                                filter_action="native",
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
                            dcc.Graph(figure=risk_bar_fig, config=GRAPH_CONFIG),
                            html.Br(),
                            html.H5("Risk Signal Evidence Quotes"),
                            html.P(
                                "Showing the most recent high-frequency risk examples with matched keywords and transcript evidence.",
                                className="text-muted",
                            ),
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
                                page_size=8,
                                sort_action="native",
                                filter_action="native",
                                style_table={
                                    "overflowX": "auto",
                                    "overflowY": "auto",
                                    "maxHeight": "620px",
                                },
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
                                    "position": "sticky",
                                    "top": 0,
                                    "zIndex": 1,
                                },
                            ),
                        ],
                    ),

                    dcc.Tab(
                        label="AI Briefing / SWOT",
                        children=[
                            html.Br(),
                            html.H5("LLM-Generated Executive Briefings"),
                            html.P(
                                "These briefings are generated from structured NLP outputs and selected transcript evidence snippets. "
                                "They are designed to support competitive intelligence review, not to replace analyst judgment.",
                                className="text-muted",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.Label("Company", className="fw-bold"),
                                            dcc.Dropdown(
                                                id="ai-company-filter",
                                                options=ai_company_options,
                                                value=default_ai_company,
                                                clearable=False,
                                            ),
                                        ],
                                        md=6,
                                    ),
                                    dbc.Col(
                                        [
                                            html.Label("Period", className="fw-bold"),
                                            dcc.Dropdown(
                                                id="ai-period-filter",
                                                options=ai_period_options,
                                                value=default_ai_period,
                                                clearable=False,
                                            ),
                                        ],
                                        md=6,
                                    ),
                                ],
                                className="mb-4",
                            ),
                            html.Div(
                                id="ai-briefing-container",
                                children=create_ai_briefing_cards(llm_insights_df),
                            ),
                        ],
                    ),

                    dcc.Tab(
                        label="LLM Quality Monitor",
                        children=[
                            html.Br(),
                            html.H5("LLM Insight Evaluation Monitor"),
                            html.P(
                                "This tab tracks whether generated LLM insights are complete, traceable, and aligned with source-side NLP signals. "
                                "Scores are rule-based audit proxies and do not claim factual correctness.",
                                className="text-muted",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        create_metric_card(
                                            "Evaluated Insights",
                                            str(int(llm_eval_summary["total_evaluated_insights"])),
                                            "LLM records checked",
                                        ),
                                        md=3,
                                    ),
                                    dbc.Col(
                                        create_metric_card(
                                            "Overall Quality",
                                            f"{float(llm_eval_summary['avg_overall_quality']):.2f} / 5",
                                            "Weighted audit score",
                                        ),
                                        md=3,
                                    ),
                                    dbc.Col(
                                        create_metric_card(
                                            "Source Traceability",
                                            f"{float(llm_eval_summary['avg_source_traceability']):.2f} / 5",
                                            "Topic, risk, quote availability",
                                        ),
                                        md=3,
                                    ),
                                    dbc.Col(
                                        create_metric_card(
                                            "Manual Review",
                                            str(int(llm_eval_summary["manual_review_count"])),
                                            "Records flagged for review",
                                        ),
                                        md=3,
                                    ),
                                ],
                                className="mb-4",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        create_metric_card(
                                            "Format Compliance",
                                            f"{float(llm_eval_summary['avg_format_compliance']):.2f} / 5",
                                            "Required sections present",
                                        ),
                                        md=3,
                                    ),
                                    dbc.Col(
                                        create_metric_card(
                                            "Topic Alignment",
                                            f"{float(llm_eval_summary['avg_topic_alignment']):.2f} / 5",
                                            "Matches top topic signals",
                                        ),
                                        md=3,
                                    ),
                                    dbc.Col(
                                        create_metric_card(
                                            "Risk Alignment",
                                            f"{float(llm_eval_summary['avg_risk_alignment']):.2f} / 5",
                                            "Matches top risk signals",
                                        ),
                                        md=3,
                                    ),
                                    dbc.Col(
                                        create_metric_card(
                                            "Evidence Quotes",
                                            f"{float(llm_eval_summary['avg_evidence_quotes']):.2f}",
                                            "Average quotes available",
                                        ),
                                        md=3,
                                    ),
                                ],
                                className="mb-4",
                            ),
                            html.H5("Manual Review Queue"),
                            html.P(
                                "Records below alignment or traceability thresholds are listed here for analyst review. "
                                "The table shows a short review reason instead of the full audit note to keep the dashboard readable.",
                                className="text-muted",
                            ),
                            dash_table.DataTable(
                                data=low_quality_df.to_dict("records"),
                                columns=[
                                    {"name": "Company", "id": "company_name"},
                                    {"name": "Year", "id": "year"},
                                    {"name": "Quarter", "id": "quarter"},
                                    {"name": "Topic Alignment", "id": "topic_alignment_score"},
                                    {"name": "Risk Alignment", "id": "risk_alignment_score"},
                                    {"name": "Source Traceability", "id": "source_traceability_score"},
                                    {"name": "Evidence Quotes", "id": "evidence_quote_count"},
                                    {"name": "Overall Quality", "id": "overall_quality_score"},
                                    {"name": "Review Reason", "id": "review_reason"},
                                ],
                                page_size=5,
                                sort_action="native",
                                filter_action="native",
                                style_table={"overflowX": "auto", "maxHeight": "480px", "overflowY": "auto"},
                                style_cell={
                                    "textAlign": "left",
                                    "padding": "8px",
                                    "fontFamily": "Arial",
                                    "fontSize": "13px",
                                    "maxWidth": "220px",
                                    "whiteSpace": "normal",
                                },
                                style_header={
                                    "fontWeight": "bold",
                                    "backgroundColor": "#f8f9fa",
                                },
                                style_data_conditional=[
                                    {
                                        "if": {"column_id": "review_reason"},
                                        "maxWidth": "420px",
                                        "whiteSpace": "normal",
                                    },
                                    {
                                        "if": {"filter_query": "{risk_alignment_score} < 3", "column_id": "risk_alignment_score"},
                                        "backgroundColor": "#fff3cd",
                                    },
                                    {
                                        "if": {"filter_query": "{overall_quality_score} < 3.5", "column_id": "overall_quality_score"},
                                        "backgroundColor": "#f8d7da",
                                    },
                                ],
                            ),
                        ],
                    ),
                ]
            ),

            html.Br(),
            html.Hr(),
            html.P(
                "Note: Sentiment, topic, risk, and LLM-generated briefing outputs are baseline analytics artifacts. "
                "They are designed to support business analysis, not to directly measure company performance or replace analyst judgment.",
                className="text-muted",
            ),
        ],
        fluid=True,
    )

    return app


app = build_dashboard()


@app.callback(
    Output("ai-briefing-container", "children"),
    Input("ai-company-filter", "value"),
    Input("ai-period-filter", "value"),
)
def update_ai_briefing_cards(selected_company, selected_period):
    llm_insights_df = load_llm_insights_data()
    filtered_df = llm_insights_df.copy()

    if selected_company and selected_company != "all":
        filtered_df = filtered_df[filtered_df["company_name"] == selected_company]

    if selected_period and selected_period != "all":
        filtered_df = filtered_df[filtered_df["period"] == selected_period]

    return create_ai_briefing_cards(filtered_df)


if __name__ == "__main__":
    app.run(debug=False)