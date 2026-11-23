from pathlib import Path
import base64

import pandas as pd
import plotly.express as px
from dash import Dash, html, dcc, Input, Output

# 1. LOAD DATA

BASE_DIR = Path(__file__).resolve().parent

df = pd.read_csv(
    BASE_DIR / "cleaned_viral_videos.csv"
)

df["duration_min"] = df["duration_sec"] / 60

categories = sorted(df["category"].unique())
max_duration = int(df["duration_min"].max()) + 1

# 2. COLORS AND STYLES

BG = "#0B1220"
CARD = "#1A2639"
TEXT = "#F8FAFC"
MUTED = "#94A3B8"
BLUE = "#38BDF8"

panel_style = {
    "backgroundColor": CARD,
    "padding": "24px",
    "borderRadius": "16px"
}

# 3. HELPER FUNCTIONS

def kpi_card(label, element_id):
    return html.Div([
        html.P(
            label,
            style={
                "color": MUTED,
                "fontSize": "13px",
                "margin": "0"
            }
        ),
        html.H2(
            id=element_id,
            style={
                "color": TEXT,
                "fontSize": "32px",
                "margin": "12px 0 0"
            }
        )
    ], style={
        **panel_style,
        "flex": "1",
        "minWidth": "180px"
    })


# Display an existing PNG from the project
def image_card(title, filename, description):
    file_path = BASE_DIR / filename

    if file_path.exists():
        encoded = base64.b64encode(
            file_path.read_bytes()
        ).decode("utf-8")

        image_content = html.Img(
            src=f"data:image/png;base64,{encoded}",
            style={
                "width": "100%",
                "height": "auto",
                "display": "block",
                "borderRadius": "8px"
            }
        )
    else:
        image_content = html.P(
            f"Image not found: {filename}",
            style={"color": "#FCA5A5"}
        )

    return html.Div([
        html.H3(
            title,
            style={
                "fontSize": "19px",
                "marginTop": "0"
            }
        ),
        html.P(
            description,
            style={
                "color": MUTED,
                "fontSize": "13px"
            }
        ),
        image_content
    ], style=panel_style)


# 4. CREATE DASH APPLICATION

app = Dash(__name__)
app.title = "YouTube Viral Intelligence"

app.layout = html.Div([

    # HEADER
    html.Div([
        html.P(
            "DATA SCIENCE TOOLBOX · GROUP PROJECT",
            style={
                "color": BLUE,
                "fontSize": "14px",
                "letterSpacing": "3px",
                "fontWeight": "bold"
            }
        ),

        html.H1(
            "YouTube Viral Intelligence",
            style={
                "fontSize": "48px",
                "fontWeight": "800",
                "margin": "12px 0"
            }
        ),

        html.P(
            "What factors are associated with "
            "YouTube video virality?",
            style={
                "color": MUTED,
                "fontSize": "19px"
            }
        )
    ], style={"marginBottom": "30px"}),

    # FILTERS
    html.Div([
        html.H3("Explore the Dataset"),

        html.Div([

            html.Div([
                html.P(
                    "Video Category",
                    style={"color": MUTED}
                ),

                dcc.Dropdown(
                    id="category-filter",
                    options=[
                        {
                            "label": "All Categories",
                            "value": "All"
                        }
                    ] + [
                        {"label": c, "value": c}
                        for c in categories
                    ],
                    value="All",
                    clearable=False,
                    style={"color": "#111827"}
                )
            ], style={
                "flex": "1",
                "minWidth": "230px"
            }),

            html.Div([
                html.P(
                    "Video Duration (minutes)",
                    style={"color": MUTED}
                ),

                dcc.RangeSlider(
                    id="duration-filter",
                    min=0,
                    max=max_duration,
                    step=1,
                    value=[0, max_duration],
                    marks={
                        0: "0",
                        max_duration: str(max_duration)
                    },
                    tooltip={"placement": "bottom"}
                )
            ], style={
                "flex": "2",
                "minWidth": "250px"
            })

        ], style={
            "display": "flex",
            "gap": "30px",
            "flexWrap": "wrap"
        })
    ], style=panel_style),

    # KPI CARDS
    html.Div([
        kpi_card("TOTAL VIDEOS", "total-videos"),
        kpi_card("VIRAL RATE", "viral-rate"),
        kpi_card("VIDEO CATEGORIES", "total-categories"),
        kpi_card("AVERAGE VIEWS", "avg-views")
    ], style={
        "display": "flex",
        "flexWrap": "wrap",
        "gap": "18px",
        "marginTop": "24px"
    }),

    # INTERACTIVE ANALYSIS
    html.Div([
        html.H2(
            "Interactive Category Analysis",
            style={"fontSize": "24px"}
        ),

        html.P(
            "The chart and KPIs respond to the filters.",
            style={"color": MUTED}
        ),

        dcc.Graph(
            id="category-chart",
            config={"displaylogo": False}
        )
    ], style={
        **panel_style,
        "marginTop": "30px"
    }),

    # EDA SECTION
    html.Div([
        html.H2(
            "01 / Exploratory Data Analysis",
            style={"fontSize": "29px"}
        ),

        html.P(
            "Exploring the relationship between "
            "video characteristics and virality.",
            style={"color": MUTED}
        ),

        html.Div([
            image_card(
                "Viral Rate by Category",
                "viral_rate_by_category.png",
                "Comparison across video categories."
            ),

            image_card(
                "Viral Rate by Duration",
                "viral_rate_by_duration.png",
                "Relationship between video length and virality."
            )
        ], style={
            "display": "grid",
            "gridTemplateColumns":
                "repeat(auto-fit, minmax(350px, 1fr))",
            "gap": "20px"
        }),

        html.Div([
            image_card(
                "Viral Rate Heatmap",
                "viral_rate_heatmap.png",
                "Viral rate patterns across upload times."
            )
        ], style={"marginTop": "20px"})

    ], style={"marginTop": "40px"}),

    # MACHINE LEARNING SECTION
    html.Div([
        html.H2(
            "02 / Machine Learning",
            style={"fontSize": "29px"}
        ),

        html.P(
            "Evaluating the performance of "
            "viral video classification models.",
            style={"color": MUTED}
        ),

        html.Div([
            image_card(
                "Confusion Matrices",
                "ml_confusion_matrices.png",
                "Comparison of classification errors."
            ),

            image_card(
                "ROC Curve Comparison",
                "ml_roc_comparison.png",
                "Comparison of model discrimination performance."
            )
        ], style={
            "display": "grid",
            "gridTemplateColumns":
                "repeat(auto-fit, minmax(350px, 1fr))",
            "gap": "20px"
        })

    ], style={"marginTop": "40px"}),

    # FOOTER
    html.P(
        "Source: Synthetic YouTube Video Dataset | "
        "Viral = Top 20% by views | "
        "Results are based on simulated data.",
        style={
            "color": MUTED,
            "fontSize": "13px",
            "marginTop": "40px",
            "textAlign": "center"
        }
    )

], style={
    "backgroundColor": BG,
    "color": TEXT,
    "minHeight": "100vh",
    "padding": "40px",
    "fontFamily": "Arial, sans-serif",
    "maxWidth": "1450px",
    "margin": "auto"
})


# 5. INTERACTIVE CALLBACK

@app.callback(
    Output("total-videos", "children"),
    Output("viral-rate", "children"),
    Output("total-categories", "children"),
    Output("avg-views", "children"),
    Output("category-chart", "figure"),
    Input("category-filter", "value"),
    Input("duration-filter", "value")
)
def update_dashboard(category, duration):

    filtered = df.copy()

    if category != "All":
        filtered = filtered[
            filtered["category"] == category
        ]

    filtered = filtered[
        (filtered["duration_min"] >= duration[0]) &
        (filtered["duration_min"] <= duration[1])
    ]

    total = len(filtered)

    if total > 0:
        viral = filtered["viral"].mean() * 100
        avg_views = filtered["views"].mean()
        num_categories = filtered["category"].nunique()
    else:
        viral = 0
        avg_views = 0
        num_categories = 0

    category_data = (
        filtered.groupby("category")["viral"]
        .mean()
        .mul(100)
        .reset_index(name="viral_rate")
        .sort_values("viral_rate")
    )

    fig = px.bar(
        category_data,
        x="viral_rate",
        y="category",
        orientation="h",
        color="viral_rate",
        color_continuous_scale=[
            "#243B75",
            "#3B82F6",
            "#38BDF8"
        ],
        labels={
            "viral_rate": "Viral Rate (%)",
            "category": ""
        }
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        font_color=TEXT,
        height=480,
        margin=dict(l=15, r=20, t=15, b=40),
        coloraxis_showscale=False,
        xaxis=dict(
            range=[0, 100],
            gridcolor="#334155"
        ),
        yaxis=dict(showgrid=False)
    )

    return (
        f"{total:,}",
        f"{viral:.1f}%" if total else "N/A",
        str(num_categories),
        f"{avg_views:,.0f}" if total else "N/A",
        fig
    )

# 6. RUN

if __name__ == "__main__":
    app.run(debug=True)

