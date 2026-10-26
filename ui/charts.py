"""Plotly charts. Colors follow the group (fixed order), never its rank."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Categorical palette in fixed order (validated reference palette, light mode)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MUTED = "#8a8984"
TEXT = "#52514e"


def group_colors(group_names: list[str]) -> dict[str, str]:
    return {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(group_names)}


def _layout(fig: go.Figure, title: str, y_title: str, x_title: str = "") -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color="#1c1c1a"), x=0, xanchor="left"), height=420,
        margin=dict(l=10, r=10, t=56, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, title=None),
        hovermode="x unified", font=dict(color=TEXT, family="IBM Plex Sans, sans-serif", size=12),
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
    )
    fig.update_yaxes(title=y_title, ticksuffix=" %", zeroline=True, zerolinecolor=MUTED, zerolinewidth=1,
                     gridcolor="rgba(128,128,128,0.15)")
    fig.update_xaxes(title=x_title, gridcolor="rgba(128,128,128,0.08)")
    return fig


def caar_path(aar: pd.DataFrame, colors: dict[str, str], title: str) -> go.Figure:
    """Cumulative average abnormal return over event time, one line per group."""
    fig = go.Figure()
    for g, d in aar.groupby("Group", sort=False):
        d = d.sort_values("rel_day")
        fig.add_trace(go.Scatter(
            x=d["rel_day"], y=d["CAAR"] * 100, name=g, mode="lines+markers",
            line=dict(color=colors.get(g, MUTED), width=2), marker=dict(size=8, line=dict(width=2, color="white")),
            customdata=np.c_[d["AAR"] * 100, d["N"]],
            hovertemplate="%{y:.2f} % (AAR %{customdata[0]:.2f} %, N=%{customdata[1]})<extra>" + g + "</extra>",
        ))
    fig.add_vline(x=0, line=dict(color=MUTED, dash="dot", width=1), annotation_text="t=0",
                  annotation_position="top")
    return _layout(fig, title, "CAAR", "Handelstage relativ zum Event")


def caar_bars(table: pd.DataFrame, colors: dict[str, str], title: str) -> go.Figure:
    """CAAR per window and group with 95 % confidence intervals."""
    fig = go.Figure()
    for g, d in table.groupby("Group", sort=False):
        se = d["Std_CAR"] / np.sqrt(d["N"])
        ci = 1.96 * se * 100
        fig.add_trace(go.Bar(
            x=d["Window"], y=d["CAAR"] * 100, name=g, marker=dict(color=colors.get(g, MUTED), cornerradius=4),
            error_y=dict(type="data", array=ci, color=TEXT, thickness=1, width=4),
            customdata=np.c_[d["N"], d.get("p_t", pd.Series(np.nan, index=d.index))],
            hovertemplate="%{y:.2f} % (N=%{customdata[0]}, p=%{customdata[1]:.4f})<extra>" + g + "</extra>",
        ))
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08)
    return _layout(fig, title, "CAAR (95 %-KI)", "Eventfenster")


def car_histogram(cars: pd.Series, color: str, title: str) -> go.Figure:
    fig = go.Figure(go.Histogram(x=cars * 100, nbinsx=30, marker=dict(color=color, line=dict(color="white", width=2)),
                                 hovertemplate="%{x} %: %{y} Unternehmen<extra></extra>"))
    mean, med = cars.mean() * 100, cars.median() * 100
    fig.add_vline(x=mean, line=dict(color=TEXT, width=2), annotation_text=f"Mittel {mean:.2f} %")
    fig.add_vline(x=med, line=dict(color=TEXT, width=1, dash="dash"), annotation_text=f"Median {med:.2f} %",
                  annotation_position="bottom right")
    fig = _layout(fig, title, "", "CAR")
    fig.update_yaxes(title="Anzahl Unternehmen", ticksuffix="")
    fig.update_xaxes(ticksuffix=" %")
    fig.update_layout(hovermode="closest")
    return fig


def placebo_histogram(draws: pd.Series, actual: float, color: str, title: str) -> go.Figure:
    fig = go.Figure(go.Histogram(x=draws * 100, nbinsx=25, name="Placebo-CAARs",
                                 marker=dict(color=MUTED, line=dict(color="white", width=2)),
                                 hovertemplate="%{x} %: %{y} Placebos<extra></extra>"))
    fig.add_vline(x=actual * 100, line=dict(color=color, width=3),
                  annotation_text=f"Tatsächlich {actual * 100:.2f} %", annotation_font_color=TEXT)
    fig = _layout(fig, title, "", "CAAR")
    fig.update_yaxes(title="Anzahl Placebo-Tage", ticksuffix="")
    fig.update_xaxes(ticksuffix=" %")
    fig.update_layout(hovermode="closest", showlegend=False)
    return fig
