"""
Reusable Plotly chart builders.

Each function returns a figure and makes no Streamlit calls, so the same chart
can be dropped into any section, any tab, or exported.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src import analysis as A
from src import config as C
from dashboard import theme as T

OVERALL_LINE = "Overall"


def _order(df: pd.DataFrame, col: str) -> list:
    if col in C.CATEGORY_ORDER:
        present = [c for c in C.CATEGORY_ORDER[col] if c in df[col].astype(str).unique()]
        extra = [c for c in df[col].astype(str).unique() if c not in present]
        return present + sorted(extra)
    return sorted(df[col].dropna().astype(str).unique().tolist())


def _colors_for(levels, fallback_scale=C.SEQ_TEAL):
    out = []
    for i, level in enumerate(levels):
        if str(level) in C.ANSWER_COLORS:
            out.append(C.ANSWER_COLORS[str(level)])
        else:
            out.append(fallback_scale[min(len(fallback_scale) - 1,
                                          int(1 + i * (len(fallback_scale) - 2) / max(1, len(levels) - 1)))])
    return out


# --------------------------------------------------------------------------

def donut(series: pd.Series, title: str, hole: float = 0.62, height: int = 320) -> go.Figure:
    counts = series.value_counts()
    fig = go.Figure(go.Pie(
        labels=counts.index.astype(str), values=counts.values, hole=hole,
        marker=dict(colors=_colors_for(counts.index), line=dict(color="white", width=2)),
        textinfo="label+percent", textposition="outside",
        hovertemplate="%{label}<br>%{value} responses (%{percent})<extra></extra>",
    ))
    fig.update_layout(showlegend=False)
    return T.style_fig(fig, height=height, title=title)


def count_bar(series: pd.Series, title: str, order: list | None = None,
              horizontal: bool = False, height: int = 360, color: str | None = None) -> go.Figure:
    counts = series.value_counts()
    if order:
        counts = counts.reindex([o for o in order if o in counts.index])
    colors = [color] * len(counts) if color else _colors_for(counts.index)
    total = counts.sum()
    text = [f"{v} ({v / total * 100:.0f}%)" for v in counts.values]
    if horizontal:
        fig = go.Figure(go.Bar(y=counts.index.astype(str)[::-1], x=counts.values[::-1],
                               orientation="h", marker_color=colors[::-1],
                               text=text[::-1], textposition="auto",
                               hovertemplate="%{y}: %{x} responses<extra></extra>"))
        fig.update_layout(xaxis_title="Respondents")
    else:
        fig = go.Figure(go.Bar(x=counts.index.astype(str), y=counts.values, marker_color=colors,
                               text=text, textposition="outside",
                               hovertemplate="%{x}: %{y} responses<extra></extra>"))
        fig.update_layout(yaxis_title="Respondents")
    return T.style_fig(fig, height=height, title=title)


def rate_bar(df: pd.DataFrame, col: str, title: str, min_n: int = 15,
             height: int = 380, show_ci: bool = True, horizontal: bool = False) -> go.Figure:
    """Treatment rate per level with Wilson intervals and the overall baseline."""
    rates = A.rate_by_group(df, col, min_n=min_n)
    if rates.empty:
        return T.style_fig(go.Figure(), height=height, title=title)
    order = _order(df, col)
    rates["sort"] = rates["level"].apply(lambda x: order.index(x) if x in order else 999)
    rates = rates.sort_values("sort")
    baseline = df["treatment_flag"].mean() * 100

    err = dict(type="data", symmetric=False,
               array=(rates["ci_high"] - rates["rate"]).values,
               arrayminus=(rates["rate"] - rates["ci_low"]).values,
               color=C.COLORS["ink_soft"], thickness=1.2, width=4)
    colors = [C.COLORS["teal"] if r >= baseline else C.COLORS["slate"] for r in rates["rate"]]

    if horizontal:
        fig = go.Figure(go.Bar(
            y=rates["level"][::-1], x=rates["rate"][::-1], orientation="h",
            marker_color=colors[::-1], error_x=err if show_ci else None,
            text=[f"{r:.0f}%" for r in rates["rate"]][::-1], textposition="auto",
            customdata=rates[["n", "ci_low", "ci_high"]].values[::-1],
            hovertemplate="%{y}<br>%{x:.1f}% sought treatment<br>n=%{customdata[0]}"
                          "<br>95% CI %{customdata[1]:.0f}-%{customdata[2]:.0f}%<extra></extra>"))
        fig.add_vline(x=baseline, line_dash="dash", line_color=C.COLORS["rose"], line_width=1.4)
        fig.update_layout(xaxis_title="% who sought treatment", xaxis_range=[0, 100])
    else:
        fig = go.Figure(go.Bar(
            x=rates["level"], y=rates["rate"], marker_color=colors,
            error_y=err if show_ci else None,
            text=[f"{r:.0f}%" for r in rates["rate"]], textposition="outside",
            customdata=rates[["n", "ci_low", "ci_high"]].values,
            hovertemplate="%{x}<br>%{y:.1f}% sought treatment<br>n=%{customdata[0]}"
                          "<br>95% CI %{customdata[1]:.0f}-%{customdata[2]:.0f}%<extra></extra>"))
        fig.add_hline(y=baseline, line_dash="dash", line_color=C.COLORS["rose"], line_width=1.4,
                      annotation_text=f"overall {baseline:.0f}%",
                      annotation_font_color=C.COLORS["rose"], annotation_font_size=11)
        fig.update_layout(yaxis_title="% who sought treatment", yaxis_range=[0, 105])
    return T.style_fig(fig, height=height, title=title)


def stacked_share(df: pd.DataFrame, index_col: str, stack_col: str, title: str,
                  height: int = 380, stack_order: list | None = None,
                  horizontal: bool = True) -> go.Figure:
    """100% stacked composition of stack_col within each level of index_col."""
    pivot = pd.crosstab(df[index_col].astype(str), df[stack_col].astype(str), normalize="index") * 100
    counts = df[index_col].astype(str).value_counts()
    idx_order = [o for o in _order(df, index_col) if o in pivot.index]
    pivot = pivot.reindex(idx_order)
    levels = stack_order or _order(df, stack_col)
    levels = [l for l in levels if l in pivot.columns]

    fig = go.Figure()
    for level, color in zip(levels, _colors_for(levels)):
        vals = pivot[level]
        fig.add_bar(
            **({"y": pivot.index, "x": vals, "orientation": "h"} if horizontal
               else {"x": pivot.index, "y": vals}),
            name=str(level), marker_color=color,
            text=[f"{v:.0f}%" if v >= 7 else "" for v in vals],
            textposition="inside", insidetextfont=dict(color="white", size=11),
            customdata=[[counts.get(i, 0)] for i in pivot.index],
            hovertemplate=f"%{{{'y' if horizontal else 'x'}}} — {level}<br>"
                          "%{" + ("x" if horizontal else "y") + ":.1f}% of group"
                          "<br>group n=%{customdata[0]}<extra></extra>")
    fig.update_layout(barmode="stack")
    if horizontal:
        fig.update_layout(xaxis_title="Share of group", xaxis_range=[0, 100],
                          yaxis=dict(autorange="reversed"))
    else:
        fig.update_layout(yaxis_title="Share of group", yaxis_range=[0, 100])
    return T.style_fig(fig, height=height, title=title)


def grouped_bar(pivot: pd.DataFrame, title: str, ylabel: str, height: int = 380,
                colors: list | None = None, text_fmt: str = "{:.0f}%") -> go.Figure:
    fig = go.Figure()
    palette = colors or [C.COLORS["teal"], C.COLORS["plum"], C.COLORS["ochre"], C.COLORS["sky"]]
    for i, col in enumerate(pivot.columns):
        fig.add_bar(x=pivot.index.astype(str), y=pivot[col], name=str(col),
                    marker_color=palette[i % len(palette)],
                    text=[text_fmt.format(v) if pd.notna(v) else "" for v in pivot[col]],
                    textposition="outside",
                    hovertemplate=f"%{{x}} — {col}<br>%{{y:.1f}}<extra></extra>")
    fig.update_layout(barmode="group", yaxis_title=ylabel)
    return T.style_fig(fig, height=height, title=title)


def heatmap(pivot: pd.DataFrame, title: str, colorbar: str, height: int = 420,
            text: pd.DataFrame | None = None, scale: list | None = None,
            zmid: float | None = None) -> go.Figure:
    values = pivot.values.astype(float)
    labels = text.values if text is not None else np.round(values, 1)
    fig = go.Figure(go.Heatmap(
        z=values, x=pivot.columns.astype(str), y=pivot.index.astype(str),
        colorscale=scale or C.SEQ_TEAL, zmid=zmid,
        text=labels, texttemplate="%{text}", textfont=dict(size=11),
        colorbar=dict(title=dict(text=colorbar, side="right"), thickness=12, len=0.82,
                      outlinewidth=0),
        hovertemplate="%{y} / %{x}<br>%{z:.2f}<extra></extra>",
        xgap=3, ygap=3))
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return T.style_fig(fig, height=height, title=title)


def lollipop(frame: pd.DataFrame, label_col: str, value_col: str, title: str,
             xlabel: str, height: int = 460, color_col: str | None = None,
             reference: float | None = None, value_fmt: str = "{:.2f}") -> go.Figure:
    frame = frame.iloc[::-1]
    colors = (frame[color_col] if color_col else [C.COLORS["teal"]] * len(frame))
    base = reference if reference is not None else 0
    fig = go.Figure()
    for label, value, color in zip(frame[label_col], frame[value_col], colors):
        fig.add_shape(type="line", x0=base, x1=value, y0=label, y1=label,
                      line=dict(color=color, width=2.4))
    fig.add_trace(go.Scatter(
        x=frame[value_col], y=frame[label_col], mode="markers+text",
        marker=dict(size=12, color=list(colors), line=dict(color="white", width=1.5)),
        text=[value_fmt.format(v) for v in frame[value_col]],
        textposition="middle right", textfont=dict(size=11, color=C.COLORS["ink_soft"]),
        hovertemplate="%{y}<br>" + xlabel + ": %{x:.3f}<extra></extra>"))
    if reference is not None:
        fig.add_vline(x=reference, line_color=C.COLORS["ink"], line_width=1.2)
    fig.update_layout(xaxis_title=xlabel, showlegend=False,
                      margin=dict(l=10, r=60, t=56, b=10))
    return T.style_fig(fig, height=height, title=title)


def dumbbell(frame: pd.DataFrame, label_col: str, left_col: str, right_col: str,
             title: str, xlabel: str, height: int = 360,
             left_name: str = "", right_name: str = "") -> go.Figure:
    fig = go.Figure()
    for _, row in frame.iterrows():
        fig.add_shape(type="line", x0=row[left_col], x1=row[right_col],
                      y0=row[label_col], y1=row[label_col],
                      line=dict(color=C.COLORS["line"], width=3))
    fig.add_trace(go.Scatter(x=frame[left_col], y=frame[label_col], mode="markers",
                             name=left_name or left_col,
                             marker=dict(size=13, color=C.COLORS["sky"],
                                         line=dict(color="white", width=1.5)),
                             hovertemplate="%{y}<br>" + (left_name or left_col) +
                                           ": %{x:.1f}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=frame[right_col], y=frame[label_col], mode="markers",
                             name=right_name or right_col,
                             marker=dict(size=13, color=C.COLORS["plum"],
                                         line=dict(color="white", width=1.5)),
                             hovertemplate="%{y}<br>" + (right_name or right_col) +
                                           ": %{x:.1f}%<extra></extra>"))
    fig.update_layout(xaxis_title=xlabel, yaxis=dict(autorange="reversed"))
    return T.style_fig(fig, height=height, title=title)


def line_with_band(frame: pd.DataFrame, x: str, y: str, low: str, high: str,
                   title: str, xlabel: str, ylabel: str, height: int = 360,
                   color: str | None = None) -> go.Figure:
    color = color or C.COLORS["teal"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(frame[x]) + list(frame[x])[::-1],
                             y=list(frame[high]) + list(frame[low])[::-1],
                             fill="toself", fillcolor="rgba(15,118,110,0.14)",
                             line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip",
                             showlegend=False))
    fig.add_trace(go.Scatter(x=frame[x], y=frame[y], mode="lines+markers+text",
                             line=dict(color=color, width=2.6), marker=dict(size=9),
                             text=[f"{v:.0f}%" for v in frame[y]], textposition="top center",
                             textfont=dict(size=11), showlegend=False,
                             hovertemplate="%{x}<br>%{y:.1f}%<extra></extra>"))
    fig.update_layout(xaxis_title=xlabel, yaxis_title=ylabel)
    return T.style_fig(fig, height=height, title=title)


def choropleth(df: pd.DataFrame, title: str, height: int = 460,
               min_n: int = 15) -> go.Figure:
    agg = df.groupby("Country")["treatment_flag"].agg(["mean", "count"]).reset_index()
    agg = agg[agg["count"] >= min_n]
    agg["rate"] = (agg["mean"] * 100).round(1)
    fig = go.Figure(go.Choropleth(
        locations=agg["Country"], locationmode="country names", z=agg["rate"],
        colorscale=C.SEQ_TEAL, marker_line_color="white", marker_line_width=0.6,
        colorbar=dict(title=dict(text="% treated", side="right"), thickness=12, len=0.75,
                      outlinewidth=0),
        customdata=agg["count"],
        hovertemplate="%{location}<br>%{z:.1f}% sought treatment"
                      "<br>n=%{customdata}<extra></extra>"))
    fig.update_geos(showframe=False, showcoastlines=False, projection_type="natural earth",
                    bgcolor="rgba(0,0,0,0)", landcolor="#E4E9ED", showland=True,
                    showcountries=True, countrycolor="white")
    fig.update_layout(margin=dict(l=0, r=0, t=56, b=0))
    return T.style_fig(fig, height=height, title=title)


def us_state_map(df: pd.DataFrame, title: str, height: int = 440, min_n: int = 8) -> go.Figure:
    us = df[(df["Country"] == "United States") & df["state"].notna()]
    agg = us.groupby("state")["treatment_flag"].agg(["mean", "count"]).reset_index()
    agg = agg[agg["count"] >= min_n]
    agg["rate"] = (agg["mean"] * 100).round(1)
    fig = go.Figure(go.Choropleth(
        locations=agg["state"], locationmode="USA-states", z=agg["rate"],
        colorscale=C.SEQ_TEAL, marker_line_color="white", marker_line_width=1,
        colorbar=dict(title=dict(text="% treated", side="right"), thickness=12, len=0.75,
                      outlinewidth=0),
        customdata=agg["count"],
        hovertemplate="%{location}<br>%{z:.1f}% sought treatment"
                      "<br>n=%{customdata}<extra></extra>"))
    fig.update_geos(scope="usa", bgcolor="rgba(0,0,0,0)", lakecolor="white")
    fig.update_layout(margin=dict(l=0, r=0, t=56, b=0))
    return T.style_fig(fig, height=height, title=title)


def sunburst(df: pd.DataFrame, path: list[str], title: str, height: int = 440) -> go.Figure:
    frame = df.copy()
    for col in path:
        frame[col] = frame[col].astype(str)
    fig = px.sunburst(frame, path=path, color=path[0],
                      color_discrete_map={"Yes": C.COLORS["teal"], "No": C.COLORS["slate"],
                                          "Maybe": C.COLORS["ochre"]})
    fig.update_traces(insidetextorientation="radial",
                      hovertemplate="%{label}<br>%{value} responses<extra></extra>",
                      marker=dict(line=dict(color="white", width=2)))
    return T.style_fig(fig, height=height, title=title)


def histogram_split(df: pd.DataFrame, value_col: str, split_col: str, title: str,
                    xlabel: str, height: int = 360, nbins: int = 28) -> go.Figure:
    fig = go.Figure()
    levels = _order(df, split_col)
    for level, color in zip(levels, _colors_for(levels)):
        vals = df[df[split_col].astype(str) == level][value_col].dropna()
        fig.add_trace(go.Histogram(x=vals, name=str(level), marker_color=color,
                                   opacity=0.72, nbinsx=nbins,
                                   hovertemplate=f"{level}<br>%{{x}}: %{{y}}<extra></extra>"))
    fig.update_layout(barmode="overlay", xaxis_title=xlabel, yaxis_title="Respondents")
    return T.style_fig(fig, height=height, title=title)


def box_split(df: pd.DataFrame, value_col: str, split_col: str, title: str,
              ylabel: str, height: int = 360) -> go.Figure:
    fig = go.Figure()
    levels = _order(df, split_col)
    for level, color in zip(levels, _colors_for(levels)):
        vals = df[df[split_col].astype(str) == level][value_col].dropna()
        fig.add_trace(go.Box(y=vals, name=str(level), marker_color=color,
                             boxmean=True, line=dict(width=1.6),
                             hovertemplate=f"{level}<br>%{{y}}<extra></extra>"))
    fig.update_layout(yaxis_title=ylabel, showlegend=False)
    return T.style_fig(fig, height=height, title=title)
