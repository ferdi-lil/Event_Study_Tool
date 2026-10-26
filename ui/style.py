"""Visual style: sober desktop-application look (dense, ruled, square) with modern type."""
from __future__ import annotations

import html

import streamlit as st

NAVY = "#1f3a5f"
INK = "#1c1c1a"
MUTED = "#6b6b66"
RULE = "#c9c9c3"
PANEL = "#ffffff"
POS = "#1d6b3a"
NEG = "#a3271f"

CSS = f"""
<style>
:root {{ --es-navy:{NAVY}; --es-ink:{INK}; --es-muted:{MUTED}; --es-rule:{RULE}; --es-panel:{PANEL}; }}

/* chrome */
[data-testid="stHeader"] {{ background: transparent; height: 0; }}
[data-testid="stToolbar"], [data-testid="stDecoration"], .stDeployButton {{ display: none !important; }}
.block-container {{ padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1280px; }}
[data-testid="stSidebar"] {{ border-right: 1px solid var(--es-rule); }}
[data-testid="stSidebarNav"] a span {{ font-size: 13px; }}

/* application title bar */
.es-titlebar {{ background: var(--es-navy); color: #fff; padding: 9px 14px; margin: 0 0 14px 0;
  display: flex; justify-content: space-between; align-items: baseline; border-radius: 2px; }}
.es-titlebar .app {{ font-size: 11px; letter-spacing: .14em; text-transform: uppercase; opacity: .75; }}
.es-titlebar .title {{ font-size: 18px; font-weight: 600; margin-top: 2px; }}
.es-titlebar .meta {{ font-size: 12px; opacity: .8; text-align: right; }}

/* section headings: small caps label with a rule */
.es-section {{ font-size: 11px; font-weight: 600; letter-spacing: .12em; text-transform: uppercase;
  color: var(--es-muted); border-bottom: 1px solid var(--es-rule); padding-bottom: 4px; margin: 22px 0 10px 0; }}
.es-note {{ font-size: 12.5px; color: var(--es-muted); margin: -4px 0 10px 0; }}

/* key-figure strip */
.es-kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); border: 1px solid var(--es-rule);
  background: var(--es-panel); margin-bottom: 12px; }}
.es-kpi {{ padding: 8px 12px; border-right: 1px solid var(--es-rule); }}
.es-kpi:last-child {{ border-right: none; }}
.es-kpi .l {{ font-size: 11px; color: var(--es-muted); text-transform: uppercase; letter-spacing: .08em;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.es-kpi .v {{ font-family: 'IBM Plex Mono', ui-monospace, Menlo, Consolas, monospace; font-size: 20px; font-weight: 500; margin-top: 2px; }}
.es-kpi .s {{ font-size: 11.5px; color: var(--es-muted); }}
.es-pos {{ color: {POS}; }} .es-neg {{ color: {NEG}; }}

/* status tags */
.es-tag {{ display: inline-block; font-size: 10.5px; letter-spacing: .08em; text-transform: uppercase;
  border: 1px solid var(--es-rule); padding: 0 6px; border-radius: 2px; color: var(--es-muted); background: #fafaf8; }}
.es-tag.ok {{ color: {POS}; border-color: #9cc3a9; }}
.es-tag.draft {{ color: #8a5a00; border-color: #d9c08a; }}
.es-tag.t {{ color: var(--es-navy); border-color: #9fb0c6; }}

/* list rows (dashboard) */
.es-row-name {{ font-weight: 600; font-size: 15px; }}
.es-row-meta {{ font-size: 12.5px; color: var(--es-muted); line-height: 1.55; }}

/* widgets */
.stButton button, .stDownloadButton button, [data-testid="stPopover"] button {{ border-radius: 2px; font-size: 13px; }}
[data-testid="stExpander"] details {{ border-radius: 2px; background: var(--es-panel); }}
[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius: 2px; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 0; border-bottom: 1px solid var(--es-rule); }}
.stTabs [data-baseweb="tab"] {{ padding: 6px 14px; font-size: 13px; border: 1px solid transparent; border-bottom: none; }}
.stTabs [aria-selected="true"] {{ border-color: var(--es-rule); background: var(--es-panel); }}
.stTabs [data-baseweb="tab-highlight"] {{ background: var(--es-navy); height: 2px; }}
[data-testid="stAlert"] {{ border-radius: 2px; }}
[data-testid="stAlertContainer"] {{ border-radius: 2px; }}
h1, h2, h3 {{ letter-spacing: -.01em; }}
</style>
"""


def apply() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def titlebar(title: str, meta: str = "") -> None:
    st.markdown(
        f'<div class="es-titlebar"><div><div class="app">Event Study Workbench</div>'
        f'<div class="title">{html.escape(title)}</div></div><div class="meta">{html.escape(meta)}</div></div>',
        unsafe_allow_html=True)


def section(title: str, note: str = "") -> None:
    st.markdown(f'<div class="es-section">{html.escape(title)}</div>', unsafe_allow_html=True)
    if note:
        st.markdown(f'<div class="es-note">{html.escape(note)}</div>', unsafe_allow_html=True)


def tag(text: str, kind: str = "") -> str:
    return f'<span class="es-tag {kind}">{html.escape(text)}</span>'


def kpis(items: list[tuple[str, str, str, int]]) -> None:
    """items: (label, value, subtext, sign) – sign colours the value (+1 green, -1 red, 0 neutral)."""
    cells = []
    for label, value, sub, sign in items:
        cls = "es-pos" if sign > 0 else "es-neg" if sign < 0 else ""
        cells.append(f'<div class="es-kpi"><div class="l" title="{html.escape(label)}">{html.escape(label)}</div>'
                     f'<div class="v {cls}">{html.escape(value)}</div><div class="s">{html.escape(sub)}</div></div>')
    st.markdown(f'<div class="es-kpis">{"".join(cells)}</div>', unsafe_allow_html=True)

