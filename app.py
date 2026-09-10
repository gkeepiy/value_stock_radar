from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = Path("data")

PHASE4_PATTERN = (
    "sp500_radar_signals_*.csv"
)

TOP_N_NO_SIGNAL = 30


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Value Stock Radar",
    layout="wide",
)


# ============================================================
# LOAD DATA
# ============================================================

def find_latest_phase4_file() -> Path:

    files = list(
        DATA_DIR.glob(
            PHASE4_PATTERN
        )
    )

    if not files:

        raise FileNotFoundError(
            "Phase 4 결과 파일을 찾을 수 없습니다."
        )

    return max(
        files,
        key=lambda p: p.stat().st_mtime,
    )


@st.cache_data(
    ttl=300
)
def load_data():

    path = (
        find_latest_phase4_file()
    )

    df = pd.read_csv(
        path
    )

    numeric_columns = [
        "Combined Score",
        "Fundamental Score",
        "Technical Score",
        "Value Score",
        "Quality Score",
        "Growth Score",
        "Stability Score",
        "Price",
        "RSI14",
        "Momentum 3M",
        "Momentum 6M",
        "52W Drawdown",
        "MA50",
        "MA200",
        "PBR",
        "Forward PER",
        "PSR",
        "EV/EBITDA",
        "FCF Yield",
        "ROE",
        "ROA",
        "Debt/Equity",
        "Operating Margin",
        "Profit Margin",
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = (
                pd.to_numeric(
                    df[column],
                    errors="coerce",
                )
            )

    return (
        df,
        path,
    )


# ============================================================
# SAFE FUNCTIONS
# ============================================================

def safe_text(
    value,
) -> str:

    if value is None:

        return "-"

    try:

        if pd.isna(
            value
        ):

            return "-"

    except Exception:

        pass

    text = str(
        value
    ).strip()

    if not text:

        return "-"

    return text


def score_text(
    value,
) -> str:

    try:

        if pd.isna(
            value
        ):

            return "-"

        return (
            f"{float(value):.1f}"
        )

    except Exception:

        return "-"


def number_text(
    value,
    digits=2,
) -> str:

    try:

        if pd.isna(
            value
        ):

            return "-"

        return (
            f"{float(value):.{digits}f}"
        )

    except Exception:

        return "-"


def percent_text(
    value,
    digits=1,
) -> str:

    try:

        if pd.isna(
            value
        ):

            return "-"

        return (
            f"{float(value) * 100:.{digits}f}%"
        )

    except Exception:

        return "-"


# ============================================================
# SELECT RADAR
# ============================================================

def select_radar(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    bool,
]:

    result = df.copy()

    result = result.sort_values(

        by=[
            "Combined Score",
            "Fundamental Score",
            "Technical Score",
        ],

        ascending=[
            False,
            False,
            False,
        ],

        na_position="last",
    )

    signal_mask = (
        result[
            "Primary Signal"
        ]
        .fillna(
            "NONE"
        )
        .astype(str)
        .str.upper()
        .str.strip()
        != "NONE"
    )

    signal_df = (
        result[
            signal_mask
        ]
        .copy()
    )

    # --------------------------------------------------------
    # Signal 있음
    # --------------------------------------------------------

    if not signal_df.empty:

        signal_df[
            "Display Rank"
        ] = np.arange(
            1,
            len(signal_df) + 1,
        )

        return (
            signal_df,
            True,
        )

    # --------------------------------------------------------
    # Signal 없음
    # --------------------------------------------------------

    fallback = (
        result
        .head(
            TOP_N_NO_SIGNAL
        )
        .copy()
    )

    fallback[
        "Display Rank"
    ] = np.arange(
        1,
        len(fallback) + 1,
    )

    return (
        fallback,
        False,
    )


# ============================================================
# FILTERS
# ============================================================

def apply_filters(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = df.copy()

    st.sidebar.header(
        "Filters"
    )

    # --------------------------------------------------------
    # Sector
    # --------------------------------------------------------

    if "Sector" in result.columns:

        sectors = sorted(
            result[
                "Sector"
            ]
            .dropna()
            .astype(str)
            .unique()
        )

        selected_sectors = (
            st.sidebar.multiselect(
                "Sector",
                sectors,
                default=sectors,
            )
        )

        if selected_sectors:

            result = result[
                result[
                    "Sector"
                ].isin(
                    selected_sectors
                )
            ]

    # --------------------------------------------------------
    # Fundamental
    # --------------------------------------------------------

    min_fundamental = (
        st.sidebar.slider(
            "Minimum Fundamental Score",
            0,
            100,
            0,
        )
    )

    result = result[
        result[
            "Fundamental Score"
        ]
        .fillna(0)
        >= min_fundamental
    ]

    # --------------------------------------------------------
    # Technical
    # --------------------------------------------------------

    min_technical = (
        st.sidebar.slider(
            "Minimum Technical Score",
            0,
            100,
            0,
        )
    )

    result = result[
        result[
            "Technical Score"
        ]
        .fillna(0)
        >= min_technical
    ]

    # --------------------------------------------------------
    # Combined
    # --------------------------------------------------------

    min_combined = (
        st.sidebar.slider(
            "Minimum Combined Score",
            0,
            100,
            0,
        )
    )

    result = result[
        result[
            "Combined Score"
        ]
        .fillna(0)
        >= min_combined
    ]

    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    search = (
        st.sidebar.text_input(
            "Ticker / Company"
        )
        .strip()
        .lower()
    )

    if search:

        ticker = (
            result[
                "Ticker"
            ]
            .fillna("")
            .astype(str)
            .str.lower()
        )

        if "Short Name" in result.columns:

            company = (
                result[
                    "Short Name"
                ]
                .fillna("")
                .astype(str)
                .str.lower()
            )

        else:

            company = pd.Series(
                "",
                index=result.index,
            )

        result = result[
            ticker.str.contains(
                search,
                regex=False,
            )
            |
            company.str.contains(
                search,
                regex=False,
            )
        ]

    result = result.sort_values(

        by=[
            "Combined Score",
            "Fundamental Score",
            "Technical Score",
        ],

        ascending=[
            False,
            False,
            False,
        ],
    )

    result[
        "Display Rank"
    ] = np.arange(
        1,
        len(result) + 1,
    )

    return result


# ============================================================
# SUMMARY
# ============================================================

def show_summary(
    df: pd.DataFrame,
):

    if df.empty:

        return

    c1, c2, c3, c4 = (
        st.columns(4)
    )

    with c1:

        st.metric(
            "Stocks",
            len(df),
        )

    with c2:

        st.metric(
            "Avg Combined",
            f"{df['Combined Score'].mean():.1f}",
        )

    with c3:

        st.metric(
            "Avg Fundamental",
            f"{df['Fundamental Score'].mean():.1f}",
        )

    with c4:

        st.metric(
            "Avg Technical",
            f"{df['Technical Score'].mean():.1f}",
        )


# ============================================================
# TOP TABLE
# ============================================================

def show_radar_table(
    df: pd.DataFrame,
):

    st.subheader(
        "Radar Ranking"
    )

    if df.empty:

        st.warning(
            "조건에 해당하는 종목이 없습니다."
        )

        return

    display = df.copy()

    if (
        "52W Drawdown"
        in display.columns
    ):

        display[
            "52W Drawdown %"
        ] = (
            display[
                "52W Drawdown"
            ]
            * 100
        )

    columns = [
        "Display Rank",
        "Ticker",
        "Short Name",
        "Combined Score",
        "Fundamental Score",
        "Technical Score",
        "Primary Signal",
        "Signal Confidence",
        "Sector",
        "Price",
        "RSI14",
        "52W Drawdown %",
    ]

    columns = [
        column
        for column in columns
        if column in display.columns
    ]

    table = (
        display[
            columns
        ]
        .copy()
    )

    table = table.rename(
        columns={
            "Display Rank":
                "Rank",

            "Short Name":
                "Company",

            "Combined Score":
                "Combined",

            "Fundamental Score":
                "Fundamental",

            "Technical Score":
                "Technical",

            "Primary Signal":
                "Signal",

            "Signal Confidence":
                "Confidence",

            "RSI14":
                "RSI",

            "52W Drawdown %":
                "52W Drawdown",
        }
    )

    for column in [
        "Combined",
        "Fundamental",
        "Technical",
        "RSI",
        "52W Drawdown",
    ]:

        if column in table.columns:

            table[
                column
            ] = (
                table[
                    column
                ]
                .round(1)
            )

    if "Price" in table.columns:

        table[
            "Price"
        ] = (
            table[
                "Price"
            ]
            .round(2)
        )

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        height=620,
    )


# ============================================================
# STOCK SELECT
# ============================================================

def stock_selector(
    df: pd.DataFrame,
):

    if df.empty:

        return None

    tickers = (
        df[
            "Ticker"
        ]
        .tolist()
    )

    def format_stock(
        ticker,
    ):

        row = df[
            df[
                "Ticker"
            ]
            == ticker
        ].iloc[0]

        return (
            f"{ticker} | "
            f"{safe_text(row.get('Short Name'))} | "
            f"Combined "
            f"{score_text(row.get('Combined Score'))}"
        )

    selected = (
        st.selectbox(
            "종목 선택",
            tickers,
            format_func=format_stock,
        )
    )

    return (
        df[
            df[
                "Ticker"
            ]
            == selected
        ]
        .iloc[0]
    )


# ============================================================
# STOCK HEADER
# ============================================================

def show_stock_header(
    row,
):

    st.header(
        f"{row['Ticker']} — "
        f"{safe_text(row.get('Short Name'))}"
    )

    st.caption(
        f"{safe_text(row.get('Sector'))}"
        f" · "
        f"{safe_text(row.get('Industry'))}"
    )

    c1, c2, c3, c4 = (
        st.columns(4)
    )

    with c1:

        st.metric(
            "Combined",
            score_text(
                row.get(
                    "Combined Score"
                )
            ),
        )

    with c2:

        st.metric(
            "Fundamental",
            score_text(
                row.get(
                    "Fundamental Score"
                )
            ),
        )

    with c3:

        st.metric(
            "Technical",
            score_text(
                row.get(
                    "Technical Score"
                )
            ),
        )

    with c4:

        st.metric(
            "Price",
            number_text(
                row.get(
                    "Price"
                )
            ),
        )

    signal = safe_text(
        row.get(
            "Primary Signal"
        )
    )

    if signal != "NONE":

        st.success(
            f"Signal: {signal}"
        )

    else:

        st.info(
            "현재 공식 Signal 없음"
        )


# ============================================================
# FUNDAMENTAL
# ============================================================

def show_fundamental(
    row,
):

    st.subheader(
        "Fundamental"
    )

    c1, c2, c3, c4 = (
        st.columns(4)
    )

    with c1:

        st.metric(
            "Value",
            score_text(
                row.get(
                    "Value Score"
                )
            ),
        )

    with c2:

        st.metric(
            "Quality",
            score_text(
                row.get(
                    "Quality Score"
                )
            ),
        )

    with c3:

        st.metric(
            "Growth",
            score_text(
                row.get(
                    "Growth Score"
                )
            ),
        )

    with c4:

        st.metric(
            "Stability",
            score_text(
                row.get(
                    "Stability Score"
                )
            ),
        )

    st.markdown(
        "#### Valuation"
    )

    c1, c2, c3, c4 = (
        st.columns(4)
    )

    with c1:

        st.metric(
            "PBR",
            number_text(
                row.get(
                    "PBR"
                )
            ),
        )

    with c2:

        st.metric(
            "Forward PER",
            number_text(
                row.get(
                    "Forward PER"
                )
            ),
        )

    with c3:

        st.metric(
            "EV / EBITDA",
            number_text(
                row.get(
                    "EV/EBITDA"
                )
            ),
        )

    with c4:

        st.metric(
            "FCF Yield",
            percent_text(
                row.get(
                    "FCF Yield"
                )
            ),
        )

    st.markdown(
        "#### Quality"
    )

    c1, c2, c3, c4 = (
        st.columns(4)
    )

    with c1:

        st.metric(
            "ROE",
            percent_text(
                row.get(
                    "ROE"
                )
            ),
        )

    with c2:

        st.metric(
            "ROA",
            percent_text(
                row.get(
                    "ROA"
                )
            ),
        )

    with c3:

        st.metric(
            "Operating Margin",
            percent_text(
                row.get(
                    "Operating Margin"
                )
            ),
        )

    with c4:

        st.metric(
            "Debt / Equity",
            number_text(
                row.get(
                    "Debt/Equity"
                )
            ),
        )


# ============================================================
# TECHNICAL
# ============================================================

def show_technical(
    row,
):

    st.subheader(
        "Technical"
    )

    c1, c2, c3, c4 = (
        st.columns(4)
    )

    with c1:

        st.metric(
            "RSI",
            number_text(
                row.get(
                    "RSI14"
                ),
                1,
            ),
        )

    with c2:

        st.metric(
            "3M Momentum",
            percent_text(
                row.get(
                    "Momentum 3M"
                )
            ),
        )

    with c3:

        st.metric(
            "6M Momentum",
            percent_text(
                row.get(
                    "Momentum 6M"
                )
            ),
        )

    with c4:

        st.metric(
            "52W Drawdown",
            percent_text(
                row.get(
                    "52W Drawdown"
                )
            ),
        )

    c1, c2 = (
        st.columns(2)
    )

    with c1:

        st.metric(
            "MA50",
            number_text(
                row.get(
                    "MA50"
                )
            ),
        )

    with c2:

        st.metric(
            "MA200",
            number_text(
                row.get(
                    "MA200"
                )
            ),
        )

    st.write(
        "**Technical State:**",
        safe_text(
            row.get(
                "Technical State"
            )
        )
    )

    st.write(
        "**Trend State:**",
        safe_text(
            row.get(
                "Trend State"
            )
        )
    )

    st.write(
        "**RSI State:**",
        safe_text(
            row.get(
                "RSI State"
            )
        )
    )

    st.write(
        "**Stochastic State:**",
        safe_text(
            row.get(
                "Stochastic State"
            )
        )
    )


# ============================================================
# SIGNALS
# ============================================================

def show_signals(
    row,
):

    st.subheader(
        "Signals"
    )

    primary = safe_text(
        row.get(
            "Primary Signal"
        )
    )

    all_signals = safe_text(
        row.get(
            "Signals"
        )
    )

    confidence = safe_text(
        row.get(
            "Signal Confidence"
        )
    )

    risks = safe_text(
        row.get(
            "Risk Flags"
        )
    )

    st.write(
        "**Primary Signal:**",
        primary,
    )

    st.write(
        "**Signals:**",
        all_signals,
    )

    st.write(
        "**Confidence:**",
        confidence,
    )

    if risks != "-":

        st.warning(
            f"Risk Flags: {risks}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    st.title(
        "AI Value Stock Radar"
    )

    st.caption(
        "S&P 500 Fundamental + Technical Radar"
    )

    try:

        raw_df, source_path = (
            load_data()
        )

    except Exception as exc:

        st.error(
            str(exc)
        )

        st.stop()

    radar_df, has_signal = (
        select_radar(
            raw_df
        )
    )

    if has_signal:

        st.success(
            (
                f"현재 Signal 발생 종목 "
                f"{len(radar_df)}개를 "
                f"Combined Score 순으로 표시합니다."
            )
        )

    else:

        st.warning(
            (
                "현재 발생한 Signal이 없습니다. "
                f"전체 S&P 500 중 Combined Score "
                f"상위 {TOP_N_NO_SIGNAL}개를 표시합니다."
            )
        )

    filtered_df = (
        apply_filters(
            radar_df
        )
    )

    show_summary(
        filtered_df
    )

    st.divider()

    show_radar_table(
        filtered_df
    )

    st.divider()

    st.header(
        "Stock Detail"
    )

    row = stock_selector(
        filtered_df
    )

    if row is None:

        st.stop()

    show_stock_header(
        row
    )

    tab1, tab2, tab3 = (
        st.tabs(
            [
                "Fundamental",
                "Technical",
                "Signals",
            ]
        )
    )

    with tab1:

        show_fundamental(
            row
        )

    with tab2:

        show_technical(
            row
        )

    with tab3:

        show_signals(
            row
        )

    st.divider()

    st.caption(
        (
            "Combined Score = "
            "(Fundamental Score + Technical Score) / 2. "
            "Scores describe configured screening criteria "
            "and are not probabilities of future returns."
        )
    )

    st.sidebar.caption(
        f"Source: {source_path.name}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()