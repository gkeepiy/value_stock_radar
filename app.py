from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = Path("data")

PHASE4_PATTERN = (
    "sp500_radar_signals_*.csv"
)

TOP_N_NO_SIGNAL = 30

PORTFOLIO_FILE = DATA_DIR / "portfolio.csv"


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
        width="stretch",
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
# SIGNAL SUMMARY
# ============================================================

def build_signal_summary(
    row,
) -> str:
    """
    Phase 4 결과에 이미 저장된 Signal/상태값을 사람이 읽기 쉬운
    짧은 한글 설명으로 바꿉니다.

    주의:
    여기서는 Signal을 새로 계산하지 않습니다.
    실제 Signal 발생 여부는 phase4_signal.py 결과를 그대로 사용합니다.
    """

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

    trend_state = safe_text(
        row.get(
            "Trend State"
        )
    )

    rsi_state = safe_text(
        row.get(
            "RSI State"
        )
    )

    stochastic_state = safe_text(
        row.get(
            "Stochastic State"
        )
    )

    technical_state = safe_text(
        row.get(
            "Technical State"
        )
    )

    fundamental = score_text(
        row.get(
            "Fundamental Score"
        )
    )

    technical = score_text(
        row.get(
            "Technical Score"
        )
    )

    combined = score_text(
        row.get(
            "Combined Score"
        )
    )

    signal_upper = primary.upper()

    # Signal 이름을 쉬운 한국어로 번역합니다.
    signal_meanings = {
        "QUALITY_VALUE":
            "기업의 펀더멘털과 가치평가 조건이 함께 양호한 구간",
        "PULLBACK":
            "중장기 흐름 안에서 주가가 조정을 받아 진입 관찰 가치가 높아진 구간",
        "OVERSOLD":
            "단기적으로 주가가 과도하게 눌린 것으로 판단되는 구간",
        "MOMENTUM_CONFIRMATION":
            "가격의 상승 흐름과 모멘텀이 함께 확인되는 구간",
    }

    meaning = signal_meanings.get(
        signal_upper,
        "현재 설정된 조건 중 하나 이상이 충족된 구간",
    )

    parts = [
        f"**{primary}** 시그널입니다.",
        f"{meaning}으로 분류되었습니다.",
        (
            f"현재 점수는 Fundamental {fundamental}, "
            f"Technical {technical}, Combined {combined}입니다."
        ),
    ]

    state_parts = []

    if trend_state not in {"-", "NONE"}:
        state_parts.append(
            f"추세: {trend_state}"
        )

    if rsi_state not in {"-", "NONE"}:
        state_parts.append(
            f"RSI: {rsi_state}"
        )

    if stochastic_state not in {"-", "NONE"}:
        state_parts.append(
            f"Stochastic: {stochastic_state}"
        )

    if technical_state not in {"-", "NONE"}:
        state_parts.append(
            f"기술 상태: {technical_state}"
        )

    if state_parts:
        parts.append(
            "현재 상태는 "
            + ", ".join(
                state_parts
            )
            + " 입니다."
        )

    if (
        all_signals not in {
            "-",
            "NONE",
            primary,
        }
    ):
        parts.append(
            f"함께 감지된 신호: {all_signals}"
        )

    return " ".join(
        parts
    )


def show_signal_reason(
    row,
):
    st.markdown(
        "#### 왜 시그널이 발생했나요?"
    )

    st.info(
        build_signal_summary(
            row
        )
    )

    risks = safe_text(
        row.get(
            "Risk Flags"
        )
    )

    if risks not in {
        "-",
        "NONE",
    }:
        st.caption(
            f"⚠️ 참고할 위험 신호: {risks}"
        )


# ============================================================
# PAGE 1 - STOCK RECOMMENDATIONS
# ============================================================

def show_stock_recommendations(raw_df: pd.DataFrame, source_path: Path):
    st.title("📊 주식추천")
    st.caption("S&P 500 Fundamental + Technical Radar")

    recommendation_df = (
        raw_df
        .sort_values(
            by=["Combined Score", "Fundamental Score", "Technical Score"],
            ascending=[False, False, False],
            na_position="last",
        )
        .copy()
    )

    recommendation_df["Display Rank"] = np.arange(1, len(recommendation_df) + 1)

    filtered_df = apply_filters(recommendation_df)
    show_summary(filtered_df)

    st.divider()
    show_radar_table(filtered_df)

    st.divider()
    st.header("Stock Detail")

    row = stock_selector(filtered_df)
    if row is None:
        return

    show_stock_header(row)

    tab1, tab2, tab3 = st.tabs(["Fundamental", "Technical", "Signals"])

    with tab1:
        show_fundamental(row)

    with tab2:
        show_technical(row)

    with tab3:
        show_signals(row)

    st.divider()
    st.caption(
        "Combined Score = (Fundamental Score + Technical Score) / 2. "
        "Scores describe configured screening criteria and are not probabilities of future returns."
    )
    st.sidebar.caption(f"Source: {source_path.name}")


# ============================================================
# PAGE 2 - SIGNALS
# ============================================================

def show_signal_page(raw_df: pd.DataFrame):
    st.title("🚨 시그널 발생")
    st.caption("시그널이 발생한 종목만 모아서 보는 페이지입니다.")

    signal_mask = (
        raw_df["Primary Signal"]
        .fillna("NONE")
        .astype(str)
        .str.upper()
        .str.strip()
        != "NONE"
    )

    signal_df = (
        raw_df[signal_mask]
        .sort_values(
            by=["Combined Score", "Fundamental Score", "Technical Score"],
            ascending=[False, False, False],
            na_position="last",
        )
        .copy()
    )

    if signal_df.empty:
        st.info("현재 발생한 시그널이 없습니다.")
        return

    signal_df["Display Rank"] = np.arange(1, len(signal_df) + 1)
    st.success(f"현재 시그널 발생 종목: {len(signal_df)}개")

    columns = [
        "Display Rank", "Ticker", "Short Name",
        "Combined Score", "Fundamental Score", "Technical Score",
        "Primary Signal", "Signal Confidence",
    ]
    columns = [c for c in columns if c in signal_df.columns]

    table = signal_df[columns].copy().rename(
        columns={
            "Display Rank": "Rank",
            "Short Name": "Company",
            "Combined Score": "Combined",
            "Fundamental Score": "Fundamental",
            "Technical Score": "Technical",
            "Primary Signal": "Signal",
            "Signal Confidence": "Confidence",
        }
    )

    st.dataframe(table, width="stretch", hide_index=True)

    st.divider()
    st.subheader("시그널 종목 상세")

    row = stock_selector(signal_df)
    if row is None:
        return

    show_stock_header(row)

    show_signal_reason(
        row
    )

    st.divider()

    show_signals(
        row
    )


# ============================================================
# CURRENCY / FX
# ============================================================

@st.cache_data(ttl=1800)
def get_fx_to_eur(
    currency: str,
) -> float:
    """
    1 단위의 해당 통화가 몇 EUR인지 반환합니다.
    EUR -> 1.0
    USD -> EUR=X
    JPY -> JPYEUR=X
    기타 통화 -> {통화}EUR=X 를 시도합니다.
    """
    currency = (
        str(currency)
        .upper()
        .strip()
    )

    if currency == "EUR":
        return 1.0

    if not currency:
        return float("nan")

    if currency == "USD":
        fx_symbol = "EUR=X"
    else:
        fx_symbol = f"{currency}EUR=X"

    try:
        ticker = yf.Ticker(
            fx_symbol
        )

        try:
            rate = float(
                ticker.fast_info[
                    "last_price"
                ]
            )

            if rate > 0:
                return rate
        except Exception:
            pass

        history = ticker.history(
            period="5d",
            interval="1d",
        )

        if (
            not history.empty
            and "Close" in history.columns
        ):
            closes = (
                history["Close"]
                .dropna()
            )

            if not closes.empty:
                rate = float(
                    closes.iloc[-1]
                )

                if rate > 0:
                    return rate

    except Exception:
        pass

    return float("nan")


@st.cache_data(ttl=1800)
def get_ticker_currency(
    ticker_symbol: str,
) -> str:
    """
    Yahoo Finance에서 종목의 거래 통화를 자동 감지합니다.
    예:
    AAPL -> USD
    IFX.DE -> EUR
    186A.T -> JPY
    """
    try:
        ticker_symbol = (
            str(ticker_symbol)
            .upper()
            .strip()
        )

        ticker = yf.Ticker(
            ticker_symbol
        )

        try:
            currency = ticker.fast_info[
                "currency"
            ]

            if currency:
                return (
                    str(currency)
                    .upper()
                    .strip()
                )
        except Exception:
            pass

        try:
            info = ticker.info
            currency = info.get(
                "currency"
            )

            if currency:
                return (
                    str(currency)
                    .upper()
                    .strip()
                )
        except Exception:
            pass

    except Exception:
        pass

    return ""


# ============================================================
# CURRENT STOCK PRICE
# ============================================================

@st.cache_data(ttl=900)
def get_current_usd_price(
    ticker_symbol: str,
) -> float:
    """
    Yahoo Finance에서 미국 주식의 최근 USD 가격을 가져옵니다.
    fast_info -> 최근 일봉 Close 순서로 시도합니다.
    """
    try:
        ticker_symbol = (
            str(ticker_symbol)
            .upper()
            .strip()
        )

        if not ticker_symbol:
            return float("nan")

        ticker = yf.Ticker(
            ticker_symbol
        )

        # 1) fast_info 우선
        try:
            fast_info = ticker.fast_info

            for key in [
                "last_price",
                "lastPrice",
                "regular_market_price",
            ]:
                try:
                    value = fast_info[key]
                    value = float(value)

                    if value > 0:
                        return value
                except Exception:
                    pass

        except Exception:
            pass

        # 2) 최근 일봉 종가 fallback
        history = ticker.history(
            period="5d",
            interval="1d",
            auto_adjust=False,
        )

        if (
            not history.empty
            and "Close" in history.columns
        ):
            closes = (
                history["Close"]
                .dropna()
            )

            if not closes.empty:
                value = float(
                    closes.iloc[-1]
                )

                if value > 0:
                    return value

    except Exception:
        pass

    return float("nan")


# ============================================================
# PORTFOLIO DATA - SUPABASE
# ============================================================

PORTFOLIO_COLUMNS = ["Ticker", "Buy Price", "Shares"]


def get_supabase_config() -> tuple[str, str]:
    try:
        url = str(st.secrets["SUPABASE_URL"]).strip().rstrip("/")

        key = str(
            st.secrets.get(
                "SUPABASE_KEY",
                st.secrets.get("SUPABASE_PUBLISHABLE_KEY", "")
            )
        ).strip()

    except Exception as exc:
        raise RuntimeError(
            "Streamlit Secrets에서 Supabase 설정을 읽지 못했습니다."
        ) from exc

    if not url:
        raise RuntimeError("SUPABASE_URL이 없습니다.")

    if not key:
        raise RuntimeError(
            "SUPABASE_KEY 또는 SUPABASE_PUBLISHABLE_KEY가 없습니다."
        )

    return url, key


def supabase_headers() -> dict:
    _, key = get_supabase_config()
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }


def normalize_portfolio(portfolio: pd.DataFrame) -> pd.DataFrame:
    result = portfolio.copy()
    for column in PORTFOLIO_COLUMNS:
        if column not in result.columns:
            result[column] = np.nan

    result["Ticker"] = (
        result["Ticker"].fillna("").astype(str).str.upper().str.strip()
    )
    for column in ["Buy Price", "Shares"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    return result[result["Ticker"] != ""][PORTFOLIO_COLUMNS].copy()


def load_portfolio() -> pd.DataFrame:
    url, _ = get_supabase_config()
    response = requests.get(
        f"{url}/rest/v1/portfolio",
        headers=supabase_headers(),
        params={"select": "ticker,buy_price,shares", "order": "ticker.asc"},
        timeout=15,
    )
    response.raise_for_status()
    rows = response.json()

    if not rows:
        return pd.DataFrame(columns=PORTFOLIO_COLUMNS)

    portfolio = pd.DataFrame(rows).rename(
        columns={
            "ticker": "Ticker",
            "buy_price": "Buy Price",
            "shares": "Shares",
        }
    )
    return normalize_portfolio(portfolio)


def save_portfolio(portfolio: pd.DataFrame) -> None:
    url, _ = get_supabase_config()
    save_df = normalize_portfolio(portfolio)

    response = requests.get(
        f"{url}/rest/v1/portfolio",
        headers=supabase_headers(),
        params={"select": "ticker"},
        timeout=15,
    )
    response.raise_for_status()

    remote_tickers = {
        str(row.get("ticker", "")).upper().strip()
        for row in response.json()
        if row.get("ticker")
    }
    local_tickers = set(save_df["Ticker"].tolist())

    if not save_df.empty:
        payload = [
            {
                "ticker": row["Ticker"],
                "buy_price": float(row["Buy Price"]),
                "shares": float(row["Shares"]),
            }
            for _, row in save_df.iterrows()
        ]

        headers = supabase_headers()
        headers["Prefer"] = "resolution=merge-duplicates,return=minimal"

        response = requests.post(
            f"{url}/rest/v1/portfolio",
            headers=headers,
            params={"on_conflict": "ticker"},
            json=payload,
            timeout=15,
        )
        response.raise_for_status()

    for ticker in sorted(remote_tickers - local_tickers):
        response = requests.delete(
            f"{url}/rest/v1/portfolio",
            headers=supabase_headers(),
            params={"ticker": f"eq.{ticker}"},
            timeout=15,
        )
        response.raise_for_status()


def build_portfolio_view(
    portfolio: pd.DataFrame,
    raw_df: pd.DataFrame,
) -> pd.DataFrame:
    if portfolio.empty:
        return portfolio.copy()

    radar_columns = [
        "Ticker",
        "Short Name",
        "Combined Score",
        "Primary Signal",
    ]

    radar_columns = [
        c for c in radar_columns
        if c in raw_df.columns
    ]

    radar = raw_df[
        radar_columns
    ].copy()

    radar["Ticker"] = (
        radar["Ticker"]
        .fillna("")
        .astype(str)
        .str.upper()
        .str.strip()
    )

    result = portfolio.merge(
        radar,
        on="Ticker",
        how="left",
    )

    currencies = {}
    prices = {}
    fx_rates = {}

    for ticker_symbol in result[
        "Ticker"
    ].tolist():

        currency = get_ticker_currency(
            ticker_symbol
        )

        price = get_current_usd_price(
            ticker_symbol
        )

        fx_rate = get_fx_to_eur(
            currency
        )

        currencies[
            ticker_symbol
        ] = currency

        prices[
            ticker_symbol
        ] = price

        fx_rates[
            ticker_symbol
        ] = fx_rate

    result["Currency"] = (
        result["Ticker"]
        .map(currencies)
    )

    # 이름은 Local Price: 해당 거래소의 원래 통화 가격
    result["Local Price"] = (
        result["Ticker"]
        .map(prices)
    )

    result["FX to EUR"] = (
        result["Ticker"]
        .map(fx_rates)
    )

    result["Price EUR"] = (
        result["Local Price"]
        * result["FX to EUR"]
    )

    # 입력한 매수가는 항상 실제 브로커의 EUR 평균 매수가
    result["Cost Basis EUR"] = (
        result["Buy Price"]
        * result["Shares"]
    )

    result["Market Value EUR"] = (
        result["Price EUR"]
        * result["Shares"]
    )

    result["P/L EUR"] = (
        result["Market Value EUR"]
        - result["Cost Basis EUR"]
    )

    result["Return % EUR"] = (
        (
            result["Price EUR"]
            / result["Buy Price"]
        )
        - 1
    ) * 100

    return result

def show_portfolio_summary(
    df: pd.DataFrame,
):
    if df.empty:
        return

    total_cost = df[
        "Cost Basis EUR"
    ].sum(min_count=1)

    total_value = df[
        "Market Value EUR"
    ].sum(min_count=1)

    total_pl = df[
        "P/L EUR"
    ].sum(min_count=1)

    if (
        pd.notna(total_cost)
        and total_cost != 0
        and pd.notna(total_value)
    ):
        total_return = (
            total_value / total_cost - 1
        ) * 100
    else:
        total_return = np.nan

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "보유 종목",
            len(df),
        )

    with c2:
        st.metric(
            "총 매입금액 (EUR)",
            (
                f"€{total_cost:,.2f}"
                if pd.notna(total_cost)
                else "-"
            ),
        )

    with c3:
        st.metric(
            "현재 평가금액 (EUR)",
            (
                f"€{total_value:,.2f}"
                if pd.notna(total_value)
                else "-"
            ),
        )

    with c4:
        st.metric(
            "총 수익률 (EUR)",
            (
                f"{total_return:,.2f}%"
                if pd.notna(total_return)
                else "-"
            ),
            delta=(
                f"€{total_pl:,.2f}"
                if pd.notna(total_pl)
                else None
            ),
        )


# ============================================================
# PORTFOLIO NEWS
# ============================================================

@st.cache_data(ttl=900)
def get_ticker_news(
    ticker_symbol: str,
    count: int = 10,
) -> list:
    """
    Yahoo Finance/yfinance에서 종목 관련 최신 뉴스를 가져옵니다.
    """
    try:
        ticker_symbol = (
            str(ticker_symbol)
            .upper()
            .strip()
        )

        if not ticker_symbol:
            return []

        ticker = yf.Ticker(
            ticker_symbol
        )

        try:
            news = ticker.get_news(
                count=count,
                tab="news",
            )
        except Exception:
            news = ticker.news

        if not isinstance(
            news,
            list,
        ):
            return []

        return news

    except Exception:
        return []


def normalize_news_item(
    item: dict,
) -> dict:
    """
    yfinance 버전별 news 구조 차이를 흡수합니다.
    """
    if not isinstance(
        item,
        dict,
    ):
        return {}

    content = item.get(
        "content"
    )

    if isinstance(
        content,
        dict,
    ):
        title = (
            content.get("title")
            or item.get("title")
            or "-"
        )

        summary = (
            content.get("summary")
            or content.get("description")
            or ""
        )

        provider = content.get(
            "provider"
        )

        if isinstance(
            provider,
            dict,
        ):
            publisher = (
                provider.get("displayName")
                or provider.get("name")
                or "-"
            )
        else:
            publisher = (
                item.get("publisher")
                or "-"
            )

        canonical = content.get(
            "canonicalUrl"
        )

        click_through = content.get(
            "clickThroughUrl"
        )

        url = ""

        if isinstance(
            canonical,
            dict,
        ):
            url = (
                canonical.get("url")
                or ""
            )

        if (
            not url
            and isinstance(
                click_through,
                dict,
            )
        ):
            url = (
                click_through.get("url")
                or ""
            )

        if not url:
            url = (
                item.get("link")
                or ""
            )

        published = (
            content.get("pubDate")
            or content.get("displayTime")
            or item.get("providerPublishTime")
        )

    else:
        title = (
            item.get("title")
            or "-"
        )

        summary = (
            item.get("summary")
            or item.get("description")
            or ""
        )

        publisher = (
            item.get("publisher")
            or "-"
        )

        url = (
            item.get("link")
            or ""
        )

        published = item.get(
            "providerPublishTime"
        )

    published_dt = None

    if isinstance(
        published,
        (int, float),
    ):
        try:
            published_dt = datetime.fromtimestamp(
                published,
                tz=timezone.utc,
            )
        except Exception:
            published_dt = None

    elif isinstance(
        published,
        str,
    ):
        try:
            published_dt = datetime.fromisoformat(
                published.replace(
                    "Z",
                    "+00:00",
                )
            )
        except Exception:
            published_dt = None

    return {
        "title": str(title),
        "summary": str(summary),
        "publisher": str(publisher),
        "url": str(url),
        "published_dt": published_dt,
    }


def short_news_summary(
    title: str,
    summary: str,
) -> str:
    """
    OpenAI API 없이 Yahoo가 제공하는 제목/요약을 짧게 표시합니다.
    """
    summary = (
        summary
        .replace("\n", " ")
        .strip()
    )

    if summary:
        if len(summary) > 260:
            return (
                summary[:257]
                + "..."
            )
        return summary

    return (
        "기사 제목을 참고하세요: "
        + title
    )


def show_portfolio_news(
    ticker_symbol: str,
):
    st.subheader(
        "📰 오늘의 관련 뉴스"
    )

    st.caption(
        "Yahoo Finance에서 가져온 최신 종목 관련 뉴스입니다. "
        "오늘 기사가 없으면 가장 최근 기사를 표시합니다."
    )

    raw_news = get_ticker_news(
        ticker_symbol,
        count=12,
    )

    normalized = []

    for item in raw_news:
        news = normalize_news_item(
            item
        )

        if (
            news
            and news.get("title")
            not in {
                "",
                "-",
            }
        ):
            normalized.append(
                news
            )

    if not normalized:
        st.info(
            f"{ticker_symbol} 관련 뉴스를 가져오지 못했습니다."
        )
        return

    now_utc = datetime.now(
        timezone.utc
    )

    today_news = [
        item
        for item in normalized
        if (
            item.get("published_dt")
            is not None
            and item[
                "published_dt"
            ].date()
            == now_utc.date()
        )
    ]

    if today_news:
        display_news = (
            today_news[:5]
        )

        st.success(
            f"오늘 올라온 {ticker_symbol} 관련 뉴스 "
            f"{len(display_news)}건을 표시합니다."
        )
    else:
        display_news = (
            normalized[:5]
        )

        st.info(
            "오늘 날짜의 관련 기사가 없어 "
            "가장 최근 뉴스부터 표시합니다."
        )

    for index, news in enumerate(
        display_news,
        start=1,
    ):
        published_dt = news.get(
            "published_dt"
        )

        if published_dt:
            date_text = (
                published_dt
                .astimezone()
                .strftime(
                    "%Y-%m-%d %H:%M"
                )
            )
        else:
            date_text = (
                "시간 정보 없음"
            )

        st.markdown(
            f"#### {index}. {news['title']}"
        )

        st.caption(
            f"{news['publisher']} · {date_text}"
        )

        st.write(
            short_news_summary(
                news["title"],
                news["summary"],
            )
        )

        if news.get("url"):
            st.link_button(
                "기사 열기",
                news["url"],
            )

        st.divider()


# ============================================================
# PAGE 3 - PORTFOLIO
# ============================================================

def show_portfolio_page(
    raw_df: pd.DataFrame,
):
    st.title("💼 포트폴리오")

    st.caption(
        "매수가는 EUR로 입력합니다. "
        "미국·독일·일본 등 종목의 거래 통화를 자동 감지해 "
        "현재가를 EUR로 환산합니다."
    )

    try:
        portfolio = load_portfolio()
    except Exception as exc:
        st.error("Supabase에서 포트폴리오를 불러오지 못했습니다.")
        st.code(str(exc))
        st.info(
            "Streamlit Secrets와 Supabase portfolio 테이블/RLS 설정을 확인해주세요."
        )
        return

    # --------------------------------------------------------
    # ADD STOCK
    # --------------------------------------------------------
    with st.expander(
        "➕ 종목 추가",
        expanded=portfolio.empty,
    ):
        with st.form(
            "add_portfolio_stock",
            clear_on_submit=True,
        ):
            c1, c2, c3 = st.columns(3)

            with c1:
                ticker = st.text_input(
                    "티커",
                    placeholder="예: AAPL",
                )

            with c2:
                buy_price = st.number_input(
                    "평균 매수가 (EUR)",
                    min_value=0.0,
                    value=0.0,
                    step=0.01,
                    format="%.2f",
                )

            with c3:
                shares = st.number_input(
                    "보유 수량",
                    min_value=0.0,
                    value=0.0,
                    step=1.0,
                    format="%.4f",
                )

            submitted = st.form_submit_button(
                "포트폴리오에 추가",
                type="primary",
            )

        if submitted:
            ticker = (
                ticker
                .upper()
                .strip()
            )

            if not ticker:
                st.error(
                    "티커를 입력해주세요."
                )

            elif buy_price <= 0:
                st.error(
                    "평균 매수가는 0보다 커야 합니다."
                )

            elif shares <= 0:
                st.error(
                    "보유 수량은 0보다 커야 합니다."
                )

            else:
                existing = (
                    portfolio["Ticker"]
                    .astype(str)
                    .str.upper()
                    == ticker
                )

                if existing.any():
                    portfolio.loc[
                        existing,
                        "Buy Price",
                    ] = buy_price

                    portfolio.loc[
                        existing,
                        "Shares",
                    ] = shares

                    message = (
                        f"{ticker} 포트폴리오 정보를 수정했습니다."
                    )

                else:
                    new_row = pd.DataFrame(
                        [
                            {
                                "Ticker": ticker,
                                "Buy Price": buy_price,
                                "Shares": shares,
                            }
                        ]
                    )

                    portfolio = pd.concat(
                        [
                            portfolio,
                            new_row,
                        ],
                        ignore_index=True,
                    )

                    message = (
                        f"{ticker} 종목을 추가했습니다."
                    )

                save_portfolio(
                    portfolio
                )

                st.success(
                    message
                )

                st.rerun()

    if portfolio.empty:
        st.info(
            "아직 등록된 종목이 없습니다. "
            "위의 '종목 추가'에서 첫 종목을 입력해주세요."
        )
        return

    # --------------------------------------------------------
    # EDIT / DELETE
    # --------------------------------------------------------
    with st.expander(
        "✏️ 종목 수정 / 삭제",
        expanded=False,
    ):
        selected_ticker = st.selectbox(
            "수정할 종목",
            portfolio["Ticker"].tolist(),
            key="portfolio_edit_ticker",
        )

        selected_row = portfolio[
            portfolio["Ticker"]
            == selected_ticker
        ].iloc[0]

        c1, c2 = st.columns(2)

        with c1:
            edited_buy_price = st.number_input(
                "수정 평균 매수가 (EUR)",
                min_value=0.0,
                value=float(
                    selected_row["Buy Price"]
                ),
                step=0.01,
                format="%.2f",
                key="portfolio_edit_price",
            )

        with c2:
            edited_shares = st.number_input(
                "수정 보유 수량",
                min_value=0.0,
                value=float(
                    selected_row["Shares"]
                ),
                step=1.0,
                format="%.4f",
                key="portfolio_edit_shares",
            )

        b1, b2 = st.columns(2)

        with b1:
            if st.button(
                "수정 저장",
                type="primary",
                width="stretch",
            ):
                if (
                    edited_buy_price <= 0
                    or edited_shares <= 0
                ):
                    st.error(
                        "매수가와 보유 수량은 0보다 커야 합니다."
                    )
                else:
                    mask = (
                        portfolio["Ticker"]
                        == selected_ticker
                    )

                    portfolio.loc[
                        mask,
                        "Buy Price",
                    ] = edited_buy_price

                    portfolio.loc[
                        mask,
                        "Shares",
                    ] = edited_shares

                    save_portfolio(
                        portfolio
                    )

                    st.success(
                        f"{selected_ticker} 정보를 수정했습니다."
                    )

                    st.rerun()

        with b2:
            if st.button(
                "종목 삭제",
                width="stretch",
            ):
                portfolio = portfolio[
                    portfolio["Ticker"]
                    != selected_ticker
                ].copy()

                save_portfolio(
                    portfolio
                )

                st.success(
                    f"{selected_ticker} 종목을 삭제했습니다."
                )

                st.rerun()

    # --------------------------------------------------------
    # PORTFOLIO VIEW
    # --------------------------------------------------------
    view = build_portfolio_view(
        portfolio,
        raw_df,
    )

    show_portfolio_summary(
        view
    )

    if (
        "Local Price" in view.columns
        and view["Local Price"].isna().any()
    ):
        missing = (
            view.loc[
                view["Local Price"].isna(),
                "Ticker",
            ]
            .astype(str)
            .tolist()
        )

        st.warning(
            "현재가를 가져오지 못한 종목: "
            + ", ".join(missing)
            + " / 잠시 후 새로고침하거나 티커를 확인해주세요."
        )

    st.divider()
    st.subheader(
        "보유 종목"
    )

    display_columns = [
        "Ticker",
        "Short Name",
        "Buy Price",
        "Currency",
        "Local Price",
        "Price EUR",
        "Shares",
        "Cost Basis EUR",
        "Market Value EUR",
        "P/L EUR",
        "Return % EUR",
        "Combined Score",
        "Primary Signal",
    ]

    display_columns = [
        c for c in display_columns
        if c in view.columns
    ]

    display = view[
        display_columns
    ].copy()

    for column in [
        "Buy Price",
        "Local Price",
        "Price EUR",
        "Shares",
        "Cost Basis EUR",
        "Market Value EUR",
        "P/L EUR",
        "Return % EUR",
        "Combined Score",
    ]:
        if column in display.columns:
            display[column] = (
                display[column]
                .round(2)
            )

    display = display.rename(
        columns={
            "Short Name": "Company",
            "Buy Price": "매수가 EUR",
            "Currency": "거래통화",
            "Local Price": "현지 현재가",
            "Price EUR": "현재가 EUR",
            "Shares": "수량",
            "Cost Basis EUR": "매입금액 EUR",
            "Market Value EUR": "평가금액 EUR",
            "P/L EUR": "손익 EUR",
            "Return % EUR": "수익률 % EUR",
            "Combined Score": "Combined",
            "Primary Signal": "Signal",
        }
    )

    st.dataframe(
        display,
        width="stretch",
        hide_index=True,
    )

    st.divider()
    st.subheader(
        "종목 상세"
    )

    selected = st.selectbox(
        "포트폴리오 종목 선택",
        view["Ticker"].tolist(),
        key="portfolio_detail_ticker",
    )

    row = view[
        view["Ticker"]
        == selected
    ].iloc[0]

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "평균 매수가",
            (
                f"€{row['Buy Price']:,.2f}"
                if pd.notna(
                    row.get("Buy Price")
                )
                else "-"
            ),
        )

    with c2:
        local_currency = safe_text(
            row.get("Currency")
        )

        local_price = row.get(
            "Local Price"
        )

        st.metric(
            f"현지 현재가 ({local_currency})",
            (
                f"{local_price:,.2f} {local_currency}"
                if pd.notna(local_price)
                else "-"
            ),
        )

    with c3:
        st.metric(
            "현재가 (EUR)",
            (
                f"€{row['Price EUR']:,.2f}"
                if pd.notna(
                    row.get("Price EUR")
                )
                else "-"
            ),
        )

    with c4:
        st.metric(
            "수익률 (EUR)",
            (
                f"{row['Return % EUR']:,.2f}%"
                if pd.notna(
                    row.get("Return % EUR")
                )
                else "-"
            ),
            delta=(
                f"€{row['P/L EUR']:,.2f}"
                if pd.notna(
                    row.get("P/L EUR")
                )
                else None
            ),
        )

    fx_rate = row.get(
        "FX to EUR"
    )

    currency = safe_text(
        row.get("Currency")
    )

    if pd.notna(fx_rate):
        st.info(
            f"거래 통화: {currency} · "
            f"환산 기준: 1 {currency} = {fx_rate:.6f} EUR"
        )
    else:
        st.warning(
            f"{selected}의 {currency} → EUR 환율을 가져오지 못했습니다."
        )

    st.caption(
        "EUR 수익률은 입력한 EUR 평균 매수가와, "
        "종목의 현지 현재가를 해당 통화의 EUR 환율로 환산한 가격을 비교해 계산합니다."
    )

    st.divider()

    show_portfolio_news(
        selected
    )


# ============================================================
# MAIN
# ============================================================

def main():
    try:
        raw_df, source_path = load_data()
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    st.sidebar.title("Value Stock Radar")
    page = st.sidebar.radio(
        "메뉴",
        [
            "📊 1. 주식추천",
            "🚨 2. 시그널 발생",
            "💼 3. 포트폴리오",
        ],
    )

    st.sidebar.divider()

    if page == "📊 1. 주식추천":
        show_stock_recommendations(raw_df, source_path)
    elif page == "🚨 2. 시그널 발생":
        show_signal_page(raw_df)
    elif page == "💼 3. 포트폴리오":
        show_portfolio_page(raw_df)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()