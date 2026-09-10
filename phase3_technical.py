from __future__ import annotations

import typing as t
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


# ============================================================
# 0) CONFIG
# ============================================================

DATA_DIR = Path("data")

PHASE2_PATTERN = (
    "sp500_fundamental_scores_*.csv"
)

PRICE_PERIOD = "2y"
PRICE_INTERVAL = "1d"

RSI_PERIOD = 14

STOCH_K_PERIOD = 14
STOCH_D_PERIOD = 3

MA_SHORT = 50
MA_LONG = 200

MOMENTUM_3M_DAYS = 63
MOMENTUM_6M_DAYS = 126

HIGH_52W_DAYS = 252

VOLUME_PERIOD = 20


# ============================================================
# 1) LOAD PHASE 2
# ============================================================

def find_latest_phase2_file() -> Path:

    files = list(
        DATA_DIR.glob(
            PHASE2_PATTERN
        )
    )

    if not files:

        raise FileNotFoundError(
            "Phase 2 결과 파일을 찾을 수 없습니다.\n"
            "data/sp500_fundamental_scores_YYYY-MM-DD.csv "
            "파일이 있는지 확인하세요."
        )

    return max(
        files,
        key=lambda p: p.stat().st_mtime,
    )


def load_phase2_data() -> tuple[
    pd.DataFrame,
    Path,
]:

    path = (
        find_latest_phase2_file()
    )

    print(
        f"📂 Phase 2 데이터 로드: {path}"
    )

    df = pd.read_csv(
        path
    )

    if "Ticker" not in df.columns:

        raise ValueError(
            "Phase 2 데이터에 Ticker 컬럼이 없습니다."
        )

    df["Ticker"] = (
        df["Ticker"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    return (
        df,
        path,
    )


# ============================================================
# 2) DOWNLOAD PRICE DATA
# ============================================================

def download_price_data(
    tickers: list[str],
) -> pd.DataFrame:

    print()
    print(
        f"📡 {len(tickers)}개 종목 "
        f"가격 데이터 다운로드..."
    )

    prices = yf.download(
        tickers=tickers,
        period=PRICE_PERIOD,
        interval=PRICE_INTERVAL,
        auto_adjust=True,
        group_by="ticker",
        threads=True,
        progress=True,
    )

    if (
        prices is None
        or prices.empty
    ):

        raise RuntimeError(
            "가격 데이터를 다운로드하지 못했습니다."
        )

    return prices


# ============================================================
# 3) EXTRACT SINGLE TICKER
# ============================================================

def extract_ticker_frame(
    prices: pd.DataFrame,
    ticker: str,
) -> pd.DataFrame:

    if isinstance(
        prices.columns,
        pd.MultiIndex,
    ):

        level0 = (
            prices.columns
            .get_level_values(0)
            .astype(str)
        )

        level1 = (
            prices.columns
            .get_level_values(1)
            .astype(str)
        )

        if ticker in level0:

            df = prices[
                ticker
            ].copy()

        elif ticker in level1:

            df = prices.xs(
                ticker,
                axis=1,
                level=1,
            ).copy()

        else:

            return pd.DataFrame()

    else:

        df = prices.copy()

    required = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    if not set(
        required
    ).issubset(
        set(
            df.columns
        )
    ):

        return pd.DataFrame()

    df = df[
        required
    ].copy()

    for column in required:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=[
            "Close"
        ]
    )

    df = df.sort_index()

    return df


# ============================================================
# 4) RSI
# ============================================================

def calculate_rsi(
    close: pd.Series,
    period: int = RSI_PERIOD,
) -> pd.Series:

    delta = close.diff()

    gain = delta.clip(
        lower=0
    )

    loss = (
        -delta.clip(
            upper=0
        )
    )

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    rs = (
        avg_gain
        / avg_loss.replace(
            0,
            np.nan,
        )
    )

    rsi = (
        100
        -
        (
            100
            / (
                1 + rs
            )
        )
    )

    rsi = rsi.where(
        avg_loss != 0,
        100.0,
    )

    rsi = rsi.where(
        avg_gain != 0,
        0.0,
    )

    return rsi


# ============================================================
# 5) STOCHASTIC
# ============================================================

def calculate_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> tuple[
    pd.Series,
    pd.Series,
]:

    lowest = low.rolling(
        STOCH_K_PERIOD
    ).min()

    highest = high.rolling(
        STOCH_K_PERIOD
    ).max()

    denominator = (
        highest
        - lowest
    )

    k = (
        100
        * (
            close
            - lowest
        )
        / denominator.replace(
            0,
            np.nan,
        )
    )

    d = k.rolling(
        STOCH_D_PERIOD
    ).mean()

    return (
        k,
        d,
    )


# ============================================================
# 6) ADD INDICATORS
# ============================================================

def add_indicators(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = df.copy()

    result[
        "MA50"
    ] = (
        result["Close"]
        .rolling(
            MA_SHORT
        )
        .mean()
    )

    result[
        "MA200"
    ] = (
        result["Close"]
        .rolling(
            MA_LONG
        )
        .mean()
    )

    result[
        "RSI14"
    ] = calculate_rsi(
        result[
            "Close"
        ]
    )

    (
        result[
            "Stoch K"
        ],
        result[
            "Stoch D"
        ],
    ) = calculate_stochastic(
        result[
            "High"
        ],
        result[
            "Low"
        ],
        result[
            "Close"
        ],
    )

    result[
        "Momentum 3M"
    ] = (
        result[
            "Close"
        ]
        / result[
            "Close"
        ].shift(
            MOMENTUM_3M_DAYS
        )
        - 1
    )

    result[
        "Momentum 6M"
    ] = (
        result[
            "Close"
        ]
        / result[
            "Close"
        ].shift(
            MOMENTUM_6M_DAYS
        )
        - 1
    )

    result[
        "52W High"
    ] = (
        result[
            "High"
        ]
        .rolling(
            HIGH_52W_DAYS,
            min_periods=100,
        )
        .max()
    )

    result[
        "52W Drawdown"
    ] = (
        result[
            "Close"
        ]
        / result[
            "52W High"
        ]
        - 1
    )

    result[
        "Volume 20D Avg"
    ] = (
        result[
            "Volume"
        ]
        .rolling(
            VOLUME_PERIOD
        )
        .mean()
    )

    result[
        "Volume Ratio"
    ] = (
        result[
            "Volume"
        ]
        / result[
            "Volume 20D Avg"
        ].replace(
            0,
            np.nan,
        )
    )

    result[
        "Daily Return"
    ] = (
        result[
            "Close"
        ]
        .pct_change()
    )

    return result


# ============================================================
# 7) CROSS DETECTION
# ============================================================

def crossed_above(
    a: pd.Series,
    b: pd.Series,
) -> bool:

    if (
        len(a) < 2
        or len(b) < 2
    ):

        return False

    values = [
        a.iloc[-2],
        a.iloc[-1],
        b.iloc[-2],
        b.iloc[-1],
    ]

    if any(
        pd.isna(x)
        for x in values
    ):

        return False

    return (
        a.iloc[-2]
        <= b.iloc[-2]
        and
        a.iloc[-1]
        > b.iloc[-1]
    )


def crossed_below(
    a: pd.Series,
    b: pd.Series,
) -> bool:

    if (
        len(a) < 2
        or len(b) < 2
    ):

        return False

    values = [
        a.iloc[-2],
        a.iloc[-1],
        b.iloc[-2],
        b.iloc[-1],
    ]

    if any(
        pd.isna(x)
        for x in values
    ):

        return False

    return (
        a.iloc[-2]
        >= b.iloc[-2]
        and
        a.iloc[-1]
        < b.iloc[-1]
    )


# ============================================================
# 8) TREND STATE
# ============================================================

def classify_trend(
    price: float,
    ma50: float,
    ma200: float,
    momentum_6m: float,
) -> str:

    required = [
        price,
        ma50,
        ma200,
    ]

    if any(
        pd.isna(x)
        for x in required
    ):

        return "UNKNOWN"

    if (
        price > ma50
        and ma50 > ma200
        and (
            pd.isna(
                momentum_6m
            )
            or momentum_6m > 0
        )
    ):

        return "STRONG_UPTREND"

    if (
        price > ma200
        and ma50 > ma200
    ):

        return "UPTREND"

    if (
        price < ma50
        and ma50 < ma200
    ):

        return "STRONG_DOWNTREND"

    if price < ma200:

        return "DOWNTREND"

    return "NEUTRAL"


# ============================================================
# 9) RSI STATE
# ============================================================

def classify_rsi(
    rsi: float,
) -> str:

    if pd.isna(
        rsi
    ):

        return "UNKNOWN"

    if rsi >= 75:

        return "EXTREME_OVERBOUGHT"

    if rsi >= 70:

        return "OVERBOUGHT"

    if rsi <= 25:

        return "EXTREME_OVERSOLD"

    if rsi <= 30:

        return "OVERSOLD"

    if rsi >= 55:

        return "BULLISH"

    if rsi <= 45:

        return "BEARISH"

    return "NEUTRAL"


# ============================================================
# 10) STOCHASTIC STATE
# ============================================================

def classify_stochastic(
    k: float,
    d: float,
) -> str:

    if (
        pd.isna(k)
        or pd.isna(d)
    ):

        return "UNKNOWN"

    if (
        k <= 20
        and d <= 20
    ):

        return "OVERSOLD"

    if (
        k >= 80
        and d >= 80
    ):

        return "OVERBOUGHT"

    if k > d:

        return "BULLISH"

    if k < d:

        return "BEARISH"

    return "NEUTRAL"


# ============================================================
# 11) DRAWDOWN STATE
# ============================================================

def classify_drawdown(
    drawdown: float,
) -> str:

    if pd.isna(
        drawdown
    ):

        return "UNKNOWN"

    if drawdown <= -0.40:

        return "DEEP_CRASH"

    if drawdown <= -0.30:

        return "DEEP_DRAWDOWN"

    if drawdown <= -0.20:

        return "MAJOR_DRAWDOWN"

    if drawdown <= -0.10:

        return "CORRECTION"

    if drawdown >= -0.03:

        return "NEAR_52W_HIGH"

    return "NORMAL"


# ============================================================
# 12) COMPONENT SCORES
# ============================================================

def trend_score(
    price: float,
    ma50: float,
    ma200: float,
) -> float:

    if any(
        pd.isna(x)
        for x in [
            price,
            ma50,
            ma200,
        ]
    ):

        return 0.0

    score = 0.0

    if price > ma200:

        score += 12.0

    if price > ma50:

        score += 8.0

    if ma50 > ma200:

        score += 10.0

    return score


def momentum_score(
    momentum_3m: float,
    momentum_6m: float,
) -> float:

    score = 0.0

    if pd.notna(
        momentum_3m
    ):

        if momentum_3m >= 0.15:

            score += 10

        elif momentum_3m >= 0.05:

            score += 8

        elif momentum_3m >= 0:

            score += 6

        elif momentum_3m >= -0.10:

            score += 3

    if pd.notna(
        momentum_6m
    ):

        if momentum_6m >= 0.25:

            score += 10

        elif momentum_6m >= 0.10:

            score += 8

        elif momentum_6m >= 0:

            score += 6

        elif momentum_6m >= -0.15:

            score += 3

    return score


def rsi_score(
    rsi: float,
) -> float:

    if pd.isna(
        rsi
    ):

        return 0.0

    if 50 <= rsi <= 65:

        return 15.0

    if 40 <= rsi < 50:

        return 12.0

    if 30 <= rsi < 40:

        return 10.0

    if 65 < rsi <= 70:

        return 10.0

    if rsi < 30:

        return 8.0

    if 70 < rsi <= 75:

        return 6.0

    return 3.0


def stochastic_score(
    k: float,
    d: float,
    bullish_cross: bool,
    bearish_cross: bool,
) -> float:

    if (
        pd.isna(k)
        or pd.isna(d)
    ):

        return 0.0

    score = 0.0

    if bullish_cross:

        score += 10.0

    elif bearish_cross:

        score += 0.0

    elif k > d:

        score += 7.0

    else:

        score += 4.0

    if (
        k <= 30
        and k > d
    ):

        score += 5.0

    elif (
        k >= 80
        and k < d
    ):

        score += 0.0

    else:

        score += 3.0

    return min(
        score,
        15.0,
    )


def volume_score(
    volume_ratio: float,
    daily_return: float,
) -> float:

    if pd.isna(
        volume_ratio
    ):

        return 0.0

    if (
        volume_ratio >= 1.5
        and daily_return > 0
    ):

        return 10.0

    if (
        volume_ratio >= 1.2
        and daily_return > 0
    ):

        return 8.0

    if volume_ratio >= 0.8:

        return 6.0

    return 4.0


# ============================================================
# 13) TECHNICAL SCORE
# ============================================================

def calculate_total_technical_score(
    trend_points: float,
    momentum_points: float,
    rsi_points: float,
    stochastic_points: float,
    volume_points: float,
) -> float:

    # Trend       30
    # Momentum    20
    # RSI         15
    # Stochastic  15
    # Volume      10
    #
    # 현재 합계 최대 90점이므로
    # 100점 만점으로 normalize

    raw = (
        trend_points
        + momentum_points
        + rsi_points
        + stochastic_points
        + volume_points
    )

    normalized = (
        raw
        / 90.0
        * 100.0
    )

    return float(
        np.clip(
            normalized,
            0,
            100,
        )
    )


# ============================================================
# 14) TECHNICAL GRADE
# ============================================================

def technical_grade(
    score: float,
) -> str:

    if pd.isna(
        score
    ):

        return "N/A"

    if score >= 85:

        return "A"

    if score >= 75:

        return "B+"

    if score >= 65:

        return "B"

    if score >= 55:

        return "C+"

    if score >= 45:

        return "C"

    if score >= 35:

        return "D"

    return "F"


# ============================================================
# 15) TECHNICAL STATE
# ============================================================

def technical_state(
    price: float,
    ma50: float,
    ma200: float,
    rsi: float,
    momentum_6m: float,
    drawdown: float,
    bullish_cross: bool,
    bearish_cross: bool,
) -> str:

    if (
        pd.notna(rsi)
        and rsi >= 75
        and pd.notna(drawdown)
        and drawdown >= -0.05
    ):

        return "OVERHEATED"

    if (
        bullish_cross
        and pd.notna(drawdown)
        and drawdown <= -0.10
    ):

        return "RECOVERY"

    if (
        pd.notna(rsi)
        and rsi <= 30
    ):

        return "OVERSOLD"

    if (
        pd.notna(price)
        and pd.notna(ma50)
        and pd.notna(ma200)
        and price > ma50 > ma200
        and pd.notna(momentum_6m)
        and momentum_6m > 0
    ):

        return "STRONG_UPTREND"

    if (
        pd.notna(price)
        and pd.notna(ma200)
        and price > ma200
    ):

        return "UPTREND"

    if (
        bearish_cross
        and pd.notna(price)
        and pd.notna(ma50)
        and price < ma50
    ):

        return "WEAKENING"

    if (
        pd.notna(price)
        and pd.notna(ma50)
        and pd.notna(ma200)
        and price < ma50 < ma200
    ):

        return "STRONG_DOWNTREND"

    if (
        pd.notna(price)
        and pd.notna(ma200)
        and price < ma200
    ):

        return "DOWNTREND"

    return "NEUTRAL"


# ============================================================
# 16) ANALYZE ONE STOCK
# ============================================================

def analyze_stock(
    ticker: str,
    prices: pd.DataFrame,
) -> dict[str, t.Any]:

    output = {

        "Ticker":
            ticker,

        "Technical Data Quality":
            "FAILED",

        "Price":
            np.nan,

        "Daily Return":
            np.nan,

        "MA50":
            np.nan,

        "MA200":
            np.nan,

        "Price vs MA50":
            np.nan,

        "Price vs MA200":
            np.nan,

        "RSI14":
            np.nan,

        "Stoch K":
            np.nan,

        "Stoch D":
            np.nan,

        "Stoch Bullish Cross":
            False,

        "Stoch Bearish Cross":
            False,

        "Momentum 3M":
            np.nan,

        "Momentum 6M":
            np.nan,

        "52W High":
            np.nan,

        "52W Drawdown":
            np.nan,

        "Volume":
            np.nan,

        "Volume 20D Avg":
            np.nan,

        "Volume Ratio":
            np.nan,

        "Trend State":
            "UNKNOWN",

        "RSI State":
            "UNKNOWN",

        "Stochastic State":
            "UNKNOWN",

        "Drawdown State":
            "UNKNOWN",

        "Trend Score":
            0.0,

        "Momentum Score":
            0.0,

        "RSI Score":
            0.0,

        "Stochastic Score":
            0.0,

        "Volume Score":
            0.0,

        "Technical Score":
            0.0,

        "Technical Grade":
            "N/A",

        "Technical State":
            "UNKNOWN",

        "Technical Failed Reason":
            None,
    }

    try:

        df = extract_ticker_frame(
            prices,
            ticker,
        )

        if len(df) < MA_LONG:

            output[
                "Technical Data Quality"
            ] = "PARTIAL"

            output[
                "Technical Failed Reason"
            ] = (
                "MA200 계산에 필요한 "
                "가격 데이터가 부족합니다."
            )

            return output

        df = add_indicators(
            df
        )

        latest = df.iloc[-1]

        price = latest[
            "Close"
        ]

        ma50 = latest[
            "MA50"
        ]

        ma200 = latest[
            "MA200"
        ]

        rsi = latest[
            "RSI14"
        ]

        k = latest[
            "Stoch K"
        ]

        d = latest[
            "Stoch D"
        ]

        momentum_3m = latest[
            "Momentum 3M"
        ]

        momentum_6m = latest[
            "Momentum 6M"
        ]

        drawdown = latest[
            "52W Drawdown"
        ]

        bullish_cross = (
            crossed_above(
                df[
                    "Stoch K"
                ],
                df[
                    "Stoch D"
                ],
            )
            and
            pd.notna(k)
            and k <= 30
        )

        bearish_cross = (
            crossed_below(
                df[
                    "Stoch K"
                ],
                df[
                    "Stoch D"
                ],
            )
            and
            pd.notna(k)
            and k >= 70
        )

        t_score = trend_score(
            price,
            ma50,
            ma200,
        )

        m_score = momentum_score(
            momentum_3m,
            momentum_6m,
        )

        r_score = rsi_score(
            rsi
        )

        s_score = stochastic_score(
            k,
            d,
            bullish_cross,
            bearish_cross,
        )

        v_score = volume_score(
            latest[
                "Volume Ratio"
            ],
            latest[
                "Daily Return"
            ],
        )

        total_score = (
            calculate_total_technical_score(
                t_score,
                m_score,
                r_score,
                s_score,
                v_score,
            )
        )

        output.update(
            {

                "Technical Data Quality":
                    "OK",

                "Price":
                    price,

                "Daily Return":
                    latest[
                        "Daily Return"
                    ],

                "MA50":
                    ma50,

                "MA200":
                    ma200,

                "Price vs MA50":
                    (
                        price / ma50 - 1
                        if pd.notna(ma50)
                        and ma50 != 0
                        else np.nan
                    ),

                "Price vs MA200":
                    (
                        price / ma200 - 1
                        if pd.notna(ma200)
                        and ma200 != 0
                        else np.nan
                    ),

                "RSI14":
                    rsi,

                "Stoch K":
                    k,

                "Stoch D":
                    d,

                "Stoch Bullish Cross":
                    bullish_cross,

                "Stoch Bearish Cross":
                    bearish_cross,

                "Momentum 3M":
                    momentum_3m,

                "Momentum 6M":
                    momentum_6m,

                "52W High":
                    latest[
                        "52W High"
                    ],

                "52W Drawdown":
                    drawdown,

                "Volume":
                    latest[
                        "Volume"
                    ],

                "Volume 20D Avg":
                    latest[
                        "Volume 20D Avg"
                    ],

                "Volume Ratio":
                    latest[
                        "Volume Ratio"
                    ],

                "Trend State":
                    classify_trend(
                        price,
                        ma50,
                        ma200,
                        momentum_6m,
                    ),

                "RSI State":
                    classify_rsi(
                        rsi
                    ),

                "Stochastic State":
                    classify_stochastic(
                        k,
                        d,
                    ),

                "Drawdown State":
                    classify_drawdown(
                        drawdown
                    ),

                "Trend Score":
                    t_score,

                "Momentum Score":
                    m_score,

                "RSI Score":
                    r_score,

                "Stochastic Score":
                    s_score,

                "Volume Score":
                    v_score,

                "Technical Score":
                    round(
                        total_score,
                        2,
                    ),

                "Technical Grade":
                    technical_grade(
                        total_score
                    ),

                "Technical State":
                    technical_state(
                        price,
                        ma50,
                        ma200,
                        rsi,
                        momentum_6m,
                        drawdown,
                        bullish_cross,
                        bearish_cross,
                    ),
            }
        )

    except Exception as exc:

        output[
            "Technical Failed Reason"
        ] = (
            f"{type(exc).__name__}: "
            f"{exc}"
        )

    return output


# ============================================================
# 17) RUN ENGINE
# ============================================================

def run_technical_engine(
    fundamentals: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:

    tickers = (
        fundamentals[
            "Ticker"
        ]
        .dropna()
        .astype(str)
        .tolist()
    )

    rows = []

    print()
    print(
        "📈 Technical Engine 실행..."
    )

    total = len(
        tickers
    )

    for index, ticker in enumerate(
        tickers,
        start=1,
    ):

        row = analyze_stock(
            ticker,
            prices,
        )

        rows.append(
            row
        )

        if (
            index % 25 == 0
            or index == total
        ):

            print(
                f"Technical 분석: "
                f"{index}/{total}"
            )

    technical_df = pd.DataFrame(
        rows
    )

    return fundamentals.merge(
        technical_df,
        on="Ticker",
        how="left",
    )


# ============================================================
# 18) TECHNICAL RANK
# ============================================================

def add_technical_rank(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = df.copy()

    result[
        "Technical Rank"
    ] = (
        result[
            "Technical Score"
        ]
        .rank(
            ascending=False,
            method="min",
        )
        .astype(
            "Int64"
        )
    )

    return result


# ============================================================
# 19) SAVE
# ============================================================

def save_phase3(
    df: pd.DataFrame,
) -> Path:

    today = datetime.now().strftime(
        "%Y-%m-%d"
    )

    path = (
        DATA_DIR
        / (
            "sp500_technical_scores_"
            f"{today}.csv"
        )
    )

    output = df.sort_values(
        by=[
            "Fundamental Score",
            "Technical Score",
        ],
        ascending=[
            False,
            False,
        ],
        na_position="last",
    )

    output.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )

    return path


# ============================================================
# 20) PRINT SUMMARY
# ============================================================

def print_summary(
    df: pd.DataFrame,
) -> None:

    print()
    print(
        "=" * 80
    )

    print(
        "TECHNICAL ENGINE SUMMARY"
    )

    print(
        "=" * 80
    )

    print(
        f"총 종목: {len(df)}"
    )

    print(
        f"평균 Technical Score: "
        f"{df['Technical Score'].mean():.2f}"
    )

    print(
        f"70점 이상: "
        f"{(df['Technical Score'] >= 70).sum()}"
    )

    print(
        f"80점 이상: "
        f"{(df['Technical Score'] >= 80).sum()}"
    )

    print()

    print(
        "Technical State:"
    )

    print(
        df[
            "Technical State"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print()


# ============================================================
# 21) PRINT TOP STOCKS
# ============================================================

def print_top_stocks(
    df: pd.DataFrame,
    top_n: int = 20,
) -> None:

    columns = [
        "Ticker",
        "Short Name",
        "Fundamental Score",
        "Technical Score",
        "Technical Grade",
        "Technical State",
        "Trend State",
        "Price",
        "RSI14",
        "Momentum 6M",
        "52W Drawdown",
    ]

    columns = [
        column
        for column in columns
        if column in df.columns
    ]

    top = (
        df
        .sort_values(
            "Technical Score",
            ascending=False,
        )
        .head(
            top_n
        )
    )

    print(
        "=" * 130
    )

    print(
        f"TOP {top_n} TECHNICAL STOCKS"
    )

    print(
        "=" * 130
    )

    print(
        top[
            columns
        ].to_string(
            index=False
        )
    )

    print()


# ============================================================
# 22) MAIN
# ============================================================

def main() -> None:

    print()
    print(
        "=============================================="
    )

    print(
        "        AI VALUE STOCK RADAR"
    )

    print(
        "        PHASE 3 - TECHNICAL ENGINE"
    )

    print(
        "=============================================="
    )

    print()

    fundamentals, source_path = (
        load_phase2_data()
    )

    tickers = (
        fundamentals[
            "Ticker"
        ]
        .dropna()
        .astype(str)
        .tolist()
    )

    print(
        f"✅ 분석 대상: "
        f"{len(tickers)}개"
    )

    prices = download_price_data(
        tickers
    )

    result = run_technical_engine(
        fundamentals,
        prices,
    )

    result = add_technical_rank(
        result
    )

    output_path = save_phase3(
        result
    )

    print_summary(
        result
    )

    print_top_stocks(
        result,
        top_n=20,
    )

    print(
        f"📂 Source: "
        f"{source_path}"
    )

    print(
        f"💾 저장 완료: "
        f"{output_path}"
    )

    print()

    print(
        "✅ PHASE 3 COMPLETE"
    )

    print()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()