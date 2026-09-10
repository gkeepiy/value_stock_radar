import time
import random
import typing as t
from datetime import datetime
from pathlib import Path
from io import StringIO

import pandas as pd
import requests
import yfinance as yf
from yfinance.exceptions import YFRateLimitError


# ============================================================
# 0) CONFIG
# ============================================================

SP500_CSV_URL = (
    "https://datahub.io/core/s-and-p-500-companies-financials/"
    "_r/-/data/constituents.csv"
)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0 Safari/537.36"
)

BASE_SLEEP = 0.8
RETRY = 6
BACKOFF_BASE = 1.8
COOLDOWN_SEC = 60

REVENUE_MAX_YEARS = 4

# 전체 S&P 500
MAX_TICKERS: t.Optional[int] = None

OUTPUT_DIR = Path("data")


CORE_FIELDS = [
    "Market Cap",
    "Revenue CAGR",
    "Forward PER",
    "PBR",
    "EV/EBITDA",
    "ROE",
    "ROA",
    "Debt/Equity",
    "Free Cash Flow",
]


NUMERIC_COLUMNS = [
    "Market Cap",
    "Revenue CAGR",
    "Revenue CAGR Years Used",
    "Revenue Growth",
    "Earnings Growth",
    "Forward PER",
    "PBR",
    "PSR",
    "EV/EBITDA",
    "ROE",
    "ROA",
    "Debt/Equity",
    "Debt/Equity Raw",
    "Operating Margin",
    "Profit Margin",
    "Free Cash Flow",
    "Operating Cash Flow",
    "FCF Yield",
]


# ============================================================
# 1) S&P 500 UNIVERSE
# ============================================================

def normalize_for_yfinance(symbol: str) -> str:
    symbol = (
        str(symbol)
        .upper()
        .strip()
    )

    return symbol.replace(
        ".",
        "-",
    )


def get_sp500_tickers() -> list[str]:

    print(
        "📡 S&P 500 구성종목 다운로드 중..."
    )

    response = requests.get(
        SP500_CSV_URL,
        headers={
            "User-Agent": USER_AGENT
        },
        timeout=30,
    )

    response.raise_for_status()

    df = pd.read_csv(
        StringIO(
            response.text
        )
    )

    if "Symbol" not in df.columns:

        raise ValueError(
            "S&P 500 데이터에 Symbol 컬럼이 없습니다."
        )

    tickers = (
        df["Symbol"]
        .dropna()
        .astype(str)
        .map(
            normalize_for_yfinance
        )
        .drop_duplicates()
        .tolist()
    )

    tickers = sorted(
        tickers
    )

    if MAX_TICKERS is not None:

        tickers = tickers[
            :MAX_TICKERS
        ]

    return tickers


# ============================================================
# 2) SAFE CALL / BACKOFF
# ============================================================

def call_with_backoff(
    fn: t.Callable[[], t.Any],
    *,
    retry: int = RETRY,
) -> t.Any:

    last_error = None

    for attempt in range(
        retry
    ):

        try:

            return fn()

        except YFRateLimitError as exc:

            last_error = exc

            wait = (
                COOLDOWN_SEC
                + random.random() * 5
            )

            print(
                f"⛔ Yahoo Rate Limit "
                f"- {wait:.1f}초 대기"
            )

            time.sleep(
                wait
            )

        except Exception as exc:

            last_error = exc

            wait = (
                BACKOFF_BASE ** attempt
                + random.random()
            )

            print(
                f"⚠️ 재시도 {attempt + 1}/{retry} "
                f"- {wait:.1f}초 대기"
            )

            time.sleep(
                wait
            )

    if last_error:

        raise last_error

    raise RuntimeError(
        "데이터 요청 실패"
    )


# ============================================================
# 3) HELPERS
# ============================================================

def to_float(
    value: t.Any,
) -> t.Optional[float]:

    if value is None:

        return None

    try:

        value = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):

        return None

    if pd.isna(
        value
    ):

        return None

    return value


def normalize_debt_to_equity(
    value: t.Any,
) -> t.Optional[float]:

    value = to_float(
        value
    )

    if value is None:

        return None

    return (
        value / 100.0
    )


# ============================================================
# 4) REVENUE CAGR
# ============================================================

def pick_revenue_row(
    df: pd.DataFrame,
) -> t.Optional[str]:

    if (
        df is None
        or df.empty
    ):

        return None

    names = [
        str(x).strip()
        for x in df.index
    ]

    lower_map = {
        name.lower(): name
        for name in names
    }

    candidates = [
        "Total Revenue",
        "Operating Revenue",
        "Revenue",
        "Revenues",
    ]

    for candidate in candidates:

        if candidate.lower() in lower_map:

            return lower_map[
                candidate.lower()
            ]

    for name in names:

        if "revenue" in name.lower():

            return name

    return None


def sort_series_index(
    series: pd.Series,
) -> pd.Series:

    try:

        return series.sort_index()

    except Exception:

        return series


def compute_revenue_cagr_best_effort(
    ticker_obj: yf.Ticker,
) -> tuple[
    t.Optional[float],
    t.Optional[int],
    str,
]:

    # --------------------------------------------------------
    # Quarterly
    # --------------------------------------------------------

    try:

        quarterly = (
            ticker_obj.get_income_stmt(
                freq="quarterly",
                pretty=True,
            )
        )

        if (
            quarterly is not None
            and not quarterly.empty
        ):

            row_name = (
                pick_revenue_row(
                    quarterly
                )
            )

            if row_name:

                revenue = (
                    pd.to_numeric(
                        quarterly.loc[
                            row_name
                        ],
                        errors="coerce",
                    )
                    .dropna()
                )

                revenue = (
                    sort_series_index(
                        revenue
                    )
                )

                if len(revenue) >= 8:

                    ttm = (
                        revenue
                        .rolling(4)
                        .sum()
                        .dropna()
                    )

                    if len(ttm) >= 5:

                        years = min(
                            REVENUE_MAX_YEARS,
                            (
                                len(ttm) - 1
                            ) // 4,
                        )

                        if years >= 1:

                            start = float(
                                ttm.iloc[
                                    -1
                                    - years * 4
                                ]
                            )

                            end = float(
                                ttm.iloc[-1]
                            )

                            if (
                                start > 0
                                and end > 0
                            ):

                                cagr = (
                                    end / start
                                ) ** (
                                    1 / years
                                ) - 1

                                return (
                                    cagr,
                                    years,
                                    "TTM",
                                )

    except Exception:

        pass

    # --------------------------------------------------------
    # Yearly fallback
    # --------------------------------------------------------

    try:

        yearly = (
            ticker_obj.get_income_stmt(
                freq="yearly",
                pretty=True,
            )
        )

        if (
            yearly is not None
            and not yearly.empty
        ):

            row_name = (
                pick_revenue_row(
                    yearly
                )
            )

            if row_name:

                revenue = (
                    pd.to_numeric(
                        yearly.loc[
                            row_name
                        ],
                        errors="coerce",
                    )
                    .dropna()
                )

                revenue = (
                    sort_series_index(
                        revenue
                    )
                )

                if len(revenue) >= 2:

                    years = min(
                        REVENUE_MAX_YEARS,
                        len(revenue) - 1,
                    )

                    start = float(
                        revenue.iloc[
                            -1 - years
                        ]
                    )

                    end = float(
                        revenue.iloc[-1]
                    )

                    if (
                        start > 0
                        and end > 0
                    ):

                        cagr = (
                            end / start
                        ) ** (
                            1 / years
                        ) - 1

                        return (
                            cagr,
                            years,
                            "Yearly",
                        )

    except Exception:

        pass

    return (
        None,
        None,
        "None",
    )


# ============================================================
# 5) EMPTY RECORD
# ============================================================

def empty_stock_record(
    ticker: str,
) -> dict:

    return {

        "Ticker":
            ticker,

        "Short Name":
            None,

        "Sector":
            "Unknown",

        "Industry":
            "Unknown",

        "Market Cap":
            None,

        "Revenue CAGR":
            None,

        "Revenue CAGR Years Used":
            None,

        "Revenue CAGR Method":
            "None",

        "Revenue Growth":
            None,

        "Earnings Growth":
            None,

        "Forward PER":
            None,

        "PBR":
            None,

        "PSR":
            None,

        "EV/EBITDA":
            None,

        "ROE":
            None,

        "ROA":
            None,

        "Debt/Equity Raw":
            None,

        "Debt/Equity":
            None,

        "Operating Margin":
            None,

        "Profit Margin":
            None,

        "Free Cash Flow":
            None,

        "Operating Cash Flow":
            None,

        "FCF Yield":
            None,

        "Data Quality":
            "FAILED",

        "Failed Reason":
            None,
    }


# ============================================================
# 6) NORMALIZE STOCK DATA
# ============================================================

def normalize_stock_data(
    record: dict,
) -> dict:

    result = dict(
        record
    )

    result[
        "Debt/Equity"
    ] = normalize_debt_to_equity(
        result.get(
            "Debt/Equity Raw"
        )
    )

    for column in NUMERIC_COLUMNS:

        if column == "Debt/Equity":

            continue

        result[
            column
        ] = to_float(
            result.get(
                column
            )
        )

    market_cap = result.get(
        "Market Cap"
    )

    fcf = result.get(
        "Free Cash Flow"
    )

    if (
        market_cap is not None
        and market_cap > 0
        and fcf is not None
    ):

        result[
            "FCF Yield"
        ] = (
            fcf
            / market_cap
        )

    return result


# ============================================================
# 7) DATA QUALITY
# ============================================================

def classify_data_quality(
    record: dict,
) -> str:

    if record.get(
        "Failed Reason"
    ):

        return "FAILED"

    valid_count = sum(

        record.get(
            field
        ) is not None

        for field
        in CORE_FIELDS
    )

    completeness = (
        valid_count
        / len(CORE_FIELDS)
    )

    if completeness >= 0.8:

        return "OK"

    return "PARTIAL"


# ============================================================
# 8) FETCH STOCK
# ============================================================

def fetch_stock_data(
    ticker: str,
) -> dict:

    record = empty_stock_record(
        ticker
    )

    try:

        ticker_obj = yf.Ticker(
            ticker
        )

        # ----------------------------------------------------
        # CAGR
        # --------------------------------------------------------

        (
            cagr,
            years,
            method,
        ) = call_with_backoff(

            lambda:
            compute_revenue_cagr_best_effort(
                ticker_obj
            )
        )

        record[
            "Revenue CAGR"
        ] = cagr

        record[
            "Revenue CAGR Years Used"
        ] = years

        record[
            "Revenue CAGR Method"
        ] = method

        # ----------------------------------------------------
        # Yahoo info
        # --------------------------------------------------------

        info = call_with_backoff(
            ticker_obj.get_info
        )

        record[
            "Short Name"
        ] = (
            info.get(
                "shortName"
            )
            or info.get(
                "longName"
            )
        )

        record[
            "Sector"
        ] = (
            info.get(
                "sector"
            )
            or "Unknown"
        )

        record[
            "Industry"
        ] = (
            info.get(
                "industry"
            )
            or "Unknown"
        )

        record[
            "Market Cap"
        ] = info.get(
            "marketCap"
        )

        record[
            "Revenue Growth"
        ] = info.get(
            "revenueGrowth"
        )

        record[
            "Earnings Growth"
        ] = info.get(
            "earningsGrowth"
        )

        record[
            "Forward PER"
        ] = info.get(
            "forwardPE"
        )

        record[
            "PBR"
        ] = info.get(
            "priceToBook"
        )

        record[
            "PSR"
        ] = info.get(
            "priceToSalesTrailing12Months"
        )

        record[
            "EV/EBITDA"
        ] = info.get(
            "enterpriseToEbitda"
        )

        record[
            "ROE"
        ] = info.get(
            "returnOnEquity"
        )

        record[
            "ROA"
        ] = info.get(
            "returnOnAssets"
        )

        record[
            "Debt/Equity Raw"
        ] = info.get(
            "debtToEquity"
        )

        record[
            "Operating Margin"
        ] = info.get(
            "operatingMargins"
        )

        record[
            "Profit Margin"
        ] = info.get(
            "profitMargins"
        )

        record[
            "Free Cash Flow"
        ] = info.get(
            "freeCashflow"
        )

        record[
            "Operating Cash Flow"
        ] = info.get(
            "operatingCashflow"
        )

    except Exception as exc:

        record[
            "Failed Reason"
        ] = (
            f"{type(exc).__name__}: "
            f"{exc}"
        )

    record = normalize_stock_data(
        record
    )

    record[
        "Data Quality"
    ] = classify_data_quality(
        record
    )

    return record


# ============================================================
# 9) COLLECT ALL
# ============================================================

def collect_all_stocks(
    tickers: list[str],
) -> pd.DataFrame:

    rows = []

    total = len(
        tickers
    )

    for index, ticker in enumerate(
        tickers,
        start=1,
    ):

        print(
            f"[{index}/{total}] "
            f"{ticker}"
        )

        record = fetch_stock_data(
            ticker
        )

        rows.append(
            record
        )

        print(
            f"   {record['Data Quality']}"
        )

        if record.get(
            "Failed Reason"
        ):

            print(
                f"   ❌ "
                f"{record['Failed Reason']}"
            )

        time.sleep(
            BASE_SLEEP
            + random.random() * 0.3
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 10) SAVE
# ============================================================

def save_to_csv(
    df: pd.DataFrame,
) -> Path:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    today = datetime.now().strftime(
        "%Y-%m-%d"
    )

    path = (
        OUTPUT_DIR
        / f"sp500_raw_{today}.csv"
    )

    df.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )

    return path


# ============================================================
# 11) MAIN
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
        "        PHASE 1 - DATA ENGINE"
    )

    print(
        "=============================================="
    )

    print()

    tickers = get_sp500_tickers()

    print(
        f"✅ S&P 500 구성종목:"
        f" {len(tickers)}개"
    )

    print()

    print(
        "📊 Fundamental 데이터 수집 시작..."
    )

    print()

    df = collect_all_stocks(
        tickers
    )

    output_path = save_to_csv(
        df
    )

    print()

    print(
        "=============================================="
    )

    print(
        "PHASE 1 RESULT"
    )

    print(
        "=============================================="
    )

    print()

    print(
        f"총 종목:"
        f" {len(df)}"
    )

    print(
        "OK:"
        f" {(df['Data Quality'] == 'OK').sum()}"
    )

    print(
        "PARTIAL:"
        f" {(df['Data Quality'] == 'PARTIAL').sum()}"
    )

    print(
        "FAILED:"
        f" {(df['Data Quality'] == 'FAILED').sum()}"
    )

    print()

    print(
        f"💾 저장 완료:"
        f" {output_path}"
    )

    print()

    print(
        "✅ PHASE 1 COMPLETE"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()