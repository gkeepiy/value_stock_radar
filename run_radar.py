from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

# 주간 Fundamental 갱신 요일
# Monday=0, Tuesday=1, ... Sunday=6
FUNDAMENTAL_UPDATE_WEEKDAY = 0

PHASE1 = "phase1_data.py"
PHASE2 = "phase2_fundamental.py"
PHASE3 = "phase3_technical.py"
PHASE4 = "phase4_signal.py"


# ============================================================
# HELPERS
# ============================================================

def now_text() -> str:
    return datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def run_phase(
    filename: str,
) -> None:

    script_path = (
        PROJECT_DIR
        / filename
    )

    if not script_path.exists():

        raise FileNotFoundError(
            f"{filename} 파일을 찾을 수 없습니다."
        )

    print()
    print("=" * 70)

    print(
        f"▶ 실행: {filename}"
    )

    print("=" * 70)
    print()

    start = time.time()

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
        ],
        cwd=PROJECT_DIR,
    )

    elapsed = (
        time.time()
        - start
    )

    if result.returncode != 0:

        raise RuntimeError(
            f"{filename} 실행 실패 "
            f"(exit code: {result.returncode})"
        )

    print()
    print(
        f"✅ {filename} 완료 "
        f"({elapsed / 60:.1f}분)"
    )


# ============================================================
# FILE CHECKS
# ============================================================

def latest_file(
    pattern: str,
) -> Path | None:

    data_dir = (
        PROJECT_DIR
        / "data"
    )

    files = list(
        data_dir.glob(
            pattern
        )
    )

    if not files:

        return None

    return max(
        files,
        key=lambda p:
            p.stat().st_mtime,
    )


def phase1_data_exists() -> bool:

    return (
        latest_file(
            "sp500_raw_*.csv"
        )
        is not None
    )


def phase2_data_exists() -> bool:

    return (
        latest_file(
            "sp500_fundamental_scores_*.csv"
        )
        is not None
    )


# ============================================================
# FUNDAMENTAL UPDATE CHECK
# ============================================================

def should_update_fundamental() -> bool:

    today = datetime.now()

    # --------------------------------------------------------
    # Phase 1 또는 Phase 2 파일이 없으면
    # 무조건 Fundamental 전체 갱신
    # --------------------------------------------------------

    if not phase1_data_exists():

        print(
            "ℹ️ Phase 1 데이터가 없습니다."
        )

        return True

    if not phase2_data_exists():

        print(
            "ℹ️ Phase 2 데이터가 없습니다."
        )

        return True

    # --------------------------------------------------------
    # 지정 요일이면 갱신
    # --------------------------------------------------------

    if (
        today.weekday()
        == FUNDAMENTAL_UPDATE_WEEKDAY
    ):

        return True

    return False


# ============================================================
# WEEKLY PIPELINE
# ============================================================

def run_weekly_pipeline() -> None:

    print()
    print(
        "📅 주간 Fundamental 갱신 실행"
    )

    print()

    run_phase(
        PHASE1
    )

    run_phase(
        PHASE2
    )

    run_phase(
        PHASE3
    )

    run_phase(
        PHASE4
    )


# ============================================================
# DAILY PIPELINE
# ============================================================

def run_daily_pipeline() -> None:

    print()
    print(
        "📈 Daily Technical / Signal 갱신 실행"
    )

    print()

    run_phase(
        PHASE3
    )

    run_phase(
        PHASE4
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    total_start = (
        time.time()
    )

    print()
    print("=" * 70)

    print(
        "           VALUE STOCK RADAR"
    )

    print(
        "          AUTOMATED PIPELINE"
    )

    print("=" * 70)

    print()

    print(
        f"시작: {now_text()}"
    )

    print()

    try:

        # ----------------------------------------------------
        # Fundamental 갱신 여부 결정
        # ----------------------------------------------------

        if should_update_fundamental():

            run_weekly_pipeline()

        else:

            run_daily_pipeline()

    except Exception as exc:

        print()
        print("=" * 70)

        print(
            "❌ RADAR 실행 중단"
        )

        print("=" * 70)

        print()

        print(
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        print()

        sys.exit(1)

    total_elapsed = (
        time.time()
        - total_start
    )

    print()
    print("=" * 70)

    print(
        "✅ VALUE STOCK RADAR COMPLETE"
    )

    print("=" * 70)

    print()

    print(
        f"완료: {now_text()}"
    )

    print(
        f"총 실행 시간: "
        f"{total_elapsed / 60:.1f}분"
    )

    print()

    print(
        "Radar 화면:"
    )

    print(
        r".\.venv\Scripts\python.exe "
        r"-m streamlit run app.py"
    )

    print()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()