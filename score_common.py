"""공통 파라미터 모델: 전 업종에 같은 가중치·λ·β를 적용해 0값 패널티 점수를 낸다.

    python score_common.py                       # data/ -> outputs/score_all.csv
    python score_common.py --data PATH --output PATH
"""

import argparse
from pathlib import Path

from imbank_score.files import OUTPUT_DIR, RAW_DATA_PATH, read_csv, write_scores
from imbank_score.params import (
    COMMON_AMOUNT_START_COL,
    COMMON_BETA,
    COMMON_FEATURES,
    COMMON_LAM,
    COMMON_SCORE_SHIFT,
    COMMON_WEIGHTS,
)
from imbank_score.preprocess import prepare_features
from imbank_score.scoring import zero_penalty_score


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="전 업종 공통 파라미터로 0값 패널티 점수를 계산한다.")
    parser.add_argument("--data", type=Path, default=RAW_DATA_PATH, help="교육용 데이터 CSV (cp949)")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "score_all.csv")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    df = prepare_features(read_csv(args.data), amount_start_col=COMMON_AMOUNT_START_COL)

    scores = zero_penalty_score(df, COMMON_FEATURES, COMMON_WEIGHTS, lam=COMMON_LAM, beta=COMMON_BETA)
    scores = scores + COMMON_SCORE_SHIFT
    write_scores(scores, args.output)

    print(f"[INFO] 사용 피처: {COMMON_FEATURES}")
    print(scores.describe())


if __name__ == "__main__":
    main()
