"""업종별 파라미터 모델: 10개 업종마다 따로 정한 가중치·λ로 업종 안에서 점수를 내고,
기준년월·업종별 외부 베타를 행마다 곱한다. 그 밖의 업종은 점수가 비어 있다(NaN).

    python score_industry.py               # data/ -> outputs/score_beta.csv
    python score_industry.py --no-beta     # 베타 없이(β=1) -> outputs/score_nobeta.csv
"""

import argparse
from pathlib import Path

from imbank_score.beta import attach_beta, load_industry_beta
from imbank_score.files import BETA_PATH, OUTPUT_DIR, RAW_DATA_PATH, read_csv, write_scores
from imbank_score.params import INDUSTRY_AMOUNT_START_COL, INDUSTRY_FEATURES, INDUSTRY_PARAMS
from imbank_score.preprocess import prepare_features
from imbank_score.scoring import score_by_industry


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="업종별 파라미터와 외부 베타로 0값 패널티 점수를 계산한다.")
    parser.add_argument("--data", type=Path, default=RAW_DATA_PATH, help="교육용 데이터 CSV (cp949)")
    parser.add_argument("--beta", type=Path, default=BETA_PATH, help="기준년월 x 업종 베타 CSV (cp949)")
    parser.add_argument("--no-beta", action="store_true", help="외부 베타를 쓰지 않는다(β=1)")
    parser.add_argument("--output", type=Path, help="기본값: outputs/score_beta.csv, --no-beta면 outputs/score_nobeta.csv")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    df = read_csv(args.data)

    if not args.no_beta:
        df, unmatched = attach_beta(df, load_industry_beta(read_csv(args.beta)))
        print(f"[β 매칭 실패율] {unmatched.mean():.2%}")
        if unmatched.any():
            print(df.loc[unmatched, ["기준년월", "업종_대분류"]].drop_duplicates().head(10))

    df = prepare_features(df, amount_start_col=INDUSTRY_AMOUNT_START_COL)
    scores, logs = score_by_industry(
        df, INDUSTRY_FEATURES, INDUSTRY_PARAMS, beta_col=None if args.no_beta else "beta"
    )
    for line in logs:
        print(line)

    output = args.output or OUTPUT_DIR / ("score_nobeta.csv" if args.no_beta else "score_beta.csv")
    write_scores(scores, output)
    print(scores.describe())


if __name__ == "__main__":
    main()
