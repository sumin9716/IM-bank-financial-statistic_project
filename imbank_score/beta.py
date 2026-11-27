"""기준년월·업종별 외부 베타(업종별 월별 신용공여 추이 자료)를 데이터에 붙인다."""

import pandas as pd

MONTH_COL = "기준년월"
INDUSTRY_COL = "업종_대분류"

# 2022년 1~3월 베타는 쓰지 않고 같은 업종의 2022년 4월 값으로 대신한다.
# notebooks/외부데이터베타포함모델링.ipynb도 같은 방식으로 1~3월을 채운다.
REFERENCE_MONTH = 202204
BACKFILL_MONTHS = (202201, 202202, 202203)

# 베타를 찾지 못한 행은 보정하지 않는다(β=1).
DEFAULT_BETA = 1.0


def normalize_month(value) -> str:
    """'2022-04', '2022.04', 202204를 모두 '202204'로 맞춘다."""
    return str(value).strip().replace("-", "").replace(".", "")


def normalize_industry(value) -> str:
    return str(value).strip().replace("  ", " ")


def load_industry_beta(beta_wide: pd.DataFrame) -> pd.DataFrame:
    """'기준년월' 열과 업종별 열로 된 표를 (기준년월, 업종_대분류, beta) 긴 표로 바꾼다.

    결과의 기준년월은 df와 병합하기 위해 'YYYYMM' 문자열이다.
    """
    beta = beta_wide.melt(id_vars=[MONTH_COL], var_name=INDUSTRY_COL, value_name="beta")
    beta = beta.dropna(subset=["beta"])
    beta[MONTH_COL] = beta[MONTH_COL].map(normalize_month).astype(int)
    beta[INDUSTRY_COL] = beta[INDUSTRY_COL].map(normalize_industry)

    reference = (
        beta.loc[beta[MONTH_COL] == REFERENCE_MONTH, [INDUSTRY_COL, "beta"]]
        .set_index(INDUSTRY_COL)["beta"]
        .to_dict()
    )
    backfill = pd.DataFrame(
        [
            {MONTH_COL: month, INDUSTRY_COL: industry, "beta": value}
            for industry, value in reference.items()
            for month in BACKFILL_MONTHS
        ]
    )
    beta = pd.concat([beta.loc[~beta[MONTH_COL].isin(BACKFILL_MONTHS)], backfill], ignore_index=True)
    beta = beta.sort_values([INDUSTRY_COL, MONTH_COL]).reset_index(drop=True)
    beta[MONTH_COL] = beta[MONTH_COL].astype(str)
    return beta


def attach_beta(df: pd.DataFrame, beta_long: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """df의 기준년월·업종_대분류 표기를 정규화한 뒤 행마다 beta를 붙인다.

    반환값은 (beta 열이 붙은 df, 베타를 찾지 못한 행 마스크). 찾지 못한 행은 DEFAULT_BETA를 쓴다.
    같은 (기준년월, 업종)이 beta_long에 두 번 있으면 MergeError가 난다.
    """
    out = df.copy()
    out[MONTH_COL] = out[MONTH_COL].map(normalize_month)
    out[INDUSTRY_COL] = out[INDUSTRY_COL].map(normalize_industry)
    out = out.merge(beta_long, on=[MONTH_COL, INDUSTRY_COL], how="left", validate="m:1")

    unmatched = out["beta"].isna()
    out["beta"] = pd.to_numeric(out["beta"], errors="coerce").fillna(DEFAULT_BETA)
    return out, unmatched
