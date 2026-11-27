"""원본 데이터 전처리: 구간형 좌수·건수의 숫자화, 파생 피처, 범주형 열의 숫자화."""

import re
from collections.abc import Iterable
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import pandas as pd

ACCOUNT_COUNT_COLUMNS = [
    "요구불예금좌수", "거치식예금좌수", "적립식예금좌수", "수익증권좌수", "신탁좌수", "퇴직연금좌수",
    "여신_운전자금대출좌수", "여신_시설자금대출좌수", "신용카드개수",
]
TRANSACTION_COUNT_COLUMNS = [
    "외환_수출실적거래건수", "외환_수입실적거래건수", "창구거래건수", "인터넷뱅킹거래건수",
    "스마트뱅킹거래건수", "폰뱅킹거래건수", "ATM거래건수", "자동이체거래건수",
]
ONLINE_TRANSACTION_COLUMNS = ["인터넷뱅킹거래건수", "스마트뱅킹거래건수", "폰뱅킹거래건수", "자동이체거래건수"]
OFFLINE_TRANSACTION_COLUMNS = ["창구거래건수", "ATM거래건수"]

# 잔액 열은 이름이 아니라 원본 CSV의 열 위치로 합산한다. 열 순서가 다른 자료에는 그대로 쓸 수 없다.
DEPOSIT_BALANCE_POSITIONS = slice(7, 10)
INVESTMENT_BALANCE_POSITIONS = slice(10, 13)
LOAN_BALANCE_POSITIONS = slice(14, 16)

GRADE_CODES = {"일반": 1, "우수": 2, "최우수": 3}
DEDICATED_CODES = {"Y": 1, "N": 0}

_EXACT = re.compile(r"(\d+)(개|건)")
_BOUNDED = re.compile(r"(\d+)(개|건)초과(\d+)(개|건)이하")
_OPEN_ENDED = re.compile(r"(\d+)(개|건)초과")


def require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"필요한 컬럼이 없습니다: {missing}")


def round_half_up(x: float) -> int:
    """.5를 항상 올린다(2.5 -> 3). 내장 round()는 짝수 쪽으로 반올림하므로 쓰지 않는다."""
    return int(Decimal(str(x)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def parse_count_range(value, open_end_add: int = 5, return_int: bool = True) -> float:
    """구간으로 공개된 좌수·건수를 숫자 하나로 바꾼다.

    'N개'/'N건' -> N, 'N개초과M개이하' -> 구간 중앙값, 'N개초과' -> N + open_end_add.
    공백은 무시하며, 그 밖의 표기와 결측은 NaN이 된다.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan

    s = re.sub(r"\s+", "", str(value))
    if m := _EXACT.fullmatch(s):
        out = float(m.group(1))
    elif m := _BOUNDED.fullmatch(s):
        mid = (int(m.group(1)) + int(m.group(3))) / 2.0
        out = round_half_up(mid) if return_int else mid
    elif m := _OPEN_ENDED.fullmatch(s):
        out = int(m.group(1)) + int(open_end_add)
    else:
        return np.nan
    return int(out) if return_int else float(out)


def convert_count_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns = ACCOUNT_COUNT_COLUMNS + TRANSACTION_COUNT_COLUMNS
    require_columns(df, columns)
    out = df.copy()
    for col in columns:
        out[col] = out[col].apply(parse_count_range)
    return out


def add_derived_columns(df: pd.DataFrame, amount_start_col: str) -> pd.DataFrame:
    """스코어 피처로 쓰는 잔액·비중·좌수·거래 열을 덧붙인다.

    거래금액은 amount_start_col부터 '요구불출금금액'까지의 열을 합한 값이다.
    """
    require_columns(
        df,
        ["여신_운전자금대출좌수", "신용카드개수", amount_start_col, "요구불출금금액",
         *ONLINE_TRANSACTION_COLUMNS, *OFFLINE_TRANSACTION_COLUMNS],
    )
    out = df.copy()
    deposit = out.iloc[:, DEPOSIT_BALANCE_POSITIONS].sum(axis=1)
    investment = out.iloc[:, INVESTMENT_BALANCE_POSITIONS].sum(axis=1)
    loan = out.iloc[:, LOAN_BALANCE_POSITIONS].sum(axis=1)
    total = (deposit + investment + loan).replace(0, np.nan)

    out["예금잔액"] = deposit
    out["투자잔액"] = investment
    out["여신대출잔액"] = loan
    # 잔액이 모두 0인 고객은 비중을 0으로 둔다.
    out["예금비중"] = (deposit / total).fillna(0.0)
    out["대출비중"] = (loan / total).fillna(0.0)
    out["여신_신용카드_좌수"] = out.loc[:, "여신_운전자금대출좌수":"신용카드개수"].sum(axis=1)

    online = out[ONLINE_TRANSACTION_COLUMNS].sum(axis=1, min_count=1).fillna(0)
    offline = out[OFFLINE_TRANSACTION_COLUMNS].sum(axis=1, min_count=1).fillna(0)
    out["거래건수"] = online + offline
    out["거래금액"] = out.loc[:, amount_start_col:"요구불출금금액"].sum(axis=1)
    return out


def encode_grade_and_dedicated(df: pd.DataFrame) -> pd.DataFrame:
    """법인_고객등급(일반/우수/최우수 -> 1/2/3)과 전담고객여부(Y/N -> 1/0)를 숫자로 바꾼다."""
    out = df.copy()
    if "법인_고객등급" in out.columns and not pd.api.types.is_numeric_dtype(out["법인_고객등급"]):
        out["법인_고객등급"] = out["법인_고객등급"].astype(str).str.strip().map(GRADE_CODES)
    if "전담고객여부" in out.columns and not pd.api.types.is_numeric_dtype(out["전담고객여부"]):
        out["전담고객여부"] = (
            out["전담고객여부"].astype(str).str.strip().str.upper().map(DEDICATED_CODES)
        )
    return out


def prepare_features(df: pd.DataFrame, amount_start_col: str) -> pd.DataFrame:
    """구간 숫자화 -> 파생 피처 -> 범주형 숫자화를 차례로 적용한다."""
    df = convert_count_columns(df)
    df = add_derived_columns(df, amount_start_col)
    return encode_grade_and_dedicated(df)
