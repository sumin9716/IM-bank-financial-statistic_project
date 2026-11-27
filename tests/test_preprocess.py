import numpy as np
import pandas as pd
import pytest

from imbank_score.preprocess import (
    add_derived_columns,
    convert_count_columns,
    encode_grade_and_dedicated,
    parse_count_range,
    prepare_features,
    round_half_up,
)
from tests.synthetic import make_raw_frame


@pytest.mark.parametrize("x, expected", [(2.5, 3), (3.5, 4), (2.4, 2), (0.5, 1)])
def test_round_half_up_rounds_half_away_from_even(x, expected):
    assert round_half_up(x) == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        ("0개", 0),
        ("7건", 7),
        (" 1 개 ", 1),
        ("1개초과3개이하", 2),
        ("2건초과3건이하", 3),  # 중앙값 2.5 -> 3
        ("10개초과", 15),
    ],
)
def test_parse_count_range(value, expected):
    result = parse_count_range(value)
    assert result == expected
    assert isinstance(result, int)


@pytest.mark.parametrize("value", [None, np.nan, "기타", "1,000개", "개", ""])
def test_parse_count_range_returns_nan_for_unknown(value):
    assert np.isnan(parse_count_range(value))


def test_parse_count_range_options():
    assert parse_count_range("2건초과3건이하", return_int=False) == 2.5
    assert parse_count_range("10개초과", open_end_add=0) == 10


def test_convert_count_columns_leaves_input_untouched(raw_frame):
    before = raw_frame.copy()
    out = convert_count_columns(raw_frame)
    pd.testing.assert_frame_equal(raw_frame, before)
    assert out["신용카드개수"].dropna().isin([0, 1, 2, 15]).all()
    assert out["창구거래건수"].dropna().isin([0, 3, 105]).all()


def test_add_derived_columns_sums_balances_by_position(raw_frame):
    df = convert_count_columns(raw_frame)
    out = add_derived_columns(df, amount_start_col="신용카드사용금액")

    deposit = df[["요구불예금잔액", "거치식예금잔액", "적립식예금잔액"]].sum(axis=1)
    investment = df[["수익증권잔액", "신탁잔액", "퇴직연금잔액"]].sum(axis=1)
    loan = df[["여신_운전자금대출잔액", "여신_시설자금대출잔액"]].sum(axis=1)
    pd.testing.assert_series_equal(out["예금잔액"], deposit, check_names=False)
    pd.testing.assert_series_equal(out["투자잔액"], investment, check_names=False)
    pd.testing.assert_series_equal(out["여신대출잔액"], loan, check_names=False)

    total = deposit + investment + loan
    nonzero = total > 0
    np.testing.assert_allclose(out.loc[nonzero, "예금비중"], (deposit / total)[nonzero])
    assert (out.loc[~nonzero, ["예금비중", "대출비중"]] == 0).all().all()


def test_add_derived_columns_amount_range_start(raw_frame):
    df = convert_count_columns(raw_frame)
    with_card = add_derived_columns(df, amount_start_col="신용카드사용금액")["거래금액"]
    without_card = add_derived_columns(df, amount_start_col="창구거래금액")["거래금액"]
    pd.testing.assert_series_equal(with_card - without_card, df["신용카드사용금액"], check_names=False)


def test_add_derived_columns_counts_missing_transactions_as_zero():
    df = convert_count_columns(_one_row_frame())
    out = add_derived_columns(df, amount_start_col="신용카드사용금액")
    assert out.loc[0, "거래건수"] == 0


def test_add_derived_columns_requires_columns(raw_frame):
    with pytest.raises(KeyError, match="요구불출금금액"):
        add_derived_columns(raw_frame.drop(columns="요구불출금금액"), "신용카드사용금액")


def test_encode_grade_and_dedicated():
    df = pd.DataFrame({"법인_고객등급": ["일반", " 최우수", "미정"], "전담고객여부": ["y", "N", "N"]})
    out = encode_grade_and_dedicated(df)
    assert out["법인_고객등급"].tolist()[:2] == [1, 3]
    assert np.isnan(out.loc[2, "법인_고객등급"])
    assert out["전담고객여부"].tolist() == [1, 0, 0]


def test_encode_grade_and_dedicated_keeps_numeric_columns():
    df = pd.DataFrame({"법인_고객등급": [1, 2], "전담고객여부": [0, 1]})
    pd.testing.assert_frame_equal(encode_grade_and_dedicated(df), df)


def test_prepare_features_produces_numeric_features(raw_frame):
    out = prepare_features(raw_frame, amount_start_col="신용카드사용금액")
    for col in ["법인_고객등급", "전담고객여부", "여신_신용카드_좌수", "거래건수", "거래금액"]:
        assert pd.api.types.is_numeric_dtype(out[col]), col


def _one_row_frame() -> pd.DataFrame:
    df = make_raw_frame(n=1)
    for col in ["인터넷뱅킹거래건수", "스마트뱅킹거래건수", "폰뱅킹거래건수", "자동이체거래건수", "창구거래건수", "ATM거래건수"]:
        df[col] = None
    return df
