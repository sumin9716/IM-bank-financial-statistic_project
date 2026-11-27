import pandas as pd
import pytest

from imbank_score.beta import (
    DEFAULT_BETA,
    attach_beta,
    load_industry_beta,
    normalize_industry,
    normalize_month,
)


@pytest.fixture
def beta_wide() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "기준년월": ["2022-01", "2022-04", "2022-05"],
            "건설업": [9.0, 1.1, 1.2],
            "정보통신업": [9.0, 0.9, None],
        }
    )


@pytest.mark.parametrize("value", ["2022-04", "2022.04", 202204, " 202204 "])
def test_normalize_month(value):
    assert normalize_month(value) == "202204"


def test_normalize_industry():
    assert normalize_industry(" 교육  서비스업 ") == "교육 서비스업"


def test_load_industry_beta_backfills_q1_with_april(beta_wide):
    beta = load_industry_beta(beta_wide).set_index(["업종_대분류", "기준년월"])["beta"]

    for month in ["202201", "202202", "202203"]:
        assert beta[("건설업", month)] == 1.1
        assert beta[("정보통신업", month)] == 0.9
    assert beta[("건설업", "202205")] == 1.2
    assert ("정보통신업", "202205") not in beta.index  # 결측은 버린다


def test_attach_beta_merges_by_month_and_industry(beta_wide):
    df = pd.DataFrame({"기준년월": [202204, 202205, 202205], "업종_대분류": ["건설업", "건설업  ", "정보통신업"]})
    out, unmatched = attach_beta(df, load_industry_beta(beta_wide))

    assert out["beta"].tolist() == [1.1, 1.2, DEFAULT_BETA]
    assert unmatched.tolist() == [False, False, True]


def test_attach_beta_rejects_duplicate_keys():
    beta_long = pd.DataFrame({"기준년월": ["202204", "202204"], "업종_대분류": ["건설업", "건설업"], "beta": [1.0, 2.0]})
    df = pd.DataFrame({"기준년월": [202204], "업종_대분류": ["건설업"]})
    with pytest.raises(pd.errors.MergeError):
        attach_beta(df, beta_long)
