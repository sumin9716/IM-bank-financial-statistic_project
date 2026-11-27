import numpy as np
import pandas as pd
import pytest

from imbank_score.params import INDUSTRY_FEATURES, INDUSTRY_PARAMS
from imbank_score.preprocess import prepare_features
from imbank_score.scoring import scale_to_unit, score_by_industry, zero_penalty_score
from tests.synthetic import make_raw_frame

FEATURES = ["a", "b"]
WEIGHTS = {"a": 1.0, "b": 1.0}


@pytest.fixture
def features_frame() -> pd.DataFrame:
    return pd.DataFrame({"a": [0.0, 1.0, 2.0, 4.0], "b": [0.0, 0.0, 0.0, 10.0]})


def test_scale_to_unit_minmax(features_frame):
    out = scale_to_unit(features_frame)
    assert out["a"].tolist() == [0.0, 0.25, 0.5, 1.0]
    assert out["b"].tolist() == [0.0, 0.0, 0.0, 1.0]


def test_scale_to_unit_constant_column_uses_unit_range():
    out = scale_to_unit(pd.DataFrame({"c": [3.0, 3.0]}))
    assert out["c"].tolist() == [1.0, 1.0]  # (3 - 0) / (1 - 0)을 1로 자름


def test_scale_to_unit_winsor_clips_tails():
    x = pd.DataFrame({"c": list(range(100)) + [10_000]})
    out = scale_to_unit(x, method="winsor_minmax", q_low=0.0, q_high=0.9)
    assert out["c"].iloc[-1] == 1.0
    assert out["c"].min() == 0.0


def test_scale_to_unit_rejects_unknown_method(features_frame):
    with pytest.raises(ValueError):
        scale_to_unit(features_frame, method="zscore")


def test_zero_penalty_score_is_rescaled_to_0_100(features_frame):
    s = zero_penalty_score(features_frame, FEATURES, WEIGHTS, lam=0.5)
    assert s.min() == 0 and s.max() == 100
    assert s.name == "score"


def test_zero_penalty_score_penalizes_sparse_features():
    # 두 행 모두 a·b 중 하나만 최댓값이다. b는 0인 비율이 더 높으므로 b가 큰 행이 낮은 점수를 받는다.
    df = pd.DataFrame({"a": [1.0, 0.0, 1.0, 0.0, 1.0], "b": [0.0, 1.0, 0.0, 0.0, 0.0]})
    s = zero_penalty_score(df, FEATURES, WEIGHTS, lam=1.0, rescale_0_100=False)
    assert s.iloc[0] > s.iloc[1]
    no_penalty = zero_penalty_score(df, FEATURES, WEIGHTS, lam=0.0, rescale_0_100=False)
    assert no_penalty.iloc[0] == no_penalty.iloc[1]


def test_zero_penalty_score_weights_are_l1_normalized(features_frame):
    doubled = {k: 2 * v for k, v in WEIGHTS.items()}
    pd.testing.assert_series_equal(
        zero_penalty_score(features_frame, FEATURES, WEIGHTS, lam=0.3, rescale_0_100=False),
        zero_penalty_score(features_frame, FEATURES, doubled, lam=0.3, rescale_0_100=False),
    )


def test_zero_penalty_score_applies_scalar_and_rowwise_beta(features_frame):
    base = zero_penalty_score(features_frame, FEATURES, WEIGHTS, lam=0.5)
    half = zero_penalty_score(features_frame, FEATURES, WEIGHTS, lam=0.5, beta=0.5)
    np.testing.assert_allclose(half, (base * 0.5).round(2))

    rowwise = pd.Series([1.0, 1.0, 2.0, 2.0], index=features_frame.index)
    capped = zero_penalty_score(features_frame, FEATURES, WEIGHTS, lam=0.5, beta=rowwise)
    assert capped.iloc[-1] == 100  # 200을 100으로 자름
    np.testing.assert_allclose(capped.iloc[:2], base.iloc[:2])


def test_zero_penalty_score_treats_non_numeric_as_zero():
    df = pd.DataFrame({"a": ["x", "1", "2"], "b": [1.0, 2.0, 3.0]})
    s = zero_penalty_score(df, FEATURES, WEIGHTS, lam=0.0)
    assert s.iloc[0] == 0


def test_zero_penalty_score_requires_every_weight(features_frame):
    with pytest.raises(KeyError):
        zero_penalty_score(features_frame, FEATURES, {"a": 1.0}, lam=0.5)


def test_score_by_industry_scores_configured_industries_only():
    df = prepare_features(make_raw_frame(n=300), amount_start_col="신용카드사용금액")
    scores, logs = score_by_industry(df, INDUSTRY_FEATURES, INDUSTRY_PARAMS)

    other = df["업종_대분류"] == "제조업"
    assert scores[other].isna().all()
    assert scores[~other].between(0, 100).all()
    assert logs[-1] == f"[INFO] 설정되지 않은 업종 행 {other.sum()}건은 score=NaN"


def test_score_by_industry_scores_each_industry_independently():
    df = prepare_features(make_raw_frame(n=300), amount_start_col="신용카드사용금액")
    scores, _ = score_by_industry(df, INDUSTRY_FEATURES, INDUSTRY_PARAMS)

    construction = df[df["업종_대분류"] == "건설업"]
    alone, _ = score_by_industry(construction, INDUSTRY_FEATURES, INDUSTRY_PARAMS)
    pd.testing.assert_series_equal(alone, scores[construction.index])


def test_score_by_industry_uses_beta_column():
    df = prepare_features(make_raw_frame(n=300), amount_start_col="신용카드사용금액")
    plain, _ = score_by_industry(df, INDUSTRY_FEATURES, INDUSTRY_PARAMS)
    with_beta, _ = score_by_industry(df.assign(beta=0.5), INDUSTRY_FEATURES, INDUSTRY_PARAMS, beta_col="beta")
    np.testing.assert_allclose(with_beta.dropna(), (plain.dropna() * 0.5).round(2), atol=0.01)


def test_score_by_industry_logs_missing_industry():
    df = prepare_features(make_raw_frame(n=50, industries=["건설업"]), amount_start_col="신용카드사용금액")
    _, logs = score_by_industry(df, INDUSTRY_FEATURES, {"정보통신업": INDUSTRY_PARAMS["정보통신업"]})
    assert logs[0] == "[SKIP] 업종 '정보통신업': 행 없음"
