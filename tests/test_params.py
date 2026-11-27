from imbank_score.params import COMMON_FEATURES, COMMON_WEIGHTS, INDUSTRY_FEATURES, INDUSTRY_PARAMS


def test_every_industry_has_a_weight_for_each_feature():
    for industry, params in INDUSTRY_PARAMS.items():
        assert set(params["weights"]) == set(INDUSTRY_FEATURES), industry
        assert 0 < params["lam"] < 1, industry


def test_common_model_weights_match_features():
    assert set(COMMON_WEIGHTS) == set(COMMON_FEATURES)
