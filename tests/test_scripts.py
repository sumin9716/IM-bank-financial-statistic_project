"""스크립트를 합성 CSV로 끝까지 실행해 본다."""

import pandas as pd

import score_common
import score_industry
from imbank_score.params import INDUSTRY_PARAMS
from tests.synthetic import make_raw_frame


def _write_raw(tmp_path):
    path = tmp_path / "raw.csv"
    make_raw_frame(n=300).to_csv(path, index=False, encoding="cp949")
    return path


def _write_beta(tmp_path):
    path = tmp_path / "beta.csv"
    beta = pd.DataFrame({"기준년월": ["2022-04", "2023-12"]})
    for industry in INDUSTRY_PARAMS:
        beta[industry] = [0.5, 1.0]
    beta.to_csv(path, index=False, encoding="cp949")
    return path


def test_score_common_writes_one_score_per_row(tmp_path, capsys):
    out = tmp_path / "out" / "score_all.csv"
    score_common.main(["--data", str(_write_raw(tmp_path)), "--output", str(out)])

    scores = pd.read_csv(out, encoding="utf-8-sig")
    assert list(scores.columns) == ["score"]
    assert len(scores) == 300
    assert scores["score"].between(10, 110).all()
    assert "사용 피처" in capsys.readouterr().out


def test_score_industry_applies_beta(tmp_path, capsys):
    raw = _write_raw(tmp_path)
    with_beta = tmp_path / "score_beta.csv"
    without_beta = tmp_path / "score_nobeta.csv"
    score_industry.main(["--data", str(raw), "--beta", str(_write_beta(tmp_path)), "--output", str(with_beta)])
    score_industry.main(["--data", str(raw), "--no-beta", "--output", str(without_beta)])

    plain = pd.read_csv(without_beta)["score"]
    scaled = pd.read_csv(with_beta)["score"]
    assert len(plain) == len(scaled) == 300
    assert plain.isna().equals(scaled.isna())  # 설정되지 않은 업종(제조업)만 비어 있다
    scored = plain.notna()
    assert (scaled[scored] <= plain[scored]).all()  # 베타가 0.5 또는 1이므로 점수가 커지지 않는다
    assert (scaled[scored] < plain[scored]).any()
    assert "[β 매칭 실패율]" in capsys.readouterr().out
