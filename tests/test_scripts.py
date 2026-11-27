"""스크립트를 합성 CSV로 끝까지 실행해 본다."""

import pandas as pd

import score_common
from tests.synthetic import make_raw_frame


def _write_raw(tmp_path):
    path = tmp_path / "raw.csv"
    make_raw_frame(n=300).to_csv(path, index=False, encoding="cp949")
    return path


def test_score_common_writes_one_score_per_row(tmp_path, capsys):
    out = tmp_path / "out" / "score_all.csv"
    score_common.main(["--data", str(_write_raw(tmp_path)), "--output", str(out)])

    scores = pd.read_csv(out, encoding="utf-8-sig")
    assert list(scores.columns) == ["score"]
    assert len(scores) == 300
    assert scores["score"].between(10, 110).all()
    assert "사용 피처" in capsys.readouterr().out
