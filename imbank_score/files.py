"""입력·출력 파일 위치와 읽기/쓰기."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

RAW_DATA_PATH = DATA_DIR / "(아이엠뱅크) 2025 교육용 데이터.csv"
BETA_PATH = DATA_DIR / "Beta_Industry.csv"

# 교육용 데이터와 베타 CSV는 cp949로 인코딩되어 있다.
CSV_ENCODING = "cp949"


def read_csv(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} 파일이 없습니다. README의 '데이터 준비'를 참고하세요.")
    return pd.read_csv(path, encoding=CSV_ENCODING)


def write_scores(scores: pd.Series, path: Path) -> None:
    """점수 한 열만 저장한다. 행 순서는 원본 데이터와 같다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    scores.rename("score").to_frame().to_csv(path, index=False, encoding="utf-8-sig")
