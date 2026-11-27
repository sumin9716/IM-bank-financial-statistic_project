"""테스트용 합성 데이터. 잔액 열을 위치로 합산하므로 실제 교육용 데이터의 열 순서를 흉내 낸다."""

import numpy as np
import pandas as pd

from imbank_score.params import INDUSTRY_PARAMS
from imbank_score.preprocess import ACCOUNT_COUNT_COLUMNS, TRANSACTION_COUNT_COLUMNS

# 위치 7~9: 예금, 10~12: 투자, 13: 합산 제외, 14~15: 여신
BALANCE_COLUMNS = [
    "요구불예금잔액", "거치식예금잔액", "적립식예금잔액",
    "수익증권잔액", "신탁잔액", "퇴직연금잔액",
    "기타잔액",
    "여신_운전자금대출잔액", "여신_시설자금대출잔액",
    "외환_수출실적금액", "외환_수입실적금액",
]
AMOUNT_COLUMNS = ["신용카드사용금액", "창구거래금액", "인터넷뱅킹거래금액", "요구불출금금액"]


def make_raw_frame(n: int = 200, seed: int = 0, industries=None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    industries = list(industries or [*INDUSTRY_PARAMS, "제조업"])
    data = {
        "기준년월": rng.choice([202201, 202204, 202312], n),
        "법인ID": np.arange(n),
        "업종_대분류": rng.choice(industries, n),
        "사업장_시도": "대구",
        "법인_고객등급": rng.choice(["일반", "우수", "최우수"], n),
        "전담고객여부": rng.choice(["Y", "N"], n),
        "설립연도": 2000,
    }
    for col in BALANCE_COLUMNS:
        data[col] = np.round(rng.lognormal(8, 2, n) * (rng.random(n) > 0.4))
    for col in ACCOUNT_COUNT_COLUMNS:
        data[col] = rng.choice(["0개", "1개", "1개초과3개이하", "10개초과", None], n)
    for col in TRANSACTION_COUNT_COLUMNS:
        data[col] = rng.choice(["0건", "1건초과5건이하", "100건초과", None], n)
    for col in AMOUNT_COLUMNS:
        data[col] = np.round(rng.lognormal(8, 2, n) * (rng.random(n) > 0.3))
    return pd.DataFrame(data)

