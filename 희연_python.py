import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from decimal import Decimal, ROUND_HALF_UP

# 한글 폰트
plt.rcParams['font.family'] = 'Malgun Gothic'

# ================================================================
# 0) 데이터 로드
# ================================================================
# iM 뱅크 기업금융 데이터
df = pd.read_csv(
    r'C:\imbankproj\TEAMPROJ_1\(아이엠뱅크) 2025 교육용 데이터.csv',
    encoding='cp949'
)

# (옵션) 신용등급 외부데이터: 현재 로직 미사용, 필요 시 후처리에서 join
# rating = pd.read_csv(
#     r"G:\내 드라이브\Colab Notebooks\iM\Team_project\stats_proj\기업신용등급 분포.csv",
#     encoding='cp949'
# )

# ================================================================
# 1) 거래채널별 거래건수/좌수 전처리
# ================================================================
def round_half_up(x: float) -> int:
    """항상 .5는 올림 (3.5->4, 2.5->3)"""
    return int(Decimal(str(x)).quantize(Decimal('1'), rounding=ROUND_HALF_UP))

def conv_count_strict(value, open_end_add: int = 5, return_int: bool = True):
    """
    허용 패턴(공백/전각 제거 후):
      1) '0개', '1건'
      2) 'N개초과M개이하' / 'N건초과M건이하'
      3) 'N개초과' / 'N건초과'
    그 외 → NaN
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan

    s = str(value).strip()
    s = re.sub(r'\s+', '', s).replace('，', ',')  # 공백/전각 제거

    m = re.fullmatch(r'^(\d+)(개|건)$', s)
    if m:
        out = float(m.group(1))
    else:
        m = re.fullmatch(r'^(\d+)(개|건)초과(\d+)(개|건)이하$', s)
        if m:
            a, b = int(m.group(1)), int(m.group(3))
            mid = (a + b) / 2.0
            out = round_half_up(mid) if return_int else mid
        else:
            m = re.fullmatch(r'^(\d+)(개|건)초과$', s)
            if m:
                a = int(m.group(1))
                out = a + int(open_end_add)
            else:
                return np.nan
    return int(out) if return_int else float(out)

columns1 = [
    '요구불예금좌수','거치식예금좌수','적립식예금좌수','수익증권좌수','신탁좌수','퇴직연금좌수',
    '여신_운전자금대출좌수','여신_시설자금대출좌수','신용카드개수'
]
columns2 = [
    '외환_수출실적거래건수','외환_수입실적거래건수','창구거래건수','인터넷뱅킹거래건수',
    '스마트뱅킹거래건수','폰뱅킹거래건수','ATM거래건수','자동이체거래건수'
]

for col in (columns1 + columns2):
    if col in df.columns:
        df[col] = df[col].apply(lambda v: conv_count_strict(v, open_end_add=5, return_int=True))

# ================================================================
# 2) 파생 칼럼
#    ⚠️ 아래 슬라이스는 컬럼 "순서"에 의존. 필요 시 컬럼명 직접 합산으로 교체 권장.
# ================================================================
df['예금잔액'] = df.iloc[:, 7:10].sum(axis=1)
df['투자잔액'] = df.iloc[:, 10:13].sum(axis=1)
df['예금_투자_잔액'] = df['예금잔액'] + df['투자잔액']
df['여신대출잔액'] = df.iloc[:, 14:16].sum(axis=1)

den = (df['예금잔액'] + df['투자잔액'] + df['여신대출잔액']).replace(0, np.nan)
df['예금비중'] = (df['예금잔액'] / den).fillna(0.0)
df['대출비중'] = (df['여신대출잔액'] / den).fillna(0.0)

df['외환실적'] = df.iloc[:, 16:18].sum(axis=1)

# 여신/카드 좌수 합산 (열 순서 의존: '여신_운전자금대출좌수' ~ '신용카드개수')
if '여신_운전자금대출좌수' in df.columns and '신용카드개수' in df.columns:
    df['여신_신용카드_좌수'] = df.loc[:, '여신_운전자금대출좌수':'신용카드개수'].sum(axis=1)
else:
    df['여신_신용카드_좌수'] = 0

# 온/오프라인 거래건수
df['온라인_거래건수'] = df[['인터넷뱅킹거래건수','스마트뱅킹거래건수','폰뱅킹거래건수','자동이체거래건수']].sum(axis=1, min_count=1).fillna(0)
df['오프라인_거래건수'] = df[['창구거래건수','ATM거래건수']].sum(axis=1, min_count=1).fillna(0)
df['거래건수'] = df['온라인_거래건수'] + df['오프라인_거래건수']

# 거래금액 (열 순서 의존: '신용카드사용금액' ~ '요구불출금금액')
if '신용카드사용금액' in df.columns and '요구불출금금액' in df.columns:
    df['거래금액'] = df.loc[:, '신용카드사용금액':'요구불출금금액'].sum(axis=1)
else:
    df['거래금액'] = 0.0

# ================================================================
# 3) 등급/전담 숫자화
# ================================================================
def ensure_numeric_grade_and_ded(df_in: pd.DataFrame) -> pd.DataFrame:
    out = df_in.copy()
    if '법인_고객등급' in out.columns and out['법인_고객등급'].dtype == 'object':
        out['법인_고객등급'] = (
            out['법인_고객등급'].astype(str).str.strip()
              .map({'일반': 1, '우수': 2, '최우수': 3})
        )
    if '전담고객여부' in out.columns and out['전담고객여부'].dtype == 'object':
        out['전담고객여부'] = (
            out['전담고객여부'].astype(str).str.strip().str.upper()
              .map({'Y': 1, 'N': 0})
        )
    return out

# ================================================================
# 4) 분포 보존 스케일러
# ================================================================
def fit_linear_scaler(X: pd.DataFrame, method='minmax', q_low=0.01, q_high=0.99):
    params = {}
    for c in X.columns:
        x = pd.to_numeric(X[c], errors='coerce')
        if method == 'minmax':
            lo, hi = np.nanmin(x), np.nanmax(x)
        elif method == 'winsor_minmax':
            lo, hi = x.quantile(q_low), x.quantile(q_high)
        else:
            raise ValueError("method must be 'minmax' or 'winsor_minmax'")
        if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
            lo, hi = 0.0, 1.0
        params[c] = (float(lo), float(hi))
    params['_method'] = method
    params['_q'] = (q_low, q_high)
    return params

def transform_linear_scaler(X: pd.DataFrame, params: dict):
    A = pd.DataFrame(index=X.index, columns=X.columns, dtype=float)
    for c in X.columns:
        lo, hi = params[c]
        x = pd.to_numeric(X[c], errors='coerce').astype(float)
        a = (x - lo) / (hi - lo) if hi > lo else pd.Series(0.0, index=X.index)
        A[c] = a.clip(0, 1)
    return A

# ================================================================
# 5) 스코어 계산 (선형 패널티)
#    S = β * (Σ ω_i·α̂ - λ·Σ ρ·α̂)
#    (여기서는 [0,100] 리스케일 후 β 곱)
# ================================================================
def score_linear_preserve_shape(
    df_est: pd.DataFrame,
    feature_cols: list,
    weights: dict,
    lam: float,
    beta: float = 1.0,
    scaler_method: str = 'minmax',
    q_low: float = 0.01, q_high: float = 0.99,
    l1_normalize: bool = True,
    rescale_0_100: bool = True
):
    df_num = ensure_numeric_grade_and_ded(df_est)
    X_raw = df_num[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)

    # 각 피처의 0 비율(ρ_i)
    rho = (X_raw == 0).sum(axis=0) / len(X_raw)

    # 스케일링 → α̂ ∈ [0,1]
    scaler_params = fit_linear_scaler(X_raw, method=scaler_method, q_low=q_low, q_high=q_high)
    A = transform_linear_scaler(X_raw, scaler_params)

    # 가중치 (선택적 L1 정규화)
    w = pd.Series({c: float(weights.get(c, 0.0)) for c in feature_cols}, dtype=float)
    if l1_normalize:
        denom = w.abs().sum()
        if denom > 0:
            w = w / denom

    # 합산항/패널티
    pos = A.mul(w, axis=1).sum(axis=1)
    pen = (A.mul(rho, axis=1)).sum(axis=1) * lam
    S_raw = (pos - pen)

    # [0,100] 리스케일
    if rescale_0_100 and (S_raw.max() > S_raw.min()):
        S_scaled = 100 * (S_raw - S_raw.min()) / (S_raw.max() - S_raw.min())
    else:
        S_scaled = S_raw

    # β 적용 후 0~100 클립
    S_final = np.clip(beta * S_scaled, 0, 100)

    out = df_est.copy()
    out['score'] = S_final.round(2)

    report = {
        "weights(L1={})".format('1' if l1_normalize else 'raw'): w.to_dict(),
        "lambda": lam, "beta": beta,
        "rho_zero_ratio": rho.to_dict(),
        "scaler": scaler_method, "quantiles": (q_low, q_high)
    }
    return out, report

# ================================================================
# 6) 업종별 하이퍼파라미터 (정의된 업종만 스코어)
# ================================================================
COMMON_FEATURES = [
    '법인_고객등급','전담고객여부','예금잔액','투자잔액','여신대출잔액',
    '예금비중','대출비중','여신_신용카드_좌수','거래건수','거래금액'
]

INDUSTRY_CONFIG = {
    '교육 서비스업': {
        'feature_cols': COMMON_FEATURES,
        'omega': {
            '법인_고객등급': 0.6, '전담고객여부': 0.6,
            '예금잔액': 1.0, '투자잔액': 1.0, '여신대출잔액': 1.2,
            '예금비중': 0.8, '대출비중': -1.2,
            '여신_신용카드_좌수': 0.8, '거래금액': 0.8, '거래건수': 0.8
        },
        'lam': 0.7, 'beta': 1.0
    },
    '농업, 임업 및 어업': {
        'feature_cols': COMMON_FEATURES,
        'omega': {
            '법인_고객등급': 0.6, '전담고객여부': 0.6,
            '예금잔액': 1.0, '투자잔액': 1.0, '여신대출잔액': 1.2,
            '예금비중': 0.8, '대출비중': -1.2,
            '여신_신용카드_좌수': 0.8, '거래금액': 0.8, '거래건수': 0.8
        },
        'lam': 0.45, 'beta': 1.0
    },
    '수도, 하수 및 폐기물 처리, 원료 재생업': {
        'feature_cols': COMMON_FEATURES,
        'omega': {
            '법인_고객등급': 0.6, '전담고객여부': 0.6,
            '예금잔액': 1.0, '투자잔액': 0.6, '여신대출잔액': 1.2,
            '예금비중': 0.9, '대출비중': -1.2,
            '여신_신용카드_좌수': 1.0, '거래금액': 1.0, '거래건수': 1.0
        },
        'lam': 0.6, 'beta': 1.0
    },
    '전기, 가스, 증기 및 공기조절 공급업': {
        'feature_cols': COMMON_FEATURES,
        'omega': {
            '법인_고객등급': 0.6, '전담고객여부': 1.2,
            '예금잔액': 1.2, '투자잔액': 0.5, '여신대출잔액': 1.2,
            '예금비중': 0.8, '대출비중': -0.7,
            '여신_신용카드_좌수': 1.2, '거래금액': 1.2, '거래건수': 1.2
        },
        'lam': 0.75, 'beta': 1.0
    },
    '전문, 과학 및 기술 서비스업': {
        'feature_cols': COMMON_FEATURES,
        'omega': {
            '법인_고객등급': 0.6, '전담고객여부': 0.6,
            '예금잔액': 0.8, '투자잔액': 0.6, '여신대출잔액': 1.2,
            '예금비중': 0.8, '대출비중': -1.2,
            '여신_신용카드_좌수': 0.8, '거래금액': 0.8, '거래건수': 0.8
        },
        'lam': 0.5, 'beta': 1.0
    },
    '정보통신업': {
        'feature_cols': COMMON_FEATURES,
        'omega': {
            '법인_고객등급': 0.6, '전담고객여부': 1.2,
            '예금잔액': 1.0, '투자잔액': 0.5, '여신대출잔액': 1.2,
            '예금비중': 1.2, '대출비중': -1.2,
            '여신_신용카드_좌수': 1.2, '거래금액': 0.8, '거래건수': 1.2
        },
        'lam': 0.75, 'beta': 1.0
    },

    '건설업': {
        'feature_cols': COMMON_FEATURES,
        'omega': {
            '법인_고객등급': 0.6,
            '전담고객여부': 0.6,
            '예금잔액': 1.2,
            '투자잔액': 0.5,
            '여신대출잔액': 1.2,
            '예금비중': 1.2,
            '대출비중': -0.8,
            '여신_신용카드_좌수': 0.8,
            '거래금액': 0.8,
            '거래건수': 0.8
        },
        'lam': 0.6, 'beta': 1.0
    }
}

# ================================================================
# 7) 업종별 스코어링 → 병합 (저장은 하지 않음)
#    - INDUSTRY_CONFIG에 정의된 업종 "만" 처리
#    - 그 외 업종은 score=NaN
# ================================================================
def score_by_industry_and_concat(
    df_all: pd.DataFrame,
    industry_col: str = '업종_대분류',
    config_map: dict = INDUSTRY_CONFIG,
    scaler_method: str = 'minmax',
    l1_normalize: bool = True,
    rescale_0_100: bool = True
):
    if industry_col not in df_all.columns:
        raise KeyError(f"'{industry_col}' 컬럼이 없습니다.")

    final_score = pd.Series(index=df_all.index, dtype=float)
    industries = list(config_map.keys())  # 오직 설정된 업종만

    logs = []
    for ind in industries:
        cfg = config_map[ind]
        feature_cols = cfg['feature_cols']
        weights     = cfg['omega']
        lam         = cfg['lam']
        beta        = cfg['beta']

        missing = [c for c in feature_cols if c not in df_all.columns]
        if missing:
            logs.append(f"[WARN] 업종 '{ind}' : 누락 컬럼 {missing} → 스킵")
            continue

        mask = (df_all[industry_col].astype(str) == ind)
        df_sub = df_all.loc[mask].copy()
        if df_sub.empty:
            logs.append(f"[SKIP] 업종 '{ind}' : 행 없음")
            continue

        scored, rpt = score_linear_preserve_shape(
            df_est=df_sub,
            feature_cols=feature_cols,
            weights=weights,
            lam=lam,
            beta=beta,
            scaler_method=scaler_method,
            l1_normalize=l1_normalize,
            rescale_0_100=rescale_0_100
        )

        final_score.loc[scored.index] = scored['score'].astype(float)
        logs.append(f"[OK] 업종 '{ind}' : n={len(df_sub)}, lam={lam}, beta={beta}")

    # 설정 외 업종은 NaN 유지
    df_out = df_all.copy()
    df_out['score'] = final_score

    other_cnt = df_out[df_out[industry_col].astype(str).isin(industries) == False].shape[0]
    if other_cnt > 0:
        logs.append(f"[INFO] 설정되지 않은 업종 행 {other_cnt}건은 score=NaN")

    return df_out, logs

# ================================================================
# 8) 실행 
# ================================================================
df_scored, run_logs = score_by_industry_and_concat(
    df_all=df,
    industry_col='업종_대분류',
    config_map=INDUSTRY_CONFIG,
    scaler_method='minmax',
    l1_normalize=True,
    rescale_0_100=True
)

#  score만 저장 
df_scored[['score']].to_csv("score_test.csv", index=False, encoding="utf-8-sig")
