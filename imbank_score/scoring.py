"""0값 패널티 선형 스코어.

    S = clip( β · rescale_0_100( Σ ω_i·α̂_i − λ·Σ ρ_i·α̂_i ), 0, 100 )

α̂_i는 피처 i를 [0, 1]로 선형 변환한 값, ω_i는 가중치(기본은 절댓값 합 1로 정규화),
ρ_i는 스코어를 매기는 집단에서 피처 i가 0인 비율이다. 피처 i의 실제 계수는 ω_i − λ·ρ_i가
되므로, 0이 많은 피처일수록 값이 커도 점수가 덜 오른다.
"""

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from imbank_score.preprocess import require_columns


def scale_to_unit(
    X: pd.DataFrame, method: str = "minmax", q_low: float = 0.01, q_high: float = 0.99
) -> pd.DataFrame:
    """열마다 [lo, hi]를 [0, 1]로 옮기고 범위 밖 값은 자른다.

    method='minmax'는 최솟값·최댓값을, 'winsor_minmax'는 q_low·q_high 분위수를 lo·hi로 쓴다.
    lo·hi가 유한하지 않거나 같으면 (0, 1)을 쓴다.
    """
    if method not in ("minmax", "winsor_minmax"):
        raise ValueError("method must be 'minmax' or 'winsor_minmax'")

    scaled = {}
    for col in X.columns:
        x = pd.to_numeric(X[col], errors="coerce").astype(float)
        if method == "minmax":
            lo, hi = np.nanmin(x), np.nanmax(x)
        else:
            lo, hi = x.quantile(q_low), x.quantile(q_high)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
            lo, hi = 0.0, 1.0
        lo, hi = float(lo), float(hi)
        a = (x - lo) / (hi - lo) if hi > lo else pd.Series(0.0, index=X.index)
        scaled[col] = a.clip(0, 1)
    return pd.DataFrame(scaled, index=X.index)


def zero_penalty_score(
    df: pd.DataFrame,
    features: Sequence[str],
    weights: Mapping[str, float],
    lam: float,
    beta: float | pd.Series = 1.0,
    *,
    scaler_method: str = "minmax",
    q_low: float = 0.01,
    q_high: float = 0.99,
    l1_normalize: bool = True,
    rescale_0_100: bool = True,
) -> pd.Series:
    """df의 행마다 점수(소수 둘째 자리)를 계산한다. 스케일 범위와 ρ는 df 안에서 정한다.

    beta는 스칼라이거나 df와 인덱스가 같은 행별 Series다. 숫자가 아닌 피처 값은 0으로 본다.
    """
    require_columns(df, features)
    X = df[list(features)].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    rho = (X == 0).sum(axis=0) / len(X)
    A = scale_to_unit(X, method=scaler_method, q_low=q_low, q_high=q_high)

    w = pd.Series({c: float(weights[c]) for c in features}, dtype=float)
    if l1_normalize and w.abs().sum() > 0:
        w = w / w.abs().sum()

    raw = A.mul(w, axis=1).sum(axis=1) - A.mul(rho, axis=1).sum(axis=1) * lam
    if rescale_0_100 and raw.max() > raw.min():
        raw = 100 * (raw - raw.min()) / (raw.max() - raw.min())
    return (raw * beta).clip(0, 100).round(2).rename("score")


def score_by_industry(
    df: pd.DataFrame,
    features: Sequence[str],
    params: Mapping[str, Mapping],
    *,
    beta_col: str | None = None,
    industry_col: str = "업종_대분류",
) -> tuple[pd.Series, list[str]]:
    """params에 있는 업종마다 그 업종 행만으로 따로 스코어링한다.

    params는 업종명 -> {'weights': {피처: ω}, 'lam': λ}. beta_col을 주면 그 열을 행별 β로 쓰고,
    없으면 β=1이다. params에 없는 업종의 점수는 NaN이다. 반환값은 (점수, 실행 로그).
    """
    require_columns(df, [industry_col, *features, *([beta_col] if beta_col else [])])
    industry = df[industry_col].astype(str)
    scores = pd.Series(np.nan, index=df.index, dtype=float, name="score")
    logs = []

    for name, p in params.items():
        sub = df.loc[industry == name]
        if sub.empty:
            logs.append(f"[SKIP] 업종 '{name}': 행 없음")
            continue
        beta = sub[beta_col] if beta_col else 1.0
        scores.loc[sub.index] = zero_penalty_score(sub, features, p["weights"], p["lam"], beta)
        logs.append(f"[OK] 업종 '{name}': n={len(sub)}, lam={p['lam']}")

    others = int((~industry.isin(list(params))).sum())
    if others:
        logs.append(f"[INFO] 설정되지 않은 업종 행 {others}건은 score=NaN")
    return scores, logs
