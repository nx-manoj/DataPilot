from ..utils.validation import ensure_polars
from ..config import get_config
import polars as pl
import pandas as pd
from typing import Union, Dict, Any, List, Optional
import math


def _js_divergence(p_counts: Dict, q_counts: Dict) -> float:
    """Computes Jensen-Shannon divergence between two discrete distributions."""
    all_keys = set(p_counts) | set(q_counts)
    p_total = sum(p_counts.values()) or 1
    q_total = sum(q_counts.values()) or 1

    p_dist = {k: p_counts.get(k, 0) / p_total for k in all_keys}
    q_dist = {k: q_counts.get(k, 0) / q_total for k in all_keys}
    m_dist = {k: (p_dist[k] + q_dist[k]) / 2 for k in all_keys}

    def _kl(a, m):
        result = 0.0
        for k in a:
            if a[k] > 0 and m[k] > 0:
                result += a[k] * math.log2(a[k] / m[k])
        return result

    return 0.5 * _kl(p_dist, m_dist) + 0.5 * _kl(q_dist, m_dist)


def compare(
    df_train: Union[pd.DataFrame, pl.DataFrame],
    df_test: Union[pd.DataFrame, pl.DataFrame],
    threshold: float = 0.1,
    use_ai: bool = False,
    ai_provider: Optional[str] = None,
    ai_model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Detects distribution drift between a training and test/production DataFrame.

    For numeric columns, compares mean, std, min, and max.
    For categorical columns, computes Jensen-Shannon divergence on value frequencies.
    Flags columns where the distributions appear significantly different.

    Args:
        df_train: Reference / training DataFrame.
        df_test:  Comparison / test or production DataFrame.
        threshold: JS divergence threshold above which a categorical column is flagged (0–1).

    Returns:
        List of dicts describing flagged columns with drift details.
        Also prints a formatted drift report to stdout.
    """
    train_df, _ = ensure_polars(df_train)
    test_df, _  = ensure_polars(df_test)

    shared_cols = [c for c in train_df.columns if c in test_df.columns]
    drift_flags: List[Dict[str, Any]] = []

    print("=" * 60)
    print("       DATAPILOT TRAIN vs TEST DRIFT REPORT       ")
    print("=" * 60)
    print(f"  Train shape: {train_df.height:,} rows × {train_df.width} cols")
    print(f"  Test  shape: {test_df.height:,}  rows × {test_df.width} cols")
    print(f"  Shared columns analysed: {len(shared_cols)}")
    print("-" * 60)

    for col in shared_cols:
        dtype = train_df[col].dtype

        if dtype.is_numeric():
            # ── Numeric drift: compare mean and std ───────────────────────────
            tr = train_df[col].drop_nulls()
            te = test_df[col].drop_nulls()

            if tr.len() == 0 or te.len() == 0:
                continue

            tr_mean, te_mean = tr.mean(), te.mean()
            tr_std,  te_std  = tr.std() or 1e-9, te.std() or 1e-9

            mean_shift = abs(tr_mean - te_mean) / (abs(tr_mean) + 1e-9)
            std_shift  = abs(tr_std - te_std)   / (abs(tr_std)  + 1e-9)

            if mean_shift > 0.15 or std_shift > 0.25:
                flag = {
                    "column": col,
                    "type": "numeric",
                    "severity": "🔴 HIGH" if mean_shift > 0.30 else "🟡 MEDIUM",
                    "train_mean": round(tr_mean, 4),
                    "test_mean":  round(te_mean, 4),
                    "mean_shift_pct": round(mean_shift * 100, 2),
                    "train_std": round(tr_std, 4),
                    "test_std":  round(te_std, 4),
                }
                drift_flags.append(flag)
                print(f"\n  ⚠️  {col}  [{flag['severity']}]  (numeric drift)")
                print(f"      Mean:  train={flag['train_mean']}  →  test={flag['test_mean']}  ({flag['mean_shift_pct']}% shift)")
                print(f"      Std:   train={flag['train_std']}  →  test={flag['test_std']}")

        else:
            # ── Categorical drift: JS divergence ──────────────────────────────
            tr_counts = dict(zip(
                train_df[col].value_counts()["value"].cast(pl.Utf8).to_list(),
                train_df[col].value_counts()["count"].to_list(),
            ))
            te_counts = dict(zip(
                test_df[col].value_counts()["value"].cast(pl.Utf8).to_list(),
                test_df[col].value_counts()["count"].to_list(),
            ))

            jsd = _js_divergence(tr_counts, te_counts)
            if jsd > threshold:
                flag = {
                    "column": col,
                    "type": "categorical",
                    "severity": "🔴 HIGH" if jsd > 0.3 else "🟡 MEDIUM",
                    "js_divergence": round(jsd, 4),
                    "train_top": max(tr_counts, key=tr_counts.get) if tr_counts else "N/A",
                    "test_top":  max(te_counts, key=te_counts.get) if te_counts else "N/A",
                }
                drift_flags.append(flag)
                print(f"\n  ⚠️  {col}  [{flag['severity']}]  (categorical drift)")
                print(f"      JS Divergence: {flag['js_divergence']} (threshold: {threshold})")
                print(f"      Train top value: '{flag['train_top']}'  |  Test top value: '{flag['test_top']}'")

    print("\n" + "-" * 60)
    if not drift_flags:
        print("  ✅ No significant distribution drift detected between datasets.")
    else:
        print(f"  🚨 {len(drift_flags)} column(s) flagged with potential distribution drift.")
        print("  💡 Drift can cause model degradation in production. Investigate flagged columns.")
    print("=" * 60)

    # ── Optional AI Commentary ────────────────────────────────────────────────
    if use_ai and drift_flags:
        cfg           = get_config()
        provider_name = ai_provider or cfg["ai_provider"]
        model_name    = ai_model    or cfg["ai_model"]
        key           = api_key     or cfg["api_key"]

        flags_text = "\n".join(
            f"- {f['column']} ({f['type']}): severity={f['severity']}, "
            + (
                f"mean shift={f.get('mean_shift_pct', 'N/A')}%"
                if f["type"] == "numeric" else
                f"JS divergence={f.get('js_divergence', 'N/A')}"
            )
            for f in drift_flags
        )
        system_prompt = (
            "You are DataPilot, a production ML expert. "
            "Given a list of features with distribution drift between train and test data, "
            "suggest 3-4 concrete mitigations (e.g. retraining window, feature monitoring, "
            "stratified sampling, drift-aware models). Be direct and technical."
        )
        user_prompt = (
            f"Train: {train_df.height:,} rows | Test: {test_df.height:,} rows\n"
            f"Drift flags (JS threshold={threshold}):\n{flags_text}"
        )
        try:
            from ..ai.factory import get_provider
            provider = get_provider(
                ai_provider=provider_name,
                ai_model=model_name,
                api_key=key,
            )
            ai_response = provider._call_with_raw_prompts(system_prompt, user_prompt)
            print(f"\n\ud83e\udd16 AI Drift Mitigations  [{provider_name.upper()}]:")
            print(ai_response)
        except Exception as e:
            print(f"\n\u26a0\ufe0f  AI error: {e}")

    return drift_flags
