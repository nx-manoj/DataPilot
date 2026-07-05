from typing import Dict, Any

def diagnose(train_score: float, test_score: float, metric_name: str = "Accuracy") -> None:
    """Diagnoses model health, highlighting overfitting, underfitting, or optimal states."""
    gap = train_score - test_score
    
    print("=" * 50)
    print("        DATAPILOT ML MODEL DIAGNOSTICS         ")
    print("=" * 50)
    print(f"📈 Training {metric_name}: {round(train_score, 4)}")
    print(f"📉 Testing {metric_name}:  {round(test_score, 4)}")
    print(f"↔️ Performance Gap:   {round(gap, 4)}")
    print("-" * 50)
    
    print("\n🔍 Diagnostic Verdict:")
    if train_score < 0.60 and test_score < 0.60:
        print("   ⚠️ UNDERFITTING DETECTED: The model performs poorly on both train and test sets.")
        print("   💡 Fix: Try increasing model complexity, engineering more features, or training longer.")
    elif gap > 0.15:
        print("   🚨 OVERFITTING DETECTED: High variance. High training score, low test score.")
        print("   💡 Fix: Add regularization, gather more training samples, prune trees, or apply dropout.")
    elif gap < -0.05:
        print("   ❓ ANOMALOUS DRIFT: Test performance noticeably beats training performance.")
        print("   💡 Fix: Check for data leakage, unrepresentative validation sets, or target leak variables.")
    else:
        print("   ✅ OPTIMAL FIT: Model generalizes well with balanced variance and low bias thresholds.")
        
    print("=" * 50)
