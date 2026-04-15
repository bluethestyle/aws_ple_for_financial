# Adaptive Distillation Analysis (2026-04-14)

## Teacher: 10ep joint_full (PLE softmax, 7 experts, adaTT OFF)
## Student: LGBM x 13 (per-task independent)

## Teacher Threshold Gating (2x Random Baseline)

| Task | Type | Threshold | Teacher | Viable | Route |
|------|------|-----------|---------|--------|-------|
| churn_signal | binary | AUC > 0.60 | 0.6870 | YES | DISTILL |
| will_acquire_accounts | binary | AUC > 0.60 | 0.7211 | YES | DISTILL |
| will_acquire_payments | binary | AUC > 0.60 | 0.6932 | YES | DISTILL |
| will_acquire_deposits | binary | AUC > 0.60 | 0.6705 | YES | DISTILL |
| will_acquire_investments | binary | AUC > 0.60 | 0.6753 | YES | DISTILL |
| will_acquire_lending | binary | AUC > 0.60 | 0.6655 | YES | DISTILL |
| top_mcc_shift | binary | AUC > 0.60 | 0.6297 | YES | DISTILL |
| nba_primary | multiclass (7) | F1 > 0.286 | 0.187 | NO | DIRECT |
| segment_prediction | multiclass (4) | F1 > 0.500 | 0.403 | NO | DIRECT |
| next_mcc | multiclass (50) | F1 > 0.040 | 0.012 | NO | DIRECT |
| product_stability | regression | R2 > 0.05 | 0.010 | NO | DIRECT |
| cross_sell_count | regression | R2 > 0.05 | 0.015 | NO | DIRECT |
| mcc_diversity_trend | regression | R2 > 0.05 | 0.037 | NO | DIRECT |

**Result: 7 DISTILL / 6 DIRECT**

## Distillation Fidelity (distilled tasks)

| Task | AUC Gap | Agreement | JSD | Ranking Corr | Calibration Gap |
|------|---------|-----------|-----|-------------|----------------|
| churn_signal | 0.022 | 88.9% | 0.006 | 0.964 | 0.101 |
| will_acquire_accounts | 0.024 | 92.5% | 0.005 | 0.982 | 0.082 |
| will_acquire_payments | 0.032 | 90.8% | 0.005 | 0.977 | 0.089 |
| will_acquire_deposits | 0.018 | 79.8% | 0.005 | 0.985 | 0.091 |
| will_acquire_investments | 0.023 | 79.7% | 0.005 | 0.975 | 0.096 |
| will_acquire_lending | 0.026 | 81.2% | 0.005 | 0.963 | 0.094 |
| top_mcc_shift | 0.036 | 99.9% | 0.000 | 0.965 | 0.008 |

**Average AUC gap: 2.6%p** (within 2-5%p target)
**Main failure mode: calibration_gap > 0.05** (probability calibration issue, not ranking)

## Direct Hard-Label Results (threshold-routed tasks)

| Task | Type | Teacher F1/R2 | Student F1/MAE | Fidelity |
|------|------|---------------|----------------|----------|
| nba_primary | multiclass | F1 0.187 | F1 0.076 | agreement 15.4% |
| segment_prediction | multiclass | F1 0.403 | F1 0.181 | agreement 21.8% |
| next_mcc | multiclass | F1 0.012 | F1 0.017 | agreement 1.3% |
| product_stability | regression | R2 0.010 | MAE 0.122 | PASS |
| cross_sell_count | regression | R2 0.015 | MAE 0.970 | RMSE gap 0.176 |
| mcc_diversity_trend | regression | R2 0.037 | MAE 1.763 | PASS |

## Key Findings

1. **Binary tasks: distillation effective** - AUC gap 2-3%p, ranking correlation 0.96+
2. **Multiclass tasks: teacher too weak for distillation** - below 2x random threshold
3. **Regression tasks: borderline** - teacher R2 < 0.05, direct training comparable
4. **Calibration is the main issue** - LGBM probability outputs need post-hoc calibration
5. **Adaptive routing prevents low-quality knowledge propagation** - MRM safeguard

## Implications for Production

- Binary product acquisition tasks (7/13) can be safely distilled
- Multiclass/regression tasks should use direct LGBM training until teacher improves
- On-prem with real financial data should re-evaluate thresholds
- Platt scaling or isotonic regression can address calibration gap
