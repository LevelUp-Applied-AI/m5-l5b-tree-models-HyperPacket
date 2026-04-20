# Challenge Tiers Analysis

## Tier 1 - Threshold Tuning
**Max F1 Threshold:** 0.35  
**Threshold for $\ge$ 80% Recall:** 0.25

**Recommendation for Retention Campaign:**
If Petra Telecom can only contact 200 customers per month for retention offers, the recommendation is to use a higher threshold closer to the Max F1 threshold (e.g., 0.35 or higher), rather than targeting 80% recall. Reaching 80% recall requires dropping the threshold to 0.25, which flags far too many customers (creating false positives), meaning the limited 200 slots would be wasted on customers who weren't actually going to churn. A threshold that prioritizes precision or balances F1 ensures the 200 interventions are highly targeted at customers with genuine churn risk.

## Tier 2 - Permutation Importance vs MDI
**MDI Top 3:**
1. num_support_calls (0.2617)
2. monthly_charges (0.2205)
3. tenure (0.1807)

**Permutation Importance Top 3:**
1. num_support_calls (0.0598)
2. contract_months (0.0412)
3. has_dependents (0.0044)

**Explanation of Disagreement:**
Mean Decrease in Impurity (MDI) inherently favors continuous, high-cardinality features like `monthly_charges` and `tenure` because the tree can make many splits on them, artificially inflating their perceived importance on the training set. Permutation Importance, on the other hand, measures the actual drop in model performance on the *test* set when a feature's values are randomly shuffled. This reveals that structural features like `contract_months` and `num_support_calls` are the true drivers of generalization, while continuous monetary values had somewhat inflated importance due to MDI's bias.

## Tier 3 - Custom Voting Ensemble
**PR-AUC Comparison:**
- LR: 0.392
- Balanced DT: 0.403
- Balanced RF: 0.419
- **Voting Ensemble: 0.448**

**Analysis:**
Yes, the custom Voting Ensemble outperformed the best individual model on the threshold-independent PR-AUC metric (0.448 vs the RF's 0.419). We would expect an ensemble of diverse models to outperform individual components when the base models make uncorrelated errors. Because Logistic Regression (a linear model) captures global additive effects smoothly, while Trees and Random Forests capture non-linear interactions and threshold effects, combining them allows the ensemble to smooth out the jagged decision boundaries of trees while retaining their ability to model complex interactions.
