# Predicting Student Subscription Purchases

A machine learning project that predicts whether a student on a free plan will buy a subscription, based on their activity on an online learning platform (lectures watched, courses started, exam activity).

The business use case: build a list of students likely to purchase, so they can be retargeted with ads ahead of a sales campaign. The difficulty is class imbalance: only about 9% of students purchase, so a model that always predicts "no purchase" already scores about 90.8% accuracy.

## Data

- 17,761 students, 7 predictor columns and 1 binary target (`purchased`).
- The dataset comes from a course project (365 Data Science) and is **not included in this repository**. Place `ml_datasource.csv` in the project folder to run the notebook.

## Approach

1. **Outlier removal:** cut extreme values on minutes watched, courses started, practice exams started and minutes on exams (0.76% of rows removed, 7 of 1,631 purchasers lost).
2. **Multicollinearity:** used the Variance Inflation Factor (VIF); dropped `practice_exams_started` (VIF 9.73), after which all VIFs were below 5.
3. **Missing values:** the 8 `NaN` values in `student_country` were Namibia's country code ("NA") misread by pandas, and were filled with `NAM`.
4. **Split and encoding:** stratified 80/20 train/test split (`random_state=365`); `student_country` ordinal-encoded, fitted on the training set only.
5. **Models:** logistic regression, K-nearest neighbors, support vector machine, decision tree and random forest, with grid search for the KNN, SVM and decision tree (cost-complexity pruning).

## Results

Metrics are for the buyer class on the test set (3,526 students, 325 buyers). Baseline accuracy (always predict "no purchase"): 0.908.

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic regression | 0.950 | 0.82 | 0.59 | 0.69 |
| KNN (tuned: 10 neighbors, distance weights) | 0.955 | 0.79 | 0.71 | 0.75 |
| SVM (tuned) | TODO | TODO | TODO | TODO |
| Decision tree (tuned: `ccp_alpha` 0.001) | 0.962 | 0.81 | 0.76 | 0.78 |
| Random forest | 0.960 | 0.81 | 0.74 | 0.77 |

**Takeaways**

- Accuracy alone is misleading here; precision and recall for the buyer class tell the real story.
- The tuned decision tree and the random forest find the most buyers; the untuned tree overfit, and pruning fixed it.
- `days_on_platform` and `minutes_watched` drive most of the predictions.
- Logistic regression (statsmodels) is highly significant overall (LLR p-value ≈ 0), but misses about 4 in 10 buyers.

## Limitations and next steps

- A single train/test split; cross-validation would give a more reliable comparison.
- `days_on_platform` dominates, which may partly reflect how the data was built (activity is measured up to the first purchase for buyers, but up to a fixed date for non-buyers).
- Possible improvements: feature scaling, class weights or resampling, threshold tuning, better encoding of country, feature engineering and boosting models.

## How to run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook "Machine Learning Project.ipynb"
```

## Credits

Project brief and dataset: 365 Data Science. Analysis and write-up: Buhlebethu Biyela.
