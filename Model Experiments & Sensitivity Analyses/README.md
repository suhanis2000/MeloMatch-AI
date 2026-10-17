# ExtraTrees Model Experiments

Version 5 of the MeloMatch-AI classifier: one main experiment and two sensitivity analyses that each change a single design decision to test how robust the result is.

| Notebook | What changes |
|---|---|
| [`MeloMatch_V5_BPM_+_Oversampler.ipynb`](MeloMatch_V5_BPM_+_Oversampler.ipynb) | **Main experiment.** All features including BPM, random oversampling, 5 candidate models |
| [`Sensitivity_Analysis_MeloMatch_V5_w_o_BPM.ipynb`](Sensitivity_Analysis_MeloMatch_V5_w_o_BPM.ipynb) | BPM removed from every feature set |
| [`Sensitivity_Analysis_MeloMatch_V5_Class_weighted_ExtraTrees_.ipynb`](Sensitivity_Analysis_MeloMatch_V5_Class_weighted_ExtraTrees_.ipynb) | No oversampling; ExtraTrees with `class_weight='balanced'` only |
## Shared methodology

All three notebooks follow a similar structure.

**1. Target.** `Fav genre` is mapped to four clusters:

| Label | Cluster | Genres | n |
|---|---|---|---|
| 0 | Calm-Reflective | Classical, Jazz, Folk, Lofi | 113 |
| 1 | High-Intensity | Metal, Rock, EDM, Rap, Hip hop | 370 |
| 2 | Melodic-Vocal | Pop, K pop, R&B | 175 |
| 3 | Other/Residual | Video game music, Country, Gospel, Latin | 78 |

**2. Feature sets.**

| Set | Features | # |
|---|---|---|
| A — mental health | Anxiety, Depression, Insomnia, OCD | 4 |
| B — listening habits | Age, Hours per day, BPM, Genre_Diversity, Instrumentalist, Composer, While working, Exploratory, Foreign languages | 9 |
| C — combined | A + B | 13 |

`Genre_Diversity` = number of genres listened to "Sometimes" or "Very frequently".

**3. Preprocessing** (`ColumnTransformer`)
- Numeric → `SimpleImputer(median)` → `StandardScaler`
- Categorical → `SimpleImputer(most_frequent)` → `OneHotEncoder(handle_unknown='ignore')`

**4. Split.** One stratified 80/20 split (`random_state=42`) shared by all feature sets: **588 train / 148 holdout**. The holdout is touched only once, at the end.

**5. Model search.** `GridSearchCV` with `StratifiedKFold(5, shuffle=True, random_state=42)`, refitting on **macro F1** (balanced accuracy and accuracy also logged). Resampling sits inside an `imblearn` pipeline, so it happens only on training folds.

| Model | Grid |
|---|---|
| SVC | linear: C ∈ {0.1, 1, 10}; rbf: C ∈ {0.1, 1, 10}, γ ∈ {scale, 0.01, 0.1} |
| Logistic Regression | C ∈ {0.01, 0.1, 1, 10} |
| CatBoost | iterations ∈ {100, 200, 400}, lr ∈ {0.03, 0.1}, depth ∈ {4, 6, 8}, l2 ∈ {3, 10} |
| ExtraTrees | n_estimators ∈ {200, 500}, max_depth ∈ {None, 10, 20}, min_samples_leaf ∈ {1, 2, 4}, criterion ∈ {gini, entropy} |
| Random Forest | same grid as ExtraTrees |

**6. Ablation.** The winning model is re-tuned separately on Sets A, B and C.

**7. Final evaluation** on the holdout with the Set C model:
- macro F1, balanced accuracy, accuracy, per-class report
- raw and normalized confusion matrices
- majority-class and stratified-random `DummyClassifier` baselines
- 1,000-sample bootstrap 95% CIs for accuracy and macro F1
- paired bootstrap 95% CI for (model − stratified baseline) macro F1
- error analysis: most common confusion pairs and prediction confidence

---

## 2. Sensitivity: no BPM — `Sensitivity_Analysis_MeloMatch_V5_w_o_BPM.ipynb`

**Why:** BPM has 107 missing values and extreme outliers (up to 999,999,999). This run removes it from Sets B and C (8 and 12 features).

**Takeaway:** removing BPM doesn't hurt and slightly helps; the winning algorithm switches from ExtraTrees to Random Forest, with CatBoost essentially tied. In its raw form, BPM adds noise rather than signal.

---

## 3. Sensitivity: class weighting — `Sensitivity_Analysis_MeloMatch_V5_Class_weighted_ExtraTrees_.ipynb`

**Why:** to check whether the result depends on `RandomOverSampler`. This run removes oversampling and handles imbalance with `ExtraTreesClassifier(class_weight='balanced')` instead. Only ExtraTrees is searched.


## Running these notebooks

The notebooks were written for Google Colab and read the CSV from a personal Drive path. Before running:

1. Change the `pd.read_csv(...)` path to point at [`../Data/mxmh_survey_results.csv`](../Data/mxmh_survey_results.csv) (or your uploaded copy in Colab).
2. If running locally, remove the `google.colab` import and `drive.mount(...)`.
3. Install dependencies: `pip install pandas numpy scikit-learn imbalanced-learn catboost shap matplotlib seaborn`.

> **Note:** the "Open in Colab" badge at the top of each notebook points to the repo root, but the notebooks live in this folder, so the badge link is broken. Open them from Colab via *File → Open notebook → GitHub* instead, or update the badge URL to include `ExtraTrees%20Model%20Experiments/`.
