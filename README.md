# MeloMatch-AI

**A machine-learning study of whether a listener's mental-health profile and listening habits can predict the kind of music they gravitate toward**

MeloMatch-AI uses the *Music & Mental Health (MxMH)* survey to train classifiers that predict a person's favorite **genre cluster** (High-Intensity, Melodic-Vocal, Calm-Reflective, or Other) from self-reported anxiety, depression, insomnia and OCD scores, plus listening behavior such as hours per day, preferred BPM, genre diversity, and whether they believe their music engagement improves their mental health.

> This is an exploratory research project. Nothing here diagnoses, treats, or replaces care from a mental-health professional.

---

Full result details are in [`ExtraTrees Model Experiments/`](ExtraTrees%20Model%20Experiments/).

---

## Repository structure

```
MeloMatch-AI/
├── README.md                         
├── Data/
│   ├── README.md                      ← dataset description & data dictionary
│   └── mxmh_survey_results.csv        ← 736 survey responses, 33 columns
└── ExtraTrees Model Experiments/
    ├── README.md                      ← methodology & results for each notebook
    ├── MeloMatch_V5_BPM_+_Oversampler.ipynb                                ← main experiment
    ├── Sensitivity_Analysis_MeloMatch_V5_w_o_BPM.ipynb                     ← same pipeline without BPM
    └── Sensitivity_Analysis_MeloMatch_V5_Class_weighted_ExtraTrees_.ipynb  ← ExtraTrees class weights instead of oversampling
```

---

## How it works

1. **Target engineering** – the 16 favorite-genre answers are grouped into 4 clusters:

   | Cluster | Genres | Count |
   |---|---|---|
   | High-Intensity | Metal, Rock, EDM, Rap, Hip hop | n = 370 |
   | Melodic-Vocal | Pop, K pop, R&B | n = 175 |
   | Calm-Reflective | Classical, Jazz, Folk, Lofi | n = 113 |
   | Other/Residual | Video game music, Country, Gospel, Latin | n = 78 |

2. **Feature engineering** – `Genre_Diversity` score counts how many of the 16 genres a respondent listens to "Sometimes" or "Very frequently".
3. **Three feature sets** for ablation:
   - **A – Mental health (4):** Anxiety, Depression, Insomnia, OCD
   - **B – Listening Context (9):** Age, Hours per day, BPM, Genre_Diversity, Instrumentalist, Composer, While working, Exploratory, Foreign languages
   - **C – Combined (13):** A + B
4. **Pipeline** – median imputation + scaling for numeric columns, most-frequent imputation + one-hot encoding for categorical columns, `RandomOverSampler` to balance classes (inside each CV fold, so no leakage), then the classifier.
5. **Model selection** – SVC, Logistic Regression, CatBoost, ExtraTrees and Random Forest are tuned with `GridSearchCV` (stratified 5-fold, optimizing macro F1) on an 80/20 stratified split (588 train / 148 holdout).
6. **Evaluation** – the winner is re-tuned on each feature set, scored once on the untouched holdout, compared to dummy baselines, and given bootstrap confidence intervals and an error analysis.

---

## Getting started

### Run in Google Colab (easiest)

Open a notebook from [`ExtraTrees Model Experiments/`](ExtraTrees%20Model%20Experiments/) in Colab, upload `Data/mxmh_survey_results.csv`, and change the `pd.read_csv(...)` path in the data-loading cell to wherever you put the file.

> The notebooks currently load data from a personal Google Drive path (`/content/drive/MyDrive/...`). Update that line before running.

### Run locally

```bash
git clone https://github.com/sususuhaniS/MeloMatch-AI.git
cd MeloMatch-AI
pip install pandas numpy scikit-learn imbalanced-learn catboost shap matplotlib seaborn jupyter
jupyter notebook
```

Then, in each notebook:
- remove or comment out `from google.colab import drive` and `drive.mount(...)`
- change the CSV path to `../Data/mxmh_survey_results.csv`
- delete the `!pip install catboost` line (already installed above)

The full five-model grid search takes several minutes on a laptop.

---

## Limitations

- **Small, imbalanced, self-selected sample.** 736 online respondents, half of them in one cluster; results may not generalize.
- **Self-reported scores.** Mental-health ratings are 0–10 self-assessments, not clinical measures.
- **Noisy BPM column.** Values include 0 and 999,999,999; outliers are flagged but not removed, which may be why the no-BPM run does better.
- **Correlation, not causation.** Predicting what people like is not the same as knowing what helps them.
- **Modest signal.** Macro F1 around 0.32–0.34 on four classes; more data or richer features are needed before building a recommender on top.

---

## Roadmap

- [ ] Clean BPM (cap/remove implausible values) and re-run the main experiment
- [ ] Finish and record the class-weighted ExtraTrees sensitivity run
- [ ] Add SHAP feature-importance plots (SHAP is already imported)
- [ ] Explore the `Music effects` column (Improve / No effect / Worsen) as a target or filter
- [ ] Prototype the recommendation layer on top of the classifier

---

## Data & acknowledgments

Data comes from the **Music & Mental Health Survey Results** dataset on Kaggle, collected by Catherine Rasgaitis in 2022. See [`Data/README.md`](Data/README.md) for the full data dictionary.

Built with scikit-learn, imbalanced-learn, CatBoost, pandas, matplotlib and seaborn.

## Author

**Suhani Sharma** — [@sususuhaniS](https://github.com/sususuhaniS)
