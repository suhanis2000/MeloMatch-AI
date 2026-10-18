# MeloMatch-AI

**This is an exploratory study that aims to predict music preference clusters based on previous respondent data encapsulating listening-context and mental-health-related variables.**

> Nothing here diagnoses, treats, or replaces care from a mental-health professional.

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
└── Model Experiments & Sensitivity/
    ├── README.md                      ← methodology & results for each notebook
    ├── MeloMatch_V5_BPM_+_Oversampler.ipynb                                ← main experiment
    ├── Sensitivity_Analysis_MeloMatch_V5_w_o_BPM.ipynb                     ← same pipeline without BPM
    └── Sensitivity_Analysis_MeloMatch_V5_Class_weighted_ExtraTrees_.ipynb  ← ExtraTrees class weights instead of oversampling
```

---

## How it works

1. **Target engineering** – the 16 favorite-genre categories are grouped into 4 heuristic clusters:

   | Cluster | Genres | Count |
   |---|---|---|
   | High-Intensity | Metal, Rock, EDM, Rap, Hip hop | n = 370 |
   | Melodic-Vocal | Pop, K pop, R&B | n = 175 |
   | Calm-Reflective | Classical, Jazz, Folk, Lofi | n = 113 |
   | Other/Residual | Video game music, Country, Gospel, Latin | n = 78 |

2. **Feature engineering** – `Genre_Diversity` score counts how many of the 16 genres a respondent listens to "Sometimes" or "Very frequently".
3. **Three feature sets** were created for ablation:
   - **A – Mental health (4):** Anxiety, Depression, Insomnia, OCD
   - **B – Listening Context (9):** Age, Hours per day, BPM, Genre_Diversity, Instrumentalist, Composer, While working, Exploratory, Foreign languages
   - **C – Combined (13):** A + B
4. **Pipeline** – median imputation + scaling for numeric columns, most-frequent imputation + one-hot encoding for categorical columns, `RandomOverSampler` to balance classes (only for each CV fold), then the classifier.
5. **Model selection** – SVC, Logistic Regression, CatBoost, ExtraTrees, and Random Forest are tuned with `GridSearchCV` (stratified 5-fold, optimizing macro F1) on an 80/20 stratified split (588 train / 148 holdout) and selected based on macro-F1 scores.
6. **Analysis** – the winner is re-tuned on each feature set, scored once on the untouched holdout, compared to dummy baselines, and given bootstrap confidence intervals and an error analysis.

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

## Data & acknowledgments

Data comes from the **Music & Mental Health Survey Results** dataset on Kaggle, curated by Catherine Rasgaitis. See [`Data/README.md`](Data/README.md) for the full data dictionary.

Built with scikit-learn, imbalanced-learn, CatBoost, pandas, matplotlib, and seaborn.

## Author

**Suhani Sharma** — [@suhanis2000](https://github.com/suhanis2000)
