# Bank Marketing Subscription Prediction

<p align="center">
  <strong>ES 335 · Machine Learning Course Project</strong>
</p>

<p align="center">
  <a href="https://23110350.github.io/Bank-Marketing/">
    <strong>🌐 View the Live Project Website</strong>
  </a>
</p>

---

## Overview

This project studies whether information available before a bank marketing call can help predict whether a customer will subscribe to a term deposit. It uses the UCI Bank Marketing dataset to compare a majority-class baseline with four supervised classification models.

The analysis focuses on the challenges of an imbalanced classification problem: a model can achieve high overall accuracy while failing to identify customers who actually subscribe. For that reason, the comparison considers precision, recall, F1-score, balanced accuracy, and ROC-AUC alongside accuracy.

## Research Question

**How effectively can classification models predict term-deposit subscription using customer and campaign information available before a marketing call, and how does the classification threshold affect the precision–recall trade-off?**

## Dataset

The project uses the **Bank Marketing `bank-full.csv` dataset**, containing 45,211 observations. The target variable is `y`, which indicates whether a customer subscribed to a term deposit.

- **Source:** [UCI Machine Learning Repository — Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- **Observations:** 45,211
- **Target:** `y` (`yes` / `no`)
- **Prediction task:** Binary classification

### Data-handling decisions

- The `duration` feature is excluded because call duration is only known after the call has ended. Including it would make the model unsuitable for a decision that must be made before contacting the customer.
- The `pdays = -1` value is retained to represent the absence of a previous contact. A separate indicator is used to distinguish customers with and without a previous contact.
- The target classes are imbalanced, so accuracy is not treated as the sole measure of model quality.

## Methodology

The experiment is implemented in `analysis/reproduce_analysis.py`.

1. A stratified 80:20 train–test split is used.
2. Three-fold stratified cross-validation is performed on the training partition.
3. Preprocessing is fitted within scikit-learn pipelines, so transformations are learned from the relevant training fold rather than the validation or test data.
4. A majority-class baseline is compared with Logistic Regression, Decision Tree, Random Forest, and XGBoost.
5. Class weighting is used where supported to address class imbalance.
6. The held-out test set is reserved for the final model comparison.

The browser-based threshold explorer uses stored held-out predictions. It recalculates classification metrics for the selected threshold; it does not train models in the browser.

## Models Compared

- **Majority-class baseline** — predicts the most frequent class for every observation.
- **Logistic Regression** — linear classification baseline.
- **Decision Tree** — rule-based nonlinear classifier.
- **Random Forest** — ensemble of decision trees.
- **XGBoost** — gradient-boosted tree ensemble.

## Results

The held-out test set contains 9,043 observations: 7,985 non-subscribers and 1,058 subscribers.

| Model | Accuracy | Balanced accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Majority baseline | 0.8830 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |
| Logistic Regression | 0.7553 | 0.6983 | 0.2667 | 0.6238 | 0.3736 | 0.7722 |
| Decision Tree | 0.7364 | 0.6666 | 0.2394 | 0.5756 | 0.3381 | 0.7175 |
| Random Forest | 0.8251 | **0.7300** | 0.3549 | 0.6059 | **0.4476** | **0.7976** |
| XGBoost | **0.8940** | 0.5730 | **0.7181** | 0.1541 | 0.2537 | 0.7807 |

### Key findings

- The majority baseline achieves 88.30% accuracy simply by predicting the non-subscription class, but it identifies no subscribers. This demonstrates why accuracy alone is misleading for this dataset.
- Random Forest achieves the highest subscriber-class F1-score (0.4476) and ROC-AUC (0.7976) among the evaluated candidate models.
- XGBoost achieves the highest accuracy (89.40%) and precision (0.7181), but its subscriber recall is only 0.1541 at the default threshold.
- Logistic Regression has higher subscriber recall (0.6238) than Random Forest, but lower precision and F1-score.

The preferred model therefore depends on the campaign objective. If missing likely subscribers is especially costly, recall may receive greater weight; if contacting unlikely subscribers is costly, precision may matter more. The threshold explorer illustrates this trade-off.

## Project Website

The interactive website presents the project background, dataset, exploratory analysis, preprocessing, model explanations, results, and conclusion.

**[Open the live Bank Marketing ML project website](https://23110350.github.io/Bank-Marketing/)**

## Repository Structure

```text
Bank-Marketing/
├── analysis/
│   └── reproduce_analysis.py
├── data/
│   └── bank-full.csv
├── docs/
│   ├── index.html
│   ├── results.html
│   ├── dataset.html
│   ├── preprocessing.html
│   ├── eda.html
│   ├── logistic.html
│   ├── decision_tree.html
│   ├── random_forest.html
│   ├── xgboost.html
│   ├── conclusion.html
│   ├── roadmap.html
│   ├── team.html
│   ├── css/
│   ├── js/
│   ├── images/
│   ├── data/
│   └── notebooks/
├── notebooks/
│   └── Reproduce_Experiment.ipynb
├── outputs/
│   ├── model_comparison.csv
│   ├── test_predictions.json
│   └── experiment_metadata.json
├── scripts/
│   └── sync_website.py
├── requirements.txt
└── README.md
```

`docs/` contains the published website and is the GitHub Pages deployment directory. The root `notebooks/` directory contains the analysis notebooks; `docs/notebooks/` contains the notebook copies made available through the website.

## Reproducing the Analysis

### Requirements

- Python 3.10 or later
- Packages listed in `requirements.txt`

### Steps

1. Clone or download this repository.
2. From the repository root, create and activate a virtual environment.
3. Install the dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Confirm that `data/bank-full.csv` is present. If needed, obtain the dataset from the [UCI Bank Marketing page](https://archive.ics.uci.edu/dataset/222/bank+marketing).
5. Run the experiment:

   ```bash
   python analysis/reproduce_analysis.py
   ```

6. Review the generated comparison and prediction files in `outputs/`.

The primary notebook, `notebooks/Reproduce_Experiment.ipynb`, provides a notebook-based entry point to the experiment.

## Limitations

- The results measure predictive performance on a held-out split from this dataset; they do not establish causal relationships.
- Performance may change with a different train–test split, data period, customer population, or campaign strategy.
- The experiment uses a fixed dataset and does not evaluate deployment-time drift, operational costs, or the financial return of a targeting campaign.
- Precision and recall depend on the classification threshold. A threshold suitable for one campaign objective may not be suitable for another.

---

## Project Information

**Course:** ES 335 — Machine Learning  
**Institution:** Indian Institute of Technology Gandhinagar
