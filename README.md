# Titanic Survival Prediction — Precision-Tuned Gradient Boosting (`GBDT`)

End-to-end machine learning pipeline for the **Kaggle Titanic Survival Prediction** problem, featuring leak-free in-fold feature engineering, 5-fold stratified cross-validation across five model families, **cost-sensitive precision tuning**, and ground-truth evaluation against the complete historical passenger manifest (`titanic3.csv`).

---

## 1. Project Structure

```text
├── README.md                        # Project documentation & evaluation report
├── train.csv                        # Training dataset (891 passengers, 12 columns)
├── test.csv                         # Unseen test dataset (418 passengers, 11 columns)
├── feature_engineering.py           # Leak-free feature engineering & imputation pipeline
├── train_model.py                   # 5-fold stratified CV & cost-sensitive model training
├── predict_test.py                  # Inference pipeline using the frozen GBDT model
├── evaluate_test.py                 # Ground-truth evaluation against titanic3.csv
├── frozen_gbdt_model.json           # Frozen Precision-Tuned Gradient Boosting artifact
├── trained_model.json               # Full trained model artifacts & CV metrics
├── submission.csv                   # Final Kaggle submission (PassengerId, Survived)
├── test_predictions_detailed.csv    # Test predictions with probabilities & features
├── test_evaluation_comparison.csv   # Row-by-row ground-truth comparison (418 rows)
└── titanic3.csv                     # Historical passenger manifest (1,309 passengers)
```

---

## 2. Exploratory Data Summary (`train.csv`)

* **Total Rows:** `891` passengers (`549` did not survive [`61.6%`], `342` survived [`38.4%`])
* **Missing Values in `train.csv`:**
  * `Age`: `177` missing (`19.9%`)
  * `Cabin`: `687` missing (`77.1%`)
  * `Embarked`: `2` missing (`0.2%`)
* **Primary Signal Highlights:**
  * **Sex:** `74.2%` of females survived vs. `18.9%` of males.
  * **Pclass:** `63.0%` survival in 1st Class, `47.3%` in 2nd Class, and `24.2%` in 3rd Class.
  * **Embarked:** `55.4%` survival from Cherbourg (`C`), `39.0%` from Queenstown (`Q`), and `33.7%` from Southampton (`S`).

---

## 3. Feature Engineering (`feature_engineering.py`)

All imputation statistics are fit **strictly on the training split** (inside each fold during cross-validation) to prevent data leakage:

| Feature | Engineering Logic | Train Survival Breakdown |
| :--- | :--- | :--- |
| **`Title`** | Extracted from `Name` (`Mrs`, `Miss`, `Master`, `Mr`, `Rare`) + one-hot encoded | `Mrs`: **79.4%** \| `Miss`: **70.3%** \| `Master`: **57.5%** \| `Rare`: **34.8%** \| `Mr`: **15.7%** |
| **`Age` & `AgeMissing`** | Imputed via median `Age` grouped by `(Title, Pclass)` + binary missingness flag | Captures class- and title-specific age distributions |
| **`IsChild` & `AgeBand`** | Indicator for `Age < 16` + age buckets (`Child`, `Teen`, `YoungAdult`, `Adult`, `Senior`) | `IsChild = 1`: **58.6%** vs. `IsChild = 0`: **36.2%** |
| **`FamilySize` & `FamilyCategory`** | `SibSp + Parch + 1` bucketed into `Alone` (`1`), `Small` (`2–4`), `Large` (`5+`) | `Small`: **57.9%** \| `Alone`: **30.4%** \| `Large`: **16.1%** |
| **`TicketGroupSize` & `FarePerPerson`** | Passengers sharing an exact `Ticket` number; `Fare / TicketGroupSize` | Normalizes group/family fares to per-person ticket cost |
| **`LogFare`** | $\log(1 + \text{Fare})$ | Compresses strong right-skewness (`0` to `512.33`) |
| **`HasCabin` & `Deck`** | Binary cabin indicator + deck letter (`A`–`G`, `U`) | `HasCabin = 1`: **66.7%** vs. `HasCabin = 0`: **30.0%** |

---

## 4. Model Training & Precision Retuning (`train_model.py`)

### 4.1 Baseline Accuracy-Tuned 5-Fold Stratified Cross-Validation

| Model | CV Accuracy (Mean ± Std) | OOF Accuracy | ROC-AUC | F1-Score | Precision | Recall | Log-Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Soft-Voting Ensemble** | **83.62% ± 1.43%** | **83.61%** | **0.8764** | **0.7781** | 0.8101 | 0.7485 | **0.4023** |
| **Random Forest** | 83.16% ± 1.92% | 83.16% | 0.8762 | 0.7692 | **0.8117** | 0.7310 | 0.4026 |
| **Logistic Regression (L2)** | 82.94% ± 2.36% | 82.94% | 0.8747 | 0.7718 | 0.7932 | **0.7515** | 0.4133 |
| **Gradient Boosting (GBDT)** | 82.94% ± 2.67% | 82.94% | 0.8708 | 0.7711 | 0.7950 | 0.7485 | 0.4079 |
| **Decision Tree (CART)** | 80.81% ± 2.96% | 80.81% | 0.8467 | 0.7482 | 0.7537 | 0.7427 | 0.9678 |

### 4.2 Cost-Sensitive Precision Retuning (`neg_weight = 1.45`)

To prioritize **Precision** (minimizing False Positives when predicting survival), models were retuned with:
1. **Cost-Sensitive Penalty (`neg_weight = 1.45`):** Negative-class (`Survived = 0`) errors are weighted $1.45\times$ higher in gradient updates, tree splits, and leaf values.
2. **Conservative Leaf Regularization:** Increased `min_samples_leaf` (`10` in GBDT, `6` in Random Forest) and L2 regularization (`l2_reg = 3.5` in GBDT).

| Model (Threshold = `0.50`) | CV Precision (Mean ± Std) | OOF Precision | $F_{0.5}$ Score | Recall | F1-Score | Accuracy | False Positives |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gradient Boosting (Prec-Tuned)** | **84.83% ± 2.75%** | **84.70%** | 0.8027 | 66.4% | 0.7443 | 82.5% | **41** |
| **Precision Soft-Voting Ensemble** | 84.60% ± 3.08% | 84.40% | **0.8095** | 69.6% | 0.7628 | 83.4% | 44 |
| **Logistic Regression (Prec-Tuned)** | 84.21% ± 4.84% | 83.74% | 0.8077 | **70.8%** | **0.7670** | **83.5%** | 47 |
| **Random Forest (Prec-Tuned)** | 83.63% ± 4.12% | 83.33% | 0.8032 | 70.2% | 0.7619 | 83.2% | 48 |
| **Decision Tree (Prec-Tuned)** | 83.15% ± 1.70% | 83.06% | 0.7672 | 58.8% | 0.6884 | 79.6% | **41** |

### 4.3 Top Feature Importances

| Rank | Random Forest (Gini Impurity) | GBDT (Gain Importance) | Logistic Regression (Std. Coef) |
| :---: | :--- | :--- | :--- |
| **1** | `Title_Mr` (`22.8%`) | `Title_Mr` (`38.4%`) | `Title_Mr` (`-0.47`) |
| **2** | `Sex` (`17.1%`) | `Sex` (`10.2%`) | `Sex` (`+0.45`) |
| **3** | `FarePerPerson` (`9.2%`) | `FarePerPerson` (`9.2%`) | `Pclass` (`-0.36`) |
| **4** | `LogFare` (`7.6%`) | `Pclass` (`8.1%`) | `Title_Mrs` (`+0.32`) |
| **5** | `Pclass` (`7.0%`) | `Age` (`6.3%`) | `FamCat_Large` (`-0.30`) |

---

## 5. Frozen Model Inference & Ground-Truth Evaluation (`test.csv`)

The **Precision-Tuned Gradient Boosting (`GBDT`)** model was frozen (`frozen_gbdt_model.json`) and evaluated on all `418` unseen passengers in `test.csv` against the historical ground-truth manifest (`titanic3.csv`).

### 5.1 Test Set Performance (`418` Unseen Passengers)

| Metric | Kaggle Gender Baseline | Frozen GBDT (Ours) | Improvement |
| :--- | :---: | :---: | :---: |
| **Precision** | `69.74%` | **75.59%** | **+5.85%** |
| **False Positives (FP)** | `46` | **31** | **-15 (-32.6%)** |
| **$F_{0.5}$ Score (Precision-Weighted)** | `0.6919` | **0.7207** | **+0.0288** |
| **Overall Accuracy** | `76.56%` (`320 / 418`) | **77.75%** (`325 / 418`) | **+1.20%** |
| **ROC-AUC** | — | **0.8086** | — |

### 5.2 Test Set Confusion Matrix (`418` Passengers)

| | Predicted `0` (Did Not Survive) | Predicted `1` (Survived) |
| :--- | :---: | :---: |
| **Actual `0` (`260` passengers)** | **229** (True Negatives — `88.1%` specificity) | **31** (False Positives) |
| **Actual `1` (`158` passengers)** | **62** (False Negatives) | **96** (True Positives) |

---

## 6. Reproducing Results

Run the full pipeline using standard Python 3 (no external dependencies required):

```bash
# 1. Engineer features on train.csv
python3 feature_engineering.py

# 2. Run 5-fold stratified CV & train precision-tuned models
python3 train_model.py

# 3. Run inference on test.csv using the frozen GBDT model
python3 predict_test.py

# 4. Evaluate test.csv predictions against titanic3.csv ground truth
python3 evaluate_test.py
```
