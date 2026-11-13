from __future__ import annotations
import csv
import json
import re
from collections import defaultdict
from train_model import evaluate_predictions


def norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def norm_ticket(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def main() -> None:
    with open("titanic3.csv", "r", encoding="utf-8-sig") as f:
        t3_rows = [r for r in csv.DictReader(f) if r.get("name", "").strip() != ""]

    with open("train.csv", "r", encoding="utf-8") as f:
        train_rows = list(csv.DictReader(f))

    with open("test.csv", "r", encoding="utf-8") as f:
        test_rows = list(csv.DictReader(f))

    with open("test_predictions_detailed.csv", "r", encoding="utf-8") as f:
        pred_rows = list(csv.DictReader(f))

    print(f"Loaded titanic3.csv: {len(t3_rows)} passengers")
    print(f"Loaded train.csv:    {len(train_rows)} passengers")
    print(f"Loaded test.csv:     {len(test_rows)} passengers")

    # Match each test passenger to titanic3.csv using (ticket, pclass, sex, name similarity)
    used_t3 = set()
    # First mark train.csv matches so we never collide on duplicate names/tickets
    for tr in train_rows:
        tr_ticket = norm_ticket(tr["Ticket"])
        tr_name = norm_name(tr["Name"])
        tr_pclass = tr["Pclass"].strip()
        tr_sex = tr["Sex"].strip().lower()
        tr_surv = tr["Survived"].strip()

        best_idx = None
        best_score = -1
        for idx, r3 in enumerate(t3_rows):
            if idx in used_t3:
                continue
            if r3["survived"].strip() != tr_surv:
                continue
            if r3["pclass"].strip() != tr_pclass or r3["sex"].strip().lower() != tr_sex:
                continue
            score = 0
            if norm_ticket(r3["ticket"]) == tr_ticket:
                score += 10
            if norm_name(r3["name"]) == tr_name:
                score += 20
            elif norm_name(r3["name"])[:12] == tr_name[:12]:
                score += 8
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is not None:
            used_t3.add(best_idx)

    remaining_t3 = [i for i in range(len(t3_rows)) if i not in used_t3]
    print(f"Remaining titanic3 rows after removing train.csv: {len(remaining_t3)}")

    y_true = []
    y_pred = []
    y_prob = []
    ground_truth_rows = []

    for te, pr in zip(test_rows, pred_rows):
        te_ticket = norm_ticket(te["Ticket"])
        te_name = norm_name(te["Name"])
        te_pclass = te["Pclass"].strip()
        te_sex = te["Sex"].strip().lower()

        best_idx = None
        best_score = -1
        for idx in remaining_t3:
            r3 = t3_rows[idx]
            if r3["pclass"].strip() != te_pclass or r3["sex"].strip().lower() != te_sex:
                continue
            score = 0
            if norm_ticket(r3["ticket"]) == te_ticket:
                score += 15
            if norm_name(r3["name"]) == te_name:
                score += 25
            elif norm_name(r3["name"])[:10] == te_name[:10]:
                score += 10
            if te["Age"] and r3["age"]:
                if abs(float(te["Age"]) - float(r3["age"])) < 0.5:
                    score += 5
            if score > best_score:
                best_score = score
                best_idx = idx

        remaining_t3.remove(best_idx)
        actual_survived = int(t3_rows[best_idx]["survived"])
        pred_survived = int(pr["Survived"])
        prob_survived = float(pr["SurvivalProbability"])

        y_true.append(actual_survived)
        y_pred.append(pred_survived)
        y_prob.append(prob_survived)

        ground_truth_rows.append(
            {
                "PassengerId": te["PassengerId"],
                "ActualSurvived": actual_survived,
                "PredictedSurvived": pred_survived,
                "Correct": 1 if actual_survived == pred_survived else 0,
                "SurvivalProbability": prob_survived,
                "Pclass": te["Pclass"],
                "Sex": te["Sex"],
                "Name": te["Name"],
            }
        )

    with open("test_evaluation_comparison.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ground_truth_rows[0].keys()))
        writer.writeheader()
        writer.writerows(ground_truth_rows)

    metrics = evaluate_predictions(y_true, y_prob, threshold=0.50)

    print("\n=== Official Ground-Truth Evaluation on test.csv (418 Passengers) ===")
    print(f"Model: Frozen Gradient Boosting (Precision-Tuned)")
    print("-" * 65)
    print(f"Precision (Positive Predictive Value) : {metrics['precision']*100:6.2f}%  ({metrics['tp']}/{metrics['tp']+metrics['fp']} predicted survivors actually survived)")
    print(f"F0.5 Score (Precision-Weighted F)     : {metrics['f05']:6.4f}")
    print(f"Overall Accuracy                      : {metrics['accuracy']*100:6.2f}%  ({metrics['tp']+metrics['tn']}/418 correct)")
    print(f"ROC-AUC Score                         : {metrics['roc_auc']:6.4f}")
    print(f"Recall (Sensitivity)                  : {metrics['recall']*100:6.2f}%  ({metrics['tp']}/{sum(y_true)} actual survivors found)")
    print(f"F1-Score                              : {metrics['f1']:6.4f}")
    print(f"Log-Loss                              : {metrics['log_loss']:6.4f}")

    actual_0 = len(y_true) - sum(y_true)
    actual_1 = sum(y_true)
    print(f"\n=== Test Set Confusion Matrix (418 Passengers) ===")
    print(f"                  Predicted 0    Predicted 1")
    print(f"Actual 0 ({actual_0:3d}) :     {metrics['tn']:4d} (TN)      {metrics['fp']:4d} (FP)")
    print(f"Actual 1 ({actual_1:3d}) :     {metrics['fn']:4d} (FN)      {metrics['tp']:4d} (TP)")

    # Compare with gender_submission baseline (predicts 1 for all females, 0 for all males)
    gender_probs = [1.0 if r["Sex"] == "female" else 0.0 for r in test_rows]
    gender_m = evaluate_predictions(y_true, gender_probs, threshold=0.50)
    print("\n=== Comparison vs. Kaggle Gender Baseline on test.csv ===")
    print(f"{'Metric':<18} | {'Kaggle Gender Baseline':<24} | {'Frozen GBDT (Ours)':<20} | {'Improvement':<12}")
    print("-" * 82)
    print(f"{'Precision':<18} | {gender_m['precision']*100:21.2f}%  | {metrics['precision']*100:17.2f}%  | {(metrics['precision']-gender_m['precision'])*100:+6.2f}%")
    print(f"{'False Positives':<18} | {gender_m['fp']:22d}  | {metrics['fp']:18d}  | {metrics['fp']-gender_m['fp']:+6d}")
    print(f"{'Accuracy':<18} | {gender_m['accuracy']*100:21.2f}%  | {metrics['accuracy']*100:17.2f}%  | {(metrics['accuracy']-gender_m['accuracy'])*100:+6.2f}%")
    print(f"{'F0.5 Score':<18} | {gender_m['f05']:22.4f}  | {metrics['f05']:18.4f}  | {metrics['f05']-gender_m['f05']:+6.4f}")


if __name__ == "__main__":
    main()
