from __future__ import annotations
import csv
import json
import math
from collections import Counter, defaultdict
from feature_engineering import fit_imputers, load_csv, save_csv, transform_rows
from train_model import NUMERIC_FEATURES, sigmoid


def predict_tree_one(node: dict, row: list[float]) -> float:
    while not node["leaf"]:
        if row[node["feat"]] <= node["thresh"]:
            node = node["left"]
        else:
            node = node["right"]
    return node["val"]


def main() -> None:
    # 1. Load frozen GBDT model from trained_model.json
    with open("trained_model.json", "r", encoding="utf-8") as f:
        artifacts = json.load(f)

    gbdt_model = artifacts["gbdt"]
    features = artifacts["features"]
    base_score = gbdt_model["base_score"]
    lr = gbdt_model["learning_rate"]
    trees = gbdt_model["trees"]

    # Save standalone frozen GBDT artifact
    frozen_artifact = {
        "model_type": "GradientBoostingClassifier (Precision-Tuned)",
        "frozen": True,
        "neg_weight": artifacts.get("neg_weight", 1.45),
        "threshold": 0.50,
        "cv_metrics": artifacts["cv_results_at_050"]["Gradient Boosting (Prec-Tuned)"],
        "features": features,
        "gbdt": gbdt_model,
    }
    with open("frozen_gbdt_model.json", "w", encoding="utf-8") as f:
        json.dump(frozen_artifact, f, indent=2)

    # 2. Fit feature imputers strictly on train.csv, then transform test.csv
    train_raw = load_csv("train.csv")
    test_raw = load_csv("test.csv")
    imputers = fit_imputers(train_raw)

    # For TicketGroupSize on test, combine train+test ticket counts so shared family tickets in test are recognized
    imputers["ticket_counts"] = Counter(r["Ticket"].strip() for r in (train_raw + test_raw))

    test_fe, test_cols = transform_rows(test_raw, imputers, is_train=False)
    save_csv("test_features.csv", test_fe, test_cols)

    # 3. Run inference with frozen GBDT model
    submission_rows = []
    detailed_rows = []

    for r_raw, r_fe in zip(test_raw, test_fe):
        x = [float(r_fe[col]) for col in features]
        z = base_score + sum(lr * predict_tree_one(t, x) for t in trees)
        prob = sigmoid(z)
        pred = 1 if prob >= 0.50 else 0

        submission_rows.append(
            {
                "PassengerId": r_fe["PassengerId"],
                "Survived": pred,
            }
        )
        detailed_rows.append(
            {
                "PassengerId": r_fe["PassengerId"],
                "Survived": pred,
                "SurvivalProbability": round(prob, 4),
                "Pclass": r_raw["Pclass"],
                "Sex": r_raw["Sex"],
                "Age": r_fe["Age"],
                "Title": r_fe["Title"],
                "Fare": r_fe["Fare"],
                "Embarked": r_fe["Embarked"],
                "Name": r_raw["Name"],
            }
        )

    save_csv("submission.csv", submission_rows, ["PassengerId", "Survived"])
    save_csv(
        "test_predictions_detailed.csv",
        detailed_rows,
        [
            "PassengerId",
            "Survived",
            "SurvivalProbability",
            "Pclass",
            "Sex",
            "Age",
            "Title",
            "Fare",
            "Embarked",
            "Name",
        ],
    )

    # 4. Print summary of test predictions
    total = len(submission_rows)
    survived_cnt = sum(r["Survived"] for r in submission_rows)
    not_survived_cnt = total - survived_cnt

    print("=== Frozen Model: Gradient Boosting (Prec-Tuned) ===")
    print(f"Frozen artifact saved to: frozen_gbdt_model.json")
    print(f"Submission saved to:      submission.csv ({total} rows)")
    print(f"Detailed output saved to: test_predictions_detailed.csv\n")

    print("=== Test Set Prediction Summary (test.csv) ===")
    print(f"Total Passengers : {total}")
    print(f"Predicted 1 (Survived)     : {survived_cnt:3d} ({survived_cnt/total*100:5.1f}%)")
    print(f"Predicted 0 (Not Survived) : {not_survived_cnt:3d} ({not_survived_cnt/total*100:5.1f}%)\n")

    for col in ["Sex", "Pclass", "Title"]:
        groups = defaultdict(list)
        for r in detailed_rows:
            groups[r[col]].append(r["Survived"])
        print(f"Predicted Survival by {col}:")
        for k, v in sorted(groups.items(), key=lambda x: str(x[0])):
            print(f"  {str(k):8s}: {sum(v):3d}/{len(v):3d} ({sum(v)/len(v)*100:5.1f}%)")
        print()

    print("First 15 rows of submission.csv:")
    for r in detailed_rows[:15]:
        print(
            f"  PassengerId={r['PassengerId']:4d} | Survived={r['Survived']} "
            f"(prob={r['SurvivalProbability']:.3f}) | {r['Sex']:6s} | Pclass={r['Pclass']} | {r['Name']}"
        )


if __name__ == "__main__":
    main()
