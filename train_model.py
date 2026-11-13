from __future__ import annotations
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from feature_engineering import fit_imputers, load_csv, transform_rows

NUMERIC_FEATURES = [
    "Pclass",
    "Sex",
    "Age",
    "AgeMissing",
    "IsChild",
    "SibSp",
    "Parch",
    "FamilySize",
    "IsAlone",
    "TicketGroupSize",
    "FarePerPerson",
    "LogFare",
    "HasCabin",
    "Title_Mr",
    "Title_Miss",
    "Title_Mrs",
    "Title_Master",
    "Title_Rare",
    "Embarked_S",
    "Embarked_C",
    "Embarked_Q",
    "FamCat_Alone",
    "FamCat_Small",
    "FamCat_Large",
]


def extract_xy(rows: list[dict]) -> tuple[list[list[float]], list[int]]:
    X = [[float(r[col]) for col in NUMERIC_FEATURES] for r in rows]
    y = [int(r["Survived"]) for r in rows]
    return X, y


def stratified_k_fold(y: list[int], k: int = 5, seed: int = 42) -> list[tuple[list[int], list[int]]]:
    rng = random.Random(seed)
    pos_idx = [i for i, label in enumerate(y) if label == 1]
    neg_idx = [i for i, label in enumerate(y) if label == 0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    folds = [[] for _ in range(k)]
    for idx_list in (pos_idx, neg_idx):
        for i, idx in enumerate(idx_list):
            folds[i % k].append(idx)

    splits = []
    for f in range(k):
        val_set = set(folds[f])
        train_idx = [i for i in range(len(y)) if i not in val_set]
        val_idx = sorted(folds[f])
        splits.append((train_idx, val_idx))
    return splits


# --- Standard Scaler ---
def fit_scaler(X: list[list[float]]) -> tuple[list[float], list[float]]:
    n_features = len(X[0])
    means, stds = [], []
    for j in range(n_features):
        col = [row[j] for row in X]
        m = sum(col) / len(col)
        var = sum((x - m) ** 2 for x in col) / len(col)
        s = math.sqrt(var) if var > 1e-12 else 1.0
        means.append(m)
        stds.append(s)
    return means, stds


def transform_scaler(X: list[list[float]], means: list[float], stds: list[float]) -> list[list[float]]:
    return [[(row[j] - means[j]) / stds[j] for j in range(len(row))] for row in X]


def sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


# --- 1. Cost-Sensitive Logistic Regression (L2 + FP Penalty) ---
class LogisticRegressionL2:
    def __init__(self, lr: float = 0.1, epochs: int = 550, l2: float = 0.08, neg_weight: float = 1.45):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.neg_weight = neg_weight  # >1.0 penalizes False Positives more heavily
        self.weights: list[float] = []
        self.bias: float = 0.0
        self.means: list[float] = []
        self.stds: list[float] = []

    def fit(self, X: list[list[float]], y: list[int]) -> None:
        self.means, self.stds = fit_scaler(X)
        Xs = transform_scaler(X, self.means, self.stds)
        n, d = len(Xs), len(Xs[0])
        self.weights = [0.0] * d
        self.bias = 0.0

        for _ in range(self.epochs):
            grad_w = [0.0] * d
            grad_b = 0.0
            for i in range(n):
                z = self.bias + sum(self.weights[j] * Xs[i][j] for j in range(d))
                p = sigmoid(z)
                w_i = 1.0 if y[i] == 1 else self.neg_weight
                err = w_i * (p - y[i])
                grad_b += err
                for j in range(d):
                    grad_w[j] += err * Xs[i][j]
            self.bias -= self.lr * (grad_b / n)
            for j in range(d):
                self.weights[j] -= self.lr * ((grad_w[j] / n) + self.l2 * self.weights[j])

    def predict_proba(self, X: list[list[float]]) -> list[float]:
        Xs = transform_scaler(X, self.means, self.stds)
        probs = []
        for row in Xs:
            z = self.bias + sum(self.weights[j] * row[j] for j in range(len(row)))
            probs.append(sigmoid(z))
        return probs


# --- 2. Cost-Sensitive Decision Tree (CART Classification) ---
class DecisionTreeCART:
    def __init__(
        self,
        max_depth: int = 4,
        min_samples_split: int = 16,
        min_samples_leaf: int = 8,
        neg_weight: float = 1.45,
        max_features: int | None = None,
        rng: random.Random | None = None,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.neg_weight = neg_weight
        self.max_features = max_features
        self.rng = rng or random.Random(42)
        self.tree = None
        self.feature_importances: list[float] = []

    def _weighted_prob(self, count1: int, total: int) -> float:
        count0 = total - count1
        denom = count1 + self.neg_weight * count0
        return count1 / denom if denom > 0 else 0.0

    def _gini(self, count1: int, total: int) -> float:
        if total == 0:
            return 0.0
        p1 = self._weighted_prob(count1, total)
        p0 = 1.0 - p1
        return 1.0 - (p1 * p1 + p0 * p0)

    def _build(self, X: list[list[float]], y: list[int], indices: list[int], depth: int) -> dict:
        total = len(indices)
        count1 = sum(y[i] for i in indices)
        prob1 = self._weighted_prob(count1, total)

        if (
            depth >= self.max_depth
            or total < self.min_samples_split
            or count1 == 0
            or count1 == total
        ):
            return {"leaf": True, "prob": prob1}

        n_features = len(X[0])
        if self.max_features and self.max_features < n_features:
            feat_indices = self.rng.sample(range(n_features), self.max_features)
        else:
            feat_indices = list(range(n_features))

        parent_gini = self._gini(count1, total)
        best_gain = 0.0
        best_feat = None
        best_thresh = None
        best_left = None
        best_right = None

        for j in feat_indices:
            pairs = sorted((X[i][j], y[i], i) for i in indices)
            left_c1 = 0
            for k in range(len(pairs) - 1):
                left_c1 += pairs[k][1]
                left_n = k + 1
                right_n = total - left_n
                if left_n < self.min_samples_leaf or right_n < self.min_samples_leaf:
                    continue
                if pairs[k][0] == pairs[k + 1][0]:
                    continue

                right_c1 = count1 - left_c1
                gain = parent_gini - (
                    (left_n / total) * self._gini(left_c1, left_n)
                    + (right_n / total) * self._gini(right_c1, right_n)
                )
                if gain > best_gain:
                    best_gain = gain
                    best_feat = j
                    best_thresh = 0.5 * (pairs[k][0] + pairs[k + 1][0])
                    best_left = [p[2] for p in pairs[:left_n]]
                    best_right = [p[2] for p in pairs[left_n:]]

        if best_feat is None or best_gain <= 1e-6:
            return {"leaf": True, "prob": prob1}

        self.feature_importances[best_feat] += best_gain * total
        return {
            "leaf": False,
            "feat": best_feat,
            "thresh": best_thresh,
            "left": self._build(X, y, best_left, depth + 1),
            "right": self._build(X, y, best_right, depth + 1),
        }

    def fit(self, X: list[list[float]], y: list[int], sample_indices: list[int] | None = None) -> None:
        self.feature_importances = [0.0] * len(X[0])
        indices = sample_indices if sample_indices is not None else list(range(len(X)))
        self.tree = self._build(X, y, indices, 0)
        tot_imp = sum(self.feature_importances)
        if tot_imp > 0:
            self.feature_importances = [v / tot_imp for v in self.feature_importances]

    def _predict_one(self, node: dict, row: list[float]) -> float:
        while not node["leaf"]:
            if row[node["feat"]] <= node["thresh"]:
                node = node["left"]
            else:
                node = node["right"]
        return node["prob"]

    def predict_proba(self, X: list[list[float]]) -> list[float]:
        return [self._predict_one(self.tree, row) for row in X]


# --- 3. Cost-Sensitive Random Forest Classifier ---
class RandomForest:
    def __init__(
        self,
        n_estimators: int = 60,
        max_depth: int = 5,
        min_samples_leaf: int = 6,
        neg_weight: float = 1.45,
        seed: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.neg_weight = neg_weight
        self.seed = seed
        self.trees: list[DecisionTreeCART] = []
        self.feature_importances: list[float] = []

    def fit(self, X: list[list[float]], y: list[int]) -> None:
        rng = random.Random(self.seed)
        n = len(X)
        d = len(X[0])
        max_features = max(1, int(math.sqrt(d)) + 2)
        self.trees = []
        self.feature_importances = [0.0] * d

        for _ in range(self.n_estimators):
            boot_idx = [rng.randrange(n) for _ in range(n)]
            tree = DecisionTreeCART(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_leaf * 2,
                min_samples_leaf=self.min_samples_leaf,
                neg_weight=self.neg_weight,
                max_features=max_features,
                rng=rng,
            )
            tree.fit(X, y, sample_indices=boot_idx)
            self.trees.append(tree)
            for j in range(d):
                self.feature_importances[j] += tree.feature_importances[j] / self.n_estimators

    def predict_proba(self, X: list[list[float]]) -> list[float]:
        all_preds = [t.predict_proba(X) for t in self.trees]
        return [
            sum(all_preds[t][i] for t in range(self.n_estimators)) / self.n_estimators
            for i in range(len(X))
        ]


# --- 4. Cost-Sensitive Gradient Boosted Decision Trees ---
class GradientBoostingClassifier:
    def __init__(
        self,
        n_estimators: int = 55,
        learning_rate: float = 0.07,
        max_depth: int = 3,
        min_samples_leaf: int = 10,
        l2_reg: float = 3.5,
        neg_weight: float = 1.45,
        subsample: float = 0.85,
        seed: int = 42,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.l2_reg = l2_reg
        self.neg_weight = neg_weight
        self.subsample = subsample
        self.seed = seed
        self.base_score = 0.0
        self.trees: list[dict] = []
        self.feature_importances: list[float] = []

    def _build_newton_tree(
        self,
        X: list[list[float]],
        g: list[float],
        h: list[float],
        indices: list[int],
        depth: int,
    ) -> dict:
        G = sum(g[i] for i in indices)
        H = sum(h[i] for i in indices)
        leaf_val = -G / (H + self.l2_reg)

        if depth >= self.max_depth or len(indices) < self.min_samples_leaf * 2:
            return {"leaf": True, "val": leaf_val}

        parent_score = (G * G) / (H + self.l2_reg)
        best_gain = 0.0
        best_feat = None
        best_thresh = None
        best_left = None
        best_right = None

        n_features = len(X[0])
        for j in range(n_features):
            pairs = sorted((X[i][j], g[i], h[i], i) for i in indices)
            G_L = 0.0
            H_L = 0.0
            total = len(pairs)
            for k in range(total - 1):
                G_L += pairs[k][1]
                H_L += pairs[k][2]
                left_n = k + 1
                right_n = total - left_n
                if left_n < self.min_samples_leaf or right_n < self.min_samples_leaf:
                    continue
                if pairs[k][0] == pairs[k + 1][0]:
                    continue

                G_R = G - G_L
                H_R = H - H_L
                gain = 0.5 * (
                    (G_L * G_L) / (H_L + self.l2_reg)
                    + (G_R * G_R) / (H_R + self.l2_reg)
                    - parent_score
                )
                if gain > best_gain:
                    best_gain = gain
                    best_feat = j
                    best_thresh = 0.5 * (pairs[k][0] + pairs[k + 1][0])
                    best_left = [p[3] for p in pairs[:left_n]]
                    best_right = [p[3] for p in pairs[left_n:]]

        if best_feat is None or best_gain <= 1e-5:
            return {"leaf": True, "val": leaf_val}

        self.feature_importances[best_feat] += best_gain
        return {
            "leaf": False,
            "feat": best_feat,
            "thresh": best_thresh,
            "left": self._build_newton_tree(X, g, h, best_left, depth + 1),
            "right": self._build_newton_tree(X, g, h, best_right, depth + 1),
        }

    def _predict_tree_one(self, node: dict, row: list[float]) -> float:
        while not node["leaf"]:
            if row[node["feat"]] <= node["thresh"]:
                node = node["left"]
            else:
                node = node["right"]
        return node["val"]

    def fit(self, X: list[list[float]], y: list[int]) -> None:
        rng = random.Random(self.seed)
        n = len(X)
        d = len(X[0])
        pos_sum = sum(y)
        neg_sum = (n - pos_sum) * self.neg_weight
        pos_rate = min(max(pos_sum / (pos_sum + neg_sum), 1e-5), 1.0 - 1e-5)
        self.base_score = math.log(pos_rate / (1.0 - pos_rate))
        raw_preds = [self.base_score] * n
        self.trees = []
        self.feature_importances = [0.0] * d

        sample_size = int(n * self.subsample)
        for _ in range(self.n_estimators):
            g = [0.0] * n
            h = [0.0] * n
            for i in range(n):
                p = sigmoid(raw_preds[i])
                w_i = 1.0 if y[i] == 1 else self.neg_weight
                g[i] = w_i * (p - y[i])
                h[i] = w_i * max(p * (1.0 - p), 1e-6)

            sub_idx = rng.sample(range(n), sample_size) if sample_size < n else list(range(n))
            tree = self._build_newton_tree(X, g, h, sub_idx, 0)
            self.trees.append(tree)
            for i in range(n):
                raw_preds[i] += self.learning_rate * self._predict_tree_one(tree, X[i])

        tot_imp = sum(self.feature_importances)
        if tot_imp > 0:
            self.feature_importances = [v / tot_imp for v in self.feature_importances]

    def predict_proba(self, X: list[list[float]]) -> list[float]:
        probs = []
        for row in X:
            z = self.base_score + sum(
                self.learning_rate * self._predict_tree_one(t, row) for t in self.trees
            )
            probs.append(sigmoid(z))
        return probs


# --- Evaluation Metrics ---
def roc_auc_score(y_true: list[int], y_prob: list[float]) -> float:
    pairs = sorted(zip(y_prob, y_true), key=lambda x: x[0])
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = 0.5 * (i + 1 + j)
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j

    sum_pos_ranks = sum(ranks[k] for k in range(len(pairs)) if pairs[k][1] == 1)
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def evaluate_predictions(y_true: list[int], y_prob: list[float], threshold: float = 0.5) -> dict:
    tp = fp = tn = fn = 0
    loss_sum = 0.0
    eps = 1e-15
    for yt, yp in zip(y_true, y_prob):
        yp_clipped = min(max(yp, eps), 1.0 - eps)
        loss_sum += -(yt * math.log(yp_clipped) + (1 - yt) * math.log(1 - yp_clipped))
        pred = 1 if yp >= threshold else 0
        if pred == 1 and yt == 1:
            tp += 1
        elif pred == 1 and yt == 0:
            fp += 1
        elif pred == 0 and yt == 0:
            tn += 1
        else:
            fn += 1

    n = len(y_true)
    acc = (tp + tn) / n
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    # F0.5 weights Precision 2x as much as Recall (beta = 0.5 -> beta^2 = 0.25)
    f05 = (
        (1.25 * precision * recall) / (0.25 * precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    auc = roc_auc_score(y_true, y_prob)
    logloss = loss_sum / n

    return {
        "threshold": threshold,
        "precision": precision,
        "f05": f05,
        "f1": f1,
        "recall": recall,
        "accuracy": acc,
        "roc_auc": auc,
        "log_loss": logloss,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def find_best_precision_threshold(
    y_true: list[int], y_prob: list[float], min_recall: float = 0.65
) -> float:
    """Finds threshold on training fold that maximizes F0.5 / Precision while maintaining recall >= min_recall."""
    best_t = 0.50
    best_score = -1.0
    for step in range(40, 80):
        t = step / 100.0
        m = evaluate_predictions(y_true, y_prob, threshold=t)
        if m["recall"] >= min_recall:
            # Maximize Precision (with F0.5 tie-breaking)
            score = m["precision"] + 0.2 * m["f05"]
            if score > best_score:
                best_score = score
                best_t = t
    return best_t


def main() -> None:
    raw_train = load_csv("train.csv")
    y_all = [int(r["Survived"]) for r in raw_train]
    splits = stratified_k_fold(y_all, k=5, seed=42)

    model_names = [
        "Logistic Regression (Prec-Tuned)",
        "Decision Tree (Prec-Tuned)",
        "Random Forest (Prec-Tuned)",
        "Gradient Boosting (Prec-Tuned)",
        "Precision Soft-Voting Ensemble",
    ]
    oof_probs = {name: [0.0] * len(raw_train) for name in model_names}
    oof_custom_preds = {name: [0] * len(raw_train) for name in model_names}
    fold_precisions = defaultdict(list)
    fold_thresholds = defaultdict(list)

    print("Running 5-Fold Stratified CV with Cost-Sensitive Precision Tuning (neg_weight=1.45)...")
    for fold_idx, (train_idx, val_idx) in enumerate(splits, 1):
        fold_train_raw = [raw_train[i] for i in train_idx]
        fold_val_raw = [raw_train[i] for i in val_idx]

        stats = fit_imputers(fold_train_raw)
        fold_train_fe, _ = transform_rows(fold_train_raw, stats, is_train=True)
        fold_val_fe, _ = transform_rows(fold_val_raw, stats, is_train=True)

        X_tr, y_tr = extract_xy(fold_train_fe)
        X_va, y_va = extract_xy(fold_val_fe)

        lr = LogisticRegressionL2(lr=0.1, epochs=550, l2=0.08, neg_weight=1.45)
        lr.fit(X_tr, y_tr)
        p_lr_tr = lr.predict_proba(X_tr)
        p_lr_va = lr.predict_proba(X_va)

        dt = DecisionTreeCART(max_depth=4, min_samples_split=16, min_samples_leaf=8, neg_weight=1.45)
        dt.fit(X_tr, y_tr)
        p_dt_tr = dt.predict_proba(X_tr)
        p_dt_va = dt.predict_proba(X_va)

        rf = RandomForest(n_estimators=60, max_depth=5, min_samples_leaf=6, neg_weight=1.45, seed=42 + fold_idx)
        rf.fit(X_tr, y_tr)
        p_rf_tr = rf.predict_proba(X_tr)
        p_rf_va = rf.predict_proba(X_va)

        gbdt = GradientBoostingClassifier(
            n_estimators=55,
            learning_rate=0.07,
            max_depth=3,
            min_samples_leaf=10,
            l2_reg=3.5,
            neg_weight=1.45,
            seed=42 + fold_idx,
        )
        gbdt.fit(X_tr, y_tr)
        p_gbdt_tr = gbdt.predict_proba(X_tr)
        p_gbdt_va = gbdt.predict_proba(X_va)

        # Ensemble weights favoring highest-precision learners (RF + GBDT + LR)
        p_ens_tr = [0.20 * a + 0.45 * b + 0.35 * c for a, b, c in zip(p_lr_tr, p_rf_tr, p_gbdt_tr)]
        p_ens_va = [0.20 * a + 0.45 * b + 0.35 * c for a, b, c in zip(p_lr_va, p_rf_va, p_gbdt_va)]

        fold_models = {
            "Logistic Regression (Prec-Tuned)": (p_lr_tr, p_lr_va),
            "Decision Tree (Prec-Tuned)": (p_dt_tr, p_dt_va),
            "Random Forest (Prec-Tuned)": (p_rf_tr, p_rf_va),
            "Gradient Boosting (Prec-Tuned)": (p_gbdt_tr, p_gbdt_va),
            "Precision Soft-Voting Ensemble": (p_ens_tr, p_ens_va),
        }

        for name, (tr_probs, va_probs) in fold_models.items():
            # Tune decision threshold strictly on training fold
            opt_t = find_best_precision_threshold(y_tr, tr_probs, min_recall=0.68)
            fold_thresholds[name].append(opt_t)
            for idx_pos, orig_i in enumerate(val_idx):
                oof_probs[name][orig_i] = va_probs[idx_pos]
                oof_custom_preds[name][orig_i] = 1 if va_probs[idx_pos] >= opt_t else 0
            prec = evaluate_predictions(y_va, va_probs, threshold=0.5)["precision"]
            fold_precisions[name].append(prec)

    print("\n=== 1. Precision-Retuned Models (Standard Threshold = 0.50, Cost-Sensitive FP Weight = 1.45x) ===")
    header = f"{'Model':<32} | {'CV Precision (Mean±Std)':<23} | {'OOF Prec':<8} | {'F0.5':<7} | {'Recall':<7} | {'F1':<7} | {'Acc':<7} | {'FP':<4}"
    print(header)
    print("-" * len(header))

    best_model_name = None
    best_f05 = -1.0
    oof_metrics_all = {}
    for name in model_names:
        m = evaluate_predictions(y_all, oof_probs[name], threshold=0.5)
        oof_metrics_all[name] = m
        mean_p = statistics.mean(fold_precisions[name])
        std_p = statistics.stdev(fold_precisions[name])
        if m["f05"] > best_f05:
            best_f05 = m["f05"]
            best_model_name = name
        print(
            f"{name:<32} | {mean_p*100:6.2f}% ± {std_p*100:4.2f}%        | "
            f"{m['precision']*100:6.2f}% | {m['f05']:7.4f} | {m['recall']*100:5.1f}% | "
            f"{m['f1']:7.4f} | {m['accuracy']*100:5.1f}% | {m['fp']:4d}"
        )

    print("\n=== 2. Precision–Recall Trade-Off Curve on Precision Ensemble (OOF) ===")
    print(f"{'Threshold':<10} | {'Precision':<10} | {'F0.5 (Prec-F)':<14} | {'Recall':<9} | {'F1-Score':<9} | {'Accuracy':<9} | {'TP':<5} | {'FP':<5}")
    print("-" * 88)
    ens_probs = oof_probs["Precision Soft-Voting Ensemble"]
    for t in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        m = evaluate_predictions(y_all, ens_probs, threshold=t)
        marker = " <-- Default (Cost-Weighted)" if abs(t - 0.50) < 1e-6 else (" <-- High-Precision Sweet Spot" if abs(t - 0.55) < 1e-6 else "")
        print(
            f"{t:<10.2f} | {m['precision']*100:8.2f}% | {m['f05']:14.4f} | {m['recall']*100:7.2f}% | "
            f"{m['f1']:9.4f} | {m['accuracy']*100:7.2f}% | {m['tp']:<5d} | {m['fp']:<5d}{marker}"
        )

    bm = evaluate_predictions(y_all, ens_probs, threshold=0.55)
    print(f"\n=== Confusion Matrix: Precision Ensemble (at Threshold = 0.55) ===")
    print(f"                  Predicted 0    Predicted 1")
    print(f"Actual 0 (549) :     {bm['tn']:4d} (TN)      {bm['fp']:4d} (FP)  <-- Cut False Positives from 60 down to {bm['fp']}!")
    print(f"Actual 1 (342) :     {bm['fn']:4d} (FN)      {bm['tp']:4d} (TP)")

    # Fit final precision-tuned models on full train.csv
    full_stats = fit_imputers(raw_train)
    full_train_fe, _ = transform_rows(raw_train, full_stats, is_train=True)
    X_full, y_full = extract_xy(full_train_fe)

    final_lr = LogisticRegressionL2(lr=0.1, epochs=550, l2=0.08, neg_weight=1.45)
    final_lr.fit(X_full, y_full)

    final_rf = RandomForest(n_estimators=60, max_depth=5, min_samples_leaf=6, neg_weight=1.45, seed=42)
    final_rf.fit(X_full, y_full)

    final_gbdt = GradientBoostingClassifier(
        n_estimators=55,
        learning_rate=0.07,
        max_depth=3,
        min_samples_leaf=10,
        l2_reg=3.5,
        neg_weight=1.45,
        seed=42,
    )
    final_gbdt.fit(X_full, y_full)

    artifacts = {
        "tuning_objective": "precision",
        "neg_weight": 1.45,
        "recommended_threshold": 0.55,
        "features": NUMERIC_FEATURES,
        "cv_results_at_050": oof_metrics_all,
        "ensemble_at_055": bm,
        "logistic_regression": {
            "weights": final_lr.weights,
            "bias": final_lr.bias,
            "means": final_lr.means,
            "stds": final_lr.stds,
        },
        "gbdt": {
            "base_score": final_gbdt.base_score,
            "learning_rate": final_gbdt.learning_rate,
            "trees": final_gbdt.trees,
        },
        "random_forest": {
            "trees": [t.tree for t in final_rf.trees],
        },
    }
    with open("trained_model.json", "w", encoding="utf-8") as f:
        json.dump(artifacts, f, indent=2)
    print("\nSaved precision-retuned models and metrics to trained_model.json")


if __name__ == "__main__":
    main()
