import csv
import math
import re
import statistics
from collections import Counter, defaultdict

TITLE_MAP = {
    "Mr": "Mr",
    "Miss": "Miss",
    "Mlle": "Miss",
    "Ms": "Miss",
    "Mrs": "Mrs",
    "Mme": "Mrs",
    "Master": "Master",
}


def extract_title(name: str) -> str:
    match = re.search(r",\s*([^\.]+)\.", name)
    if not match:
        return "Rare"
    raw_title = match.group(1).strip()
    return TITLE_MAP.get(raw_title, "Rare")


def extract_deck(cabin: str) -> str:
    cabin = (cabin or "").strip()
    if not cabin:
        return "U"
    deck = cabin[0].upper()
    return deck if deck in {"A", "B", "C", "D", "E", "F", "G"} else "U"


def family_category(size: int) -> str:
    if size == 1:
        return "Alone"
    if 2 <= size <= 4:
        return "Small"
    return "Large"


def age_band(age: float) -> str:
    if age < 12:
        return "Child"
    if age < 18:
        return "Teen"
    if age < 35:
        return "YoungAdult"
    if age < 55:
        return "Adult"
    return "Senior"


def load_csv(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fit_imputers(train_rows: list[dict]) -> dict:
    embarked_counts = Counter(
        r["Embarked"] for r in train_rows if r.get("Embarked")
    )
    embarked_mode = embarked_counts.most_common(1)[0][0]

    fare_by_pclass = defaultdict(list)
    all_fares = []
    for r in train_rows:
        if r.get("Fare") != "":
            val = float(r["Fare"])
            fare_by_pclass[int(r["Pclass"])].append(val)
            all_fares.append(val)
    fare_medians = {
        k: statistics.median(v) for k, v in fare_by_pclass.items()
    }
    global_fare_median = statistics.median(all_fares)

    age_by_title_pclass = defaultdict(list)
    age_by_title = defaultdict(list)
    all_ages = []
    for r in train_rows:
        if r.get("Age") != "":
            age_val = float(r["Age"])
            title = extract_title(r["Name"])
            pclass = int(r["Pclass"])
            age_by_title_pclass[(title, pclass)].append(age_val)
            age_by_title[title].append(age_val)
            all_ages.append(age_val)

    age_medians_tp = {
        k: statistics.median(v) for k, v in age_by_title_pclass.items()
    }
    age_medians_t = {k: statistics.median(v) for k, v in age_by_title.items()}
    global_age_median = statistics.median(all_ages)

    ticket_counts = Counter(r["Ticket"].strip() for r in train_rows)

    return {
        "embarked_mode": embarked_mode,
        "fare_medians": fare_medians,
        "global_fare_median": global_fare_median,
        "age_medians_tp": age_medians_tp,
        "age_medians_t": age_medians_t,
        "global_age_median": global_age_median,
        "ticket_counts": ticket_counts,
    }


def transform_rows(
    rows: list[dict], stats: dict, is_train: bool = True
) -> tuple[list[dict], list[str]]:
    transformed = []
    for r in rows:
        pclass = int(r["Pclass"])
        sex_num = 1 if r["Sex"].strip().lower() == "female" else 0
        sibsp = int(r["SibSp"])
        parch = int(r["Parch"])

        title = extract_title(r["Name"])

        embarked = r["Embarked"].strip() if r.get("Embarked") else ""
        if not embarked:
            embarked = stats["embarked_mode"]

        if r.get("Fare") != "" and r.get("Fare") is not None:
            fare = float(r["Fare"])
        else:
            fare = stats["fare_medians"].get(pclass, stats["global_fare_median"])

        age_missing = 1 if (r.get("Age") == "" or r.get("Age") is None) else 0
        if not age_missing:
            age = float(r["Age"])
        else:
            age = stats["age_medians_tp"].get(
                (title, pclass),
                stats["age_medians_t"].get(title, stats["global_age_median"]),
            )

        family_size = sibsp + parch + 1
        is_alone = 1 if family_size == 1 else 0
        fam_cat = family_category(family_size)

        ticket = r["Ticket"].strip()
        ticket_group_size = stats["ticket_counts"].get(ticket, 1)
        fare_per_person = fare / max(ticket_group_size, 1)
        log_fare = math.log1p(fare)

        has_cabin = 0 if (r.get("Cabin") == "" or r.get("Cabin") is None) else 1
        deck = extract_deck(r.get("Cabin", ""))

        is_child = 1 if age < 16.0 else 0
        band = age_band(age)

        out = {
            "PassengerId": int(r["PassengerId"]),
        }
        if is_train:
            out["Survived"] = int(r["Survived"])

        out.update(
            {
                "Pclass": pclass,
                "Sex": sex_num,
                "Age": round(age, 2),
                "AgeMissing": age_missing,
                "IsChild": is_child,
                "AgeBand": band,
                "SibSp": sibsp,
                "Parch": parch,
                "FamilySize": family_size,
                "IsAlone": is_alone,
                "FamilyCategory": fam_cat,
                "TicketGroupSize": ticket_group_size,
                "Fare": round(fare, 4),
                "FarePerPerson": round(fare_per_person, 4),
                "LogFare": round(log_fare, 4),
                "HasCabin": has_cabin,
                "Deck": deck,
                "Embarked": embarked,
                "Title": title,
                "Title_Mr": 1 if title == "Mr" else 0,
                "Title_Miss": 1 if title == "Miss" else 0,
                "Title_Mrs": 1 if title == "Mrs" else 0,
                "Title_Master": 1 if title == "Master" else 0,
                "Title_Rare": 1 if title == "Rare" else 0,
                "Embarked_S": 1 if embarked == "S" else 0,
                "Embarked_C": 1 if embarked == "C" else 0,
                "Embarked_Q": 1 if embarked == "Q" else 0,
                "FamCat_Alone": 1 if fam_cat == "Alone" else 0,
                "FamCat_Small": 1 if fam_cat == "Small" else 0,
                "FamCat_Large": 1 if fam_cat == "Large" else 0,
            }
        )
        transformed.append(out)

    fieldnames = list(transformed[0].keys())
    return transformed, fieldnames


def main() -> None:
    train_raw = load_csv("train.csv")
    stats = fit_imputers(train_raw)
    train_fe, train_cols = transform_rows(train_raw, stats, is_train=True)
    save_csv("train_features.csv", train_fe, train_cols)
    print(f"Saved train_features.csv ({len(train_fe)} rows, {len(train_cols)} cols) using train.csv only.")


if __name__ == "__main__":
    main()
