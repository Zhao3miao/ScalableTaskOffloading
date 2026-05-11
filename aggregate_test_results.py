import glob
import os
import re

import numpy as np
import pandas as pd
from scipy import stats


ACTION_LABELS = {
    "action_0_ratio": "Local",
    "action_1_ratio": "BS1",
    "action_2_ratio": "BS2",
    "action_3_ratio": "BS3",
}

SCENARIO_FAMILY_ORDER = {
    "campus": 0,
    "random_walk": 1,
    "subway_station": 2,
    "unknown": 99,
}

FAMILY_LABELS = {
    "campus": "Campus Scenarios",
    "random_walk": "Random Walk Scenarios",
    "subway_station": "Subway Station Scenarios",
    "unknown": "Unknown Scenarios",
}

TEST_TYPE_LABELS = {
    "in-dist": "in-dist",
    "cross-UE": "cross-UE",
    "geo-OOD": "geo-OOD",
    "geo-OOD+scale": "geo-OOD+scale",
}


def load_all_results(base_dirs):
    """
    Load all test_results.csv files under:
        {base_dir}/{method}/{model_name}/test_results.csv
    """
    all_data = []

    for base_dir in base_dirs:
        search_pattern = os.path.join(base_dir, "*", "*", "test_results.csv")
        csv_files = glob.glob(search_pattern)

        for file_path in csv_files:
            path_parts = os.path.normpath(file_path).split(os.sep)
            model_name = path_parts[-2]
            method_name = path_parts[-3]
            seed_dir = path_parts[-4]

            seed_label = seed_dir.split("seed_")[-1] if "seed_" in seed_dir else seed_dir
            train_tag = model_name.replace(f"{method_name}_", "", 1)

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    first_line = f.readline().strip()

                if not first_line:
                    continue

                if "scenario" not in first_line:
                    col_count = len(first_line.split(","))
                    names = [
                        "scenario",
                        "avg_reward",
                        "avg_tasks",
                        "avg_cost",
                        "avg_drop_ratio",
                    ]
                    for i in range(col_count - 5):
                        names.append(f"action_{i}_ratio")
                    df = pd.read_csv(file_path, names=names)
                else:
                    df = pd.read_csv(file_path)

                df.insert(0, "seed", seed_label)
                df.insert(1, "method", method_name)
                df.insert(2, "train_scenario", train_tag)
                all_data.append(df)
                print(f"  Loaded: {file_path} ({len(df)} rows)")
            except Exception as exc:
                print(f"  Error: {file_path}: {exc}")

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


def fmt_ci(mean, ci_half):
    if np.isnan(ci_half):
        return f"{mean:.2f}"
    return f"{mean:.2f}±{ci_half:.2f}"


def fmt_action_bar(means, action_cols):
    parts = []
    for col in action_cols:
        label = ACTION_LABELS.get(col, col)
        parts.append(f"{label}:{means[col]:.0f}")
    return " | ".join(parts)


def parse_scenario_info(name):
    scenario = str(name).lower()

    if "subway" in scenario:
        family = "subway_station"
    elif "random_walk" in scenario or scenario.startswith("rw"):
        family = "random_walk"
    elif "campus" in scenario or "osm" in scenario:
        family = "campus"
    else:
        family = "unknown"

    match = re.search(r"max_ue_(\d+)", scenario)
    if match:
        ue = int(match.group(1))
    else:
        digits = re.findall(r"\d+", scenario)
        ue = int(digits[-1]) if digits else np.nan

    return {"family": family, "ue": ue}


def classify_test_type(train_scenario, test_scenario):
    train_info = parse_scenario_info(train_scenario)
    test_info = parse_scenario_info(test_scenario)

    same_family = train_info["family"] == test_info["family"]
    same_ue = train_info["ue"] == test_info["ue"]

    if same_family and same_ue:
        return "in-dist"
    if same_family and not same_ue:
        return "cross-UE"
    if not same_family and same_ue:
        return "geo-OOD"
    return "geo-OOD+scale"


def scenario_sort_key(family, ue, scenario):
    return (
        SCENARIO_FAMILY_ORDER.get(family, 99),
        ue if not pd.isna(ue) else 10**9,
        scenario,
    )


def main():
    base_dirs = sorted(glob.glob("experiment_seed_*"))
    if not base_dirs:
        base_dirs = sorted(glob.glob("experiment[0-9]*"))
    if not base_dirs:
        base_dirs = ["experiment"]
    print(f"Found experiment directories: {base_dirs}")

    raw_df = load_all_results(base_dirs)
    if raw_df.empty:
        print("No test results found.")
        return

    n_seeds = raw_df["seed"].nunique()
    print(f"\nTotal records: {len(raw_df)}, Seeds: {n_seeds}")

    action_cols = sorted(
        [c for c in raw_df.columns if c.startswith("action_") and c.endswith("_ratio")]
    )
    metric_cols = ["avg_reward", "avg_tasks", "avg_cost", "avg_drop_ratio"]

    raw_df["test_type"] = raw_df.apply(
        lambda row: classify_test_type(row["train_scenario"], row["scenario"]), axis=1
    )
    raw_df["scenario_family"] = raw_df["scenario"].apply(
        lambda s: parse_scenario_info(s)["family"]
    )
    raw_df["scenario_ue"] = raw_df["scenario"].apply(lambda s: parse_scenario_info(s)["ue"])
    raw_df["train_family"] = raw_df["train_scenario"].apply(
        lambda s: parse_scenario_info(s)["family"]
    )
    raw_df["train_ue"] = raw_df["train_scenario"].apply(
        lambda s: parse_scenario_info(s)["ue"]
    )

    grouped = raw_df.groupby(["method", "train_scenario", "scenario"])

    rows = []
    for (method, train_scenario, scenario), grp in grouped:
        train_info = parse_scenario_info(train_scenario)
        scenario_info = parse_scenario_info(scenario)
        row = {
            "scenario_family": scenario_info["family"],
            "scenario_ue": scenario_info["ue"],
            "scenario": scenario,
            "method": method,
            "train_family": train_info["family"],
            "train_ue": train_info["ue"],
            "train_scenario": train_scenario,
            "test_type": classify_test_type(train_scenario, scenario),
            "n_seeds": len(grp),
        }

        for col in metric_cols:
            vals = grp[col].dropna().values
            mean = np.mean(vals)
            if len(vals) >= 2:
                sem = stats.sem(vals)
                ci = sem * stats.t.ppf(0.975, df=len(vals) - 1)
            else:
                ci = np.nan
            row[f"{col}_mean"] = round(mean, 2)
            row[f"{col}_ci"] = round(ci, 2) if not np.isnan(ci) else np.nan
            row[col] = fmt_ci(mean, ci)

        action_means = {}
        for col in action_cols:
            vals = grp[col].dropna().values
            action_means[col] = float(np.mean(vals))
            row[f"{col}_mean"] = round(action_means[col], 1)

        probs = np.array(list(action_means.values())) / 100.0
        probs = probs[probs > 0]
        row["action_entropy"] = round(-np.sum(probs * np.log2(probs)), 3)

        dominant_col = max(action_cols, key=lambda c: action_means[c])
        row["dominant_action"] = ACTION_LABELS.get(dominant_col, dominant_col)
        row["dominant_pct"] = round(action_means[dominant_col], 1)
        row["action_dist"] = fmt_action_bar(action_means, action_cols)
        rows.append(row)

    result_df = pd.DataFrame(rows)
    result_df["scenario_sort_order"] = result_df.apply(
        lambda row: scenario_sort_key(
            row["scenario_family"], row["scenario_ue"], row["scenario"]
        ),
        axis=1,
    )
    result_df["train_sort_order"] = result_df.apply(
        lambda row: scenario_sort_key(
            row["train_family"], row["train_ue"], row["train_scenario"]
        ),
        axis=1,
    )
    result_df = result_df.sort_values(
        by=["scenario_sort_order", "scenario", "method", "train_sort_order", "train_scenario"]
    ).drop(columns=["scenario_sort_order", "train_sort_order"])

    numeric_cols = (
        [
            "scenario_family",
            "scenario_ue",
            "scenario",
            "method",
            "train_family",
            "train_ue",
            "train_scenario",
            "test_type",
            "n_seeds",
        ]
        + [f"{c}_mean" for c in metric_cols]
        + [f"{c}_ci" for c in metric_cols]
        + [f"{c}_mean" for c in action_cols]
        + ["action_entropy", "dominant_action", "dominant_pct"]
    )
    result_df[numeric_cols].to_csv("aggregated_test_results.csv", index=False)
    print("\nSaved: aggregated_test_results.csv (numeric, grouped by scenario)")

    display_cols = (
        [
            "scenario_family",
            "scenario_ue",
            "scenario",
            "method",
            "train_family",
            "train_ue",
            "train_scenario",
            "test_type",
            "n_seeds",
        ]
        + metric_cols
        + ["action_dist", "action_entropy", "dominant_action", "dominant_pct"]
    )
    result_df[display_cols].to_csv("aggregated_test_display.csv", index=False)
    print("Saved: aggregated_test_display.csv (display, grouped by scenario)")

    print("\n" + "=" * 130)
    print("AGGREGATED RESULTS (grouped by scenario)")
    print("=" * 130)

    families = sorted(
        result_df["scenario_family"].unique(),
        key=lambda family: SCENARIO_FAMILY_ORDER.get(family, 99),
    )
    for family in families:
        subset_family = result_df[result_df["scenario_family"] == family]
        if subset_family.empty:
            continue

        print(f"\n{'#' * 130}")
        print(f"  {FAMILY_LABELS.get(family, family)}")
        print(f"{'#' * 130}")

        scenario_rows = (
            subset_family[["scenario", "scenario_ue"]].drop_duplicates().values.tolist()
        )
        scenario_rows = sorted(
            scenario_rows,
            key=lambda item: scenario_sort_key(family, item[1], item[0]),
        )

        for scenario, _ in scenario_rows:
            subset = subset_family[subset_family["scenario"] == scenario].sort_values(
                by=["avg_reward_mean", "method"], ascending=[False, True]
            )
            n = subset["n_seeds"].iloc[0]
            test_types = ", ".join(
                subset["test_type"]
                .drop_duplicates()
                .map(lambda t: TEST_TYPE_LABELS.get(t, t))
                .tolist()
            )

            print(f"\n  Test: {scenario}  (n_seeds={n})")
            print(f"  Type: {test_types}")
            print(
                f"  {'Method':<14} {'Train':>14} {'Reward':>18} {'Cost':>16} "
                f"{'Drop%':>16} {'Entropy':>8}  {'Action Distribution'}"
            )
            print(f"  {'-' * 126}")

            best_reward = subset["avg_reward_mean"].max()
            for _, record in subset.iterrows():
                best = "*" if record["avg_reward_mean"] == best_reward else " "
                print(
                    f"{best} {record['method']:<12} {record['train_scenario']:>14} "
                    f"{record['avg_reward']:>18} {record['avg_cost']:>16} "
                    f"{record['avg_drop_ratio']:>16} {record['action_entropy']:>8.3f}  "
                    f"{record['action_dist']}"
                )

    print(f"\n{'=' * 130}")


if __name__ == "__main__":
    main()
