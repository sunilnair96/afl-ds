import os
import pandas as pd
from pathlib import Path

from pulp import (
    LpProblem,
    LpVariable,
    LpMaximize,
    LpBinary,
    lpSum,
    PULP_CBC_CMD,
    LpStatusOptimal,
)

# ---------- CONFIGURATION ----------

PROJECT_ROOT = Path(r"C:\Users\sunil\Projects\2026ws\afl-ds")
SRC_DIR = PROJECT_ROOT / "src"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

INPUT_FILENAME = "2026AflFantasyShortList.xlsx"
OUTPUT_FILENAME = "2026AflFantasyTeam.xlsx"

SALARY_CAP = 17_800_000

# Squad structure
POSITIONS = ["DEF", "MID", "RUC", "FWD"]

STARTING_COUNTS = {
    "DEF": 6,
    "MID": 8,
    "RUC": 2,
    "FWD": 6,
}

BENCH_COUNTS = {
    "DEF": 2,
    "MID": 3,
    "RUC": 1,
    "FWD": 2,
}

TOTAL_STARTERS = sum(STARTING_COUNTS.values())  # 22
TOTAL_BENCH = sum(BENCH_COUNTS.values())        # 8
TOTAL_SQUAD = TOTAL_STARTERS + TOTAL_BENCH      # 30

# At least 4 different AFL teams represented
MIN_DISTINCT_TEAMS = 4


# ---------- HELPER FUNCTIONS ----------

def parse_positions(pos_value):
    """
    Parse the Position string into a list of eligible positions.
    Supports DPP like 'DEF/MID' and normalises 'RUCK' -> 'RUC'.
    """
    if pd.isna(pos_value):
        return []
    tokens = str(pos_value).upper().replace("\\", "/").split("/")
    cleaned = []
    for t in tokens:
        t = t.strip()
        if not t:
            continue
        if t == "RUCK":
            t = "RUC"
        cleaned.append(t)
    return [p for p in cleaned if p in POSITIONS]


def normalize_preference(value):
    """
    Normalise Preference column.
    - 'Lock'  -> 'LOCK'  (must be a starter)
    - 'Bench' -> 'BENCH' (must be on bench)
    Anything else (including blank/NaN) -> ''
    """
    if pd.isna(value):
        return ""
    s = str(value).strip().lower()
    if s == "lock":
        return "LOCK"
    if s == "bench":
        return "BENCH"
    return ""


def load_player_data():
    input_path = RAW_DATA_DIR / INPUT_FILENAME

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_excel(input_path, sheet_name="Sheet1")

    # Required base columns
    base_required_cols = ["Player", "Team", "Position", "Projection", "Price"]
    missing = [c for c in base_required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input file: {missing}")

    # If Preference column not present, create empty
    if "Preference" not in df.columns:
        df["Preference"] = ""

    # Keep only what we need plus Preference
    df = df[base_required_cols + ["Preference"]].copy()

    # Drop rows with critical missing values (do NOT include Preference here)
    df = df.dropna(subset=["Player", "Team", "Position", "Projection", "Price"]).reset_index(drop=True)

    # Normalize numeric types
    df["Projection"] = pd.to_numeric(df["Projection"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df = df.dropna(subset=["Projection", "Price"]).reset_index(drop=True)

    # Parse positions for each player (includes RUCK -> RUC)
    df["EligiblePositions"] = df["Position"].apply(parse_positions)
    df = df[df["EligiblePositions"].map(len) > 0].reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid players after cleaning and parsing positions.")

    # Normalised preference
    df["PreferenceNorm"] = df["Preference"].apply(normalize_preference)

    # ----- NEW: DPP bonus -----
    # A player is DPP if they have more than one eligible position
    df["IsDPP"] = df["EligiblePositions"].map(lambda lst: len(lst) > 1)
    # Adjusted projection = original + 2 (for DPP), else same
    df["AdjProjection"] = df["Projection"] + df["IsDPP"].astype(int) * 2
    # --------------------------

    return df


def build_and_solve_model(df):
    n_players = len(df)
    player_indices = range(n_players)

    teams = sorted(df["Team"].unique().tolist())

    # Decision variables
    start = {}
    bench = {}

    for i in player_indices:
        for pos in df.loc[i, "EligiblePositions"]:
            start[(i, pos)] = LpVariable(f"start_{i}_{pos}", cat=LpBinary)
            bench[(i, pos)] = LpVariable(f"bench_{i}_{pos}", cat=LpBinary)

    # x[i] = 1 if player i is in squad (starter or bench)
    x = {i: LpVariable(f"selected_{i}", cat=LpBinary) for i in player_indices}

    # captain[i] = 1 if player i is captain (must be a starter)
    captain = {i: LpVariable(f"captain_{i}", cat=LpBinary) for i in player_indices}

    # y_team[t] = 1 if at least one player from team t is selected
    y_team = {t: LpVariable(f"team_used_{t}", cat=LpBinary) for t in teams}

    # Problem
    prob = LpProblem("AFL_Fantasy_Optimal_Team_2026", LpMaximize)

    # ---------- OBJECTIVE ----------
    # Use adjusted projection (Projection + 2 for DPP) for optimisation
    projections = df["AdjProjection"].tolist()
    prices = df["Price"].tolist()
    player_teams = df["Team"].tolist()

    starter_points = lpSum(
        projections[i] * start[(i, pos)]
        for (i, pos) in start
    )

    # Captain gives an extra projection (so total 2x for that player)
    captain_bonus = lpSum(
        projections[i] * captain[i]
        for i in player_indices
    )

    prob += starter_points + captain_bonus, "Total_AdjProjected_Points_Including_Captain"

    # ---------- CONSTRAINTS ----------

    # 1. Link x[i] with start/bench: a player can be at most one role/position
    for i in player_indices:
        prob += (
            lpSum(start[(i, pos)] for pos in df.loc[i, "EligiblePositions"])
            + lpSum(bench[(i, pos)] for pos in df.loc[i, "EligiblePositions"])
            <= x[i],
            f"role_assignment_for_player_{i}"
        )

    # 2. Squad size
    prob += lpSum(x[i] for i in player_indices) == TOTAL_SQUAD, "Total_squad_size"

    # 3. Starters and bench counts by position
    for pos in POSITIONS:
        prob += (
            lpSum(start[(i, p)] for (i, p) in start if p == pos)
            == STARTING_COUNTS[pos],
            f"starters_{pos}"
        )

        prob += (
            lpSum(bench[(i, p)] for (i, p) in bench if p == pos)
            == BENCH_COUNTS[pos],
            f"bench_{pos}"
        )

    # 4. Total starters / bench
    prob += lpSum(start.values()) == TOTAL_STARTERS, "total_starters"
    prob += lpSum(bench.values()) == TOTAL_BENCH, "total_bench"

    # 5. Salary cap
    prob += (
        lpSum(
            prices[i]
            * (
                lpSum(start[(i, pos)] for pos in df.loc[i, "EligiblePositions"]) +
                lpSum(bench[(i, pos)] for pos in df.loc[i, "EligiblePositions"])
            )
            for i in player_indices
        )
        <= SALARY_CAP,
        "salary_cap"
    )

    # 6. Captain: exactly one, must be a starter
    prob += lpSum(captain[i] for i in player_indices) == 1, "single_captain"

    for i in player_indices:
        prob += (
            captain[i]
            <= lpSum(start[(i, pos)] for pos in df.loc[i, "EligiblePositions"]),
            f"captain_must_be_starter_{i}"
        )

    # 7. Distinct teams: at least MIN_DISTINCT_TEAMS teams represented
    for t in teams:
        prob += (
            lpSum(
                x[i]
                for i in player_indices
                if player_teams[i] == t
            )
            >= y_team[t],
            f"team_used_link_{t}"
        )

    prob += (
        lpSum(y_team[t] for t in teams) >= MIN_DISTINCT_TEAMS,
        "min_distinct_teams"
    )

    # 8. Preference constraints: LOCK / BENCH
    for i in player_indices:
        pref = df.loc[i, "PreferenceNorm"]
        sum_start_i = lpSum(start[(i, pos)] for pos in df.loc[i, "EligiblePositions"])
        sum_bench_i = lpSum(bench[(i, pos)] for pos in df.loc[i, "EligiblePositions"])

        if pref == "LOCK":
            # Must be selected, must be a starter (exactly one position), cannot be on bench
            prob += x[i] == 1, f"lock_selected_{i}"
            prob += sum_start_i == 1, f"lock_is_starter_{i}"
            prob += sum_bench_i == 0, f"lock_not_bench_{i}"

        elif pref == "BENCH":
            # Must be selected, must be on bench (exactly one position), cannot be a starter
            prob += x[i] == 1, f"bench_pref_selected_{i}"
            prob += sum_bench_i == 1, f"bench_pref_is_bench_{i}"
            prob += sum_start_i == 0, f"bench_pref_not_starter_{i}"

    # ---------- SOLVE ----------
    solver = PULP_CBC_CMD(msg=True)
    result_status = prob.solve(solver)

    if result_status != LpStatusOptimal:
        raise RuntimeError(f"Optimization did not find an optimal solution. Status code: {result_status}")

    return prob, start, bench, captain


def extract_solution(df, start, bench, captain):
    rows = []

    total_cost = 0.0
    base_points = 0.0           # adjusted projection for starters
    captain_bonus_points = 0.0  # extra adj projection for captain

    for (i, pos) in start:
        if start[(i, pos)].value() == 1:
            is_captain = captain[i].value() == 1

            rows.append({
                "Player": df.loc[i, "Player"],
                "Team": df.loc[i, "Team"],
                "FantasyPosition": pos,
                "Role": "Starter",
                "Captain": "Yes" if is_captain else "No",
                "Projection": df.loc[i, "Projection"],          # original
                "AdjProjection": df.loc[i, "AdjProjection"],    # with DPP bonus
                "Price": df.loc[i, "Price"],
                "IsDPP": bool(df.loc[i, "IsDPP"]),
                "Preference": df.loc[i, "PreferenceNorm"],
            })

            total_cost += df.loc[i, "Price"]
            base_points += df.loc[i, "AdjProjection"]
            if is_captain:
                captain_bonus_points += df.loc[i, "AdjProjection"]

    for (i, pos) in bench:
        if bench[(i, pos)].value() == 1:
            rows.append({
                "Player": df.loc[i, "Player"],
                "Team": df.loc[i, "Team"],
                "FantasyPosition": pos,
                "Role": "Bench",
                "Captain": "No",
                "Projection": df.loc[i, "Projection"],
                "AdjProjection": df.loc[i, "AdjProjection"],
                "Price": df.loc[i, "Price"],
                "IsDPP": bool(df.loc[i, "IsDPP"]),
                "Preference": df.loc[i, "PreferenceNorm"],
            })
            total_cost += df.loc[i, "Price"]

    squad_df = pd.DataFrame(rows)
    total_adj_points = base_points + captain_bonus_points

    # ---------- SORTING ----------
    # Sort by position (DEF, MID, RUC, FWD), then Starter before Bench, then Price descending
    position_order = {"DEF": 1, "MID": 2, "RUC": 3, "FWD": 4}
    role_order = {"Starter": 0, "Bench": 1}

    squad_df["PosSort"] = squad_df["FantasyPosition"].map(position_order).fillna(99)
    squad_df["RoleSort"] = squad_df["Role"].map(role_order).fillna(9)

    squad_df = squad_df.sort_values(
        ["PosSort", "RoleSort", "Price"],
        ascending=[True, True, False]
    ).reset_index(drop=True)

    squad_df = squad_df.drop(columns=["PosSort", "RoleSort"])

    return squad_df, total_cost, total_adj_points


def save_output(squad_df, total_cost, total_adj_points):
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_DIR / OUTPUT_FILENAME

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        squad_df.to_excel(writer, sheet_name="Squad", index=False)

        summary_df = pd.DataFrame(
            {
                "Metric": [
                    "Total Cost",
                    "Total Adjusted Projected Points (incl. Captain)",
                    "Squad Size",
                ],
                "Value": [
                    total_cost,
                    total_adj_points,
                    len(squad_df),
                ],
            }
        )
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

    print(f"Optimal team saved to: {output_path}")
    print(f"Total Cost: {total_cost:,.0f}")
    print(f"Total Adjusted Projected Points (incl. captain): {total_adj_points:,.2f}")


def main():
    df = load_player_data()
    prob, start, bench, captain = build_and_solve_model(df)
    squad_df, total_cost, total_adj_points = extract_solution(df, start, bench, captain)
    save_output(squad_df, total_cost, total_adj_points)


if __name__ == "__main__":
    main()
