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

# Interpretation: squad must include players from at least 4 different AFL teams
MIN_DISTINCT_TEAMS = 4


# ---------- HELPER FUNCTIONS ----------

def parse_positions(pos_value):
    """
    Parse the Position string into a list of eligible positions.
    Supports DPP like 'DEF/MID'.
    """
    if pd.isna(pos_value):
        return []
    tokens = str(pos_value).upper().replace("\\", "/").split("/")
    positions = [t.strip() for t in tokens if t.strip()]
    # Keep only known positions
    return [p for p in positions if p in POSITIONS]


def load_player_data():
    input_path = RAW_DATA_DIR / INPUT_FILENAME

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_excel(input_path, sheet_name="Sheet1")

    required_cols = ["Player", "Team", "Position", "Projection", "Price"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input file: {missing}")

    # Filter to required columns only
    df = df[required_cols].copy()

    # Drop rows with critical missing values
    df = df.dropna(subset=["Player", "Team", "Position", "Projection", "Price"]).reset_index(drop=True)

    # Normalize numeric types
    df["Projection"] = pd.to_numeric(df["Projection"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df = df.dropna(subset=["Projection", "Price"]).reset_index(drop=True)

    # Parse positions for each player
    df["EligiblePositions"] = df["Position"].apply(parse_positions)
    df = df[df["EligiblePositions"].map(len) > 0].reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid players after cleaning and parsing positions.")

    return df


def build_and_solve_model(df):
    n_players = len(df)
    player_indices = range(n_players)

    # Mapping for teams
    teams = sorted(df["Team"].unique().tolist())

    # Decision variables
    # start[i,pos] = 1 if player i is selected as a starter in position pos
    # bench[i,pos] = 1 if player i is selected on bench in position pos
    start = {}
    bench = {}

    for i in player_indices:
        for pos in df.loc[i, "EligiblePositions"]:
            start[(i, pos)] = LpVariable(f"start_{i}_{pos}", cat=LpBinary)
            bench[(i, pos)] = LpVariable(f"bench_{i}_{pos}", cat=LpBinary)

    # x[i] = 1 if player i is in squad (starter or bench)
    x = {
        i: LpVariable(f"selected_{i}", cat=LpBinary)
        for i in player_indices
    }

    # captain[i] = 1 if player i is captain (must be a starter)
    captain = {
        i: LpVariable(f"captain_{i}", cat=LpBinary)
        for i in player_indices
    }

    # y_team[t] = 1 if at least one player from team t is selected
    y_team = {
        t: LpVariable(f"team_used_{t}", cat=LpBinary)
        for t in teams
    }

    # Problem
    prob = LpProblem("AFL_Fantasy_Optimal_Team_2026", LpMaximize)

    # ---------- OBJECTIVE ----------
    # Maximize sum of projections of starters + extra projection for captain
    projections = df["Projection"].tolist()
    prices = df["Price"].tolist()
    player_teams = df["Team"].tolist()

    # Starters contribute projection
    starter_points = lpSum(
        projections[i] * start[(i, pos)]
        for (i, pos) in start
    )

    # Captain gives an extra projection (so total 2x for that player)
    captain_bonus = lpSum(
        projections[i] * captain[i]
        for i in player_indices
    )

    prob += starter_points + captain_bonus, "Total_Projected_Points_Including_Captain"

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
        # Starters
        prob += (
            lpSum(start[(i, p)] for (i, p) in start if p == pos)
            == STARTING_COUNTS[pos],
            f"starters_{pos}"
        )

        # Bench
        prob += (
            lpSum(bench[(i, p)] for (i, p) in bench if p == pos)
            == BENCH_COUNTS[pos],
            f"bench_{pos}"
        )

    # 4. Total starters / bench
    prob += (
        lpSum(start.values()) == TOTAL_STARTERS,
        "total_starters"
    )
    prob += (
        lpSum(bench.values()) == TOTAL_BENCH,
        "total_bench"
    )

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
    prob += (
        lpSum(captain[i] for i in player_indices) == 1,
        "single_captain"
    )

    for i in player_indices:
        # Captain only if a starter in (any) position
        prob += (
            captain[i]
            <= lpSum(start[(i, pos)] for pos in df.loc[i, "EligiblePositions"]),
            f"captain_must_be_starter_{i}"
        )

    # 7. Distinct teams: at least MIN_DISTINCT_TEAMS teams represented
    for t in teams:
        # If any player from team t is selected, y_team[t] must be 1
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

    # ---------- SOLVE ----------
    solver = PULP_CBC_CMD(msg=True)
    result_status = prob.solve(solver)

    if result_status != LpStatusOptimal:
        raise RuntimeError(f"Optimization did not find an optimal solution. Status code: {result_status}")

    return prob, start, bench, captain


def extract_solution(df, start, bench, captain):
    rows = []

    total_cost = 0.0
    base_points = 0.0
    captain_bonus_points = 0.0

    for (i, pos) in start:
        if start[(i, pos)].value() == 1:
            is_captain = captain[i].value() == 1

            rows.append({
                "Player": df.loc[i, "Player"],
                "Team": df.loc[i, "Team"],
                "FantasyPosition": pos,
                "Role": "Starter",
                "Captain": "Yes" if is_captain else "No",
                "Projection": df.loc[i, "Projection"],
                "Price": df.loc[i, "Price"],
            })

            total_cost += df.loc[i, "Price"]
            base_points += df.loc[i, "Projection"]
            if is_captain:
                captain_bonus_points += df.loc[i, "Projection"]

    for (i, pos) in bench:
        if bench[(i, pos)].value() == 1:
            # Bench players do not score points
            rows.append({
                "Player": df.loc[i, "Player"],
                "Team": df.loc[i, "Team"],
                "FantasyPosition": pos,
                "Role": "Bench",
                "Captain": "No",
                "Projection": df.loc[i, "Projection"],
                "Price": df.loc[i, "Price"],
            })
            total_cost += df.loc[i, "Price"]

    squad_df = pd.DataFrame(rows)

    total_points = base_points + captain_bonus_points

    return squad_df, total_cost, total_points


def save_output(squad_df, total_cost, total_points):
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_DIR / OUTPUT_FILENAME

    # Add summary rows at the bottom in a separate sheet
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        squad_df.to_excel(writer, sheet_name="Squad", index=False)

        summary_df = pd.DataFrame(
            {
                "Metric": ["Total Cost", "Total Projected Points (incl. C)"],
                "Value": [total_cost, total_points],
            }
        )
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

    print(f"Optimal team saved to: {output_path}")
    print(f"Total Cost: {total_cost:,.0f}")
    print(f"Total Projected Points (incl. captain): {total_points:,.2f}")


def main():
    df = load_player_data()
    prob, start, bench, captain = build_and_solve_model(df)
    squad_df, total_cost, total_points = extract_solution(df, start, bench, captain)
    save_output(squad_df, total_cost, total_points)


if __name__ == "__main__":
    main()
