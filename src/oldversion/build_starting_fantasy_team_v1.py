import os
import pandas as pd
from pathlib import Path
from datetime import datetime

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
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

INPUT_FILENAME = "afl_fantasy_206_price_projection.xlsx"
OUTPUT_FILENAME = "afl_fantasy_2026_starting_team.xlsx"

SALARY_CAP = 17_800_000

# Squad structure (same as old program)
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

# At least N different AFL teams represented
MIN_DISTINCT_TEAMS = 4

# Weight for cash generation term in objective
# (tune this: higher = more aggressive cash focus)
CASH_WEIGHT = 1e-4


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

def load_player_data():
    """
    Load players from afl_fantasy_206_price_projection.xlsx

    Expected columns (at minimum):
      - Player
      - Team
      - Position            (e.g. 'DEF', 'MID', 'DEF/MID', 'RUCK')
      - DOB                 (dd/mm/yyyy, string or date)
      - StartingPrice
      - Price R4
      - AVG-2025, GMS-2025
      - AVG-2024, GMS-2024
      - AVG-2023, GMS-2023
      - round 1, round 2, round 3, round 4   (projected scores, all lower case)
    """
    input_path = PROCESSED_DATA_DIR / INPUT_FILENAME

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_excel(input_path)

    required_cols = [
        "Player",
        "Team",
        "Position",
        "DOB",
        "StartingPrice",
        "Price R4",
        "AVG-2025",
        "GMS-2025",
        "AVG-2024",
        "GMS-2024",
        "AVG-2023",
        "GMS-2023",
        "round 1",
        "round 2",
        "round 3",
        "round 4",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input file: {missing}")

    df = df[required_cols].copy()

    # --- DOB handling & age filters ---------------------------------
    df["DOB"] = pd.to_datetime(df["DOB"], format="%d/%m/%Y", dayfirst=True, errors="coerce")
    df = df.dropna(subset=["DOB"]).reset_index(drop=True)

    # Exclude players born before cutoff date (older vets)
    cutoff_date = datetime(1993, 11, 1)  # 01/11/1993
    df = df[df["DOB"] >= cutoff_date].reset_index(drop=True)

    # Age >30 as of season reference date → -2 per round
    season_ref_date = datetime(2026, 1, 1)
    df["AgeYears"] = (season_ref_date - df["DOB"]).dt.days / 365.25
    df["IsOver30"] = df["AgeYears"] > 30.0

    # Rising star: born after risingStarDate, with enough games
    rising_star_date = datetime(2002, 11, 1)  # interpreting 01/11/2022 as 2002
    df["IsRisingStarDOB"] = df["DOB"] >= rising_star_date

    # --- Numeric conversions ----------------------------------------
    num_cols = [
        "StartingPrice",
        "Price R4",
        "AVG-2025",
        "GMS-2025",
        "AVG-2024",
        "GMS-2024",
        "AVG-2023",
        "GMS-2023",
        "round 1",
        "round 2",
        "round 3",
        "round 4",
    ]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=["StartingPrice", "Price R4", "round 1", "round 2", "round 3", "round 4"]
    ).reset_index(drop=True)

    # Eligible positions (and DPP check)
    df["EligiblePositions"] = df["Position"].apply(parse_positions)
    df = df[df["EligiblePositions"].map(len) > 0].reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid players after cleaning and parsing positions.")

    # --- Base projected scores R1–R4 (before age/trend tweaks) ------
    base_round_cols = ["round 1", "round 2", "round 3", "round 4"]

    # Trend logic: need >10 games in EACH of the 3 seasons
    df["HasTrendGames"] = (
        (df["GMS-2025"] > 10) & (df["GMS-2024"] > 10) & (df["GMS-2023"] > 10)
    )

    df["TrendIncreasing"] = (
        df["HasTrendGames"]
        & (df["AVG-2023"] < df["AVG-2024"])
        & (df["AVG-2024"] < df["AVG-2025"])
    )

    df["TrendDecreasing"] = (
        df["HasTrendGames"]
        & (df["AVG-2023"] > df["AVG-2024"])
        & (df["AVG-2024"] > df["AVG-2025"])
    )

    # Rising star condition: DOB after risingStarDate and total GMS > 30
    df["TotalGMS_3yrs"] = df["GMS-2025"] + df["GMS-2024"] + df["GMS-2023"]
    df["IsRisingStar"] = df["IsRisingStarDOB"] & (df["TotalGMS_3yrs"] > 30)

    # --- Apply per-round adjustments --------------------------------
    adj_scores = {col: [] for col in base_round_cols}

    for _, row in df.iterrows():
        scores = [row[c] for c in base_round_cols]
        s1, s2, s3, s4 = scores

        # Work on a mutable list
        adj = [s1, s2, s3, s4]

        # Players over 30 → -2 per round
        if row["IsOver30"]:
            adj = [v - 2 for v in adj]

        # Trend bonus / penalty
        if row["TrendIncreasing"]:
            adj = [v + 2 for v in adj]
        elif row["TrendDecreasing"]:
            adj = [v - 3 for v in adj]

        # Rising star bonus
        if row["IsRisingStar"]:
            adj = [v + 5 for v in adj]

        # Store
        for col_name, val in zip(base_round_cols, adj):
            adj_scores[col_name].append(val)

    # Attach adjusted scores to df (overwrite or keep both – here we overwrite)
    for col in base_round_cols:
        df[col] = adj_scores[col]

    # Total projected points R1–R4 (after all adjustments; R0 not included)
    df["TotalPoints_R1_R4"] = df["round 1"] + df["round 2"] + df["round 3"] + df["round 4"]

    # Slight advantage for DPP: add +2 to total R1–R4 projection if DPP
    df["IsDPP"] = df["EligiblePositions"].map(lambda lst: len(lst) > 1)
    df["AdjTotalPoints"] = df["TotalPoints_R1_R4"] + df["IsDPP"].astype(int) * 2

    # Cash generation at end of Round 4
    df["CashGen_R4"] = df["Price R4"] - df["StartingPrice"]

    return df



def build_and_solve_model(df: pd.DataFrame):
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
    prob = LpProblem("AFL_Fantasy_Starting_Team_2026", LpMaximize)

    # ---------- OBJECTIVE ----------

    # Adjusted R1–R4 projections (with DPP bonus)
    adj_points = df["AdjTotalPoints"].tolist()
    prices_start = df["StartingPrice"].tolist()
    cash_gen = df["CashGen_R4"].tolist()
    player_teams = df["Team"].tolist()

    # Starters contribute points
    starter_points = lpSum(
        adj_points[i] * start[(i, pos)]
        for (i, pos) in start
    )

    # Captain gets double – add extra adj_points[i]
    captain_bonus = lpSum(
        adj_points[i] * captain[i]
        for i in player_indices
    )

    # Cash generation: sum over all selected players
    total_cash_gen = lpSum(
        cash_gen[i] * x[i] for i in player_indices
    )

    # Total objective: maximise points + small weight * cash generation
    prob += (
        starter_points
        + captain_bonus
        + CASH_WEIGHT * total_cash_gen
    ), "Total_Points_R1_R4_Plus_CashGen"

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

    # 5. Salary cap (using StartingPrice as cost)
    prob += (
        lpSum(
            prices_start[i]
            * (
                lpSum(start[(i, pos)] for pos in df.loc[i, "EligiblePositions"])
                + lpSum(bench[(i, pos)] for pos in df.loc[i, "EligiblePositions"])
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

    # ---------- SOLVE ----------

    solver = PULP_CBC_CMD(msg=True)
    result_status = prob.solve(solver)

    if result_status != LpStatusOptimal:
        raise RuntimeError(
            f"Optimization did not find an optimal solution. Status code: {result_status}"
        )

    return prob, start, bench, captain


def extract_solution(df, start, bench, captain):
    rows = []

    total_cost = 0.0
    base_points = 0.0           # adjusted R1–R4 points for starters
    captain_bonus_points = 0.0  # extra adj points for captain
    total_cash_gen = 0.0

    for (i, pos) in start:
        if start[(i, pos)].value() == 1:
            is_captain = captain[i].value() == 1

            rows.append({
                "Player": df.loc[i, "Player"],
                "Team": df.loc[i, "Team"],
                "FantasyPosition": pos,
                "Role": "Starter",
                "Captain": "Yes" if is_captain else "No",
                "TotalPoints_R1_R4": df.loc[i, "TotalPoints_R1_R4"],
                "AdjTotalPoints": df.loc[i, "AdjTotalPoints"],
                "StartingPrice": df.loc[i, "StartingPrice"],
                "Price_R4": df.loc[i, "Price R4"],
                "CashGen_R4": df.loc[i, "CashGen_R4"],
                "IsDPP": bool(df.loc[i, "IsDPP"]),
            })

            total_cost += df.loc[i, "StartingPrice"]
            base_points += df.loc[i, "AdjTotalPoints"]
            total_cash_gen += df.loc[i, "CashGen_R4"]
            if is_captain:
                captain_bonus_points += df.loc[i, "AdjTotalPoints"]

    for (i, pos) in bench:
        if bench[(i, pos)].value() == 1:
            rows.append({
                "Player": df.loc[i, "Player"],
                "Team": df.loc[i, "Team"],
                "FantasyPosition": pos,
                "Role": "Bench",
                "Captain": "No",
                "TotalPoints_R1_R4": df.loc[i, "TotalPoints_R1_R4"],
                "AdjTotalPoints": df.loc[i, "AdjTotalPoints"],
                "StartingPrice": df.loc[i, "StartingPrice"],
                "Price_R4": df.loc[i, "Price R4"],
                "CashGen_R4": df.loc[i, "CashGen_R4"],
                "IsDPP": bool(df.loc[i, "IsDPP"]),
            })
            total_cost += df.loc[i, "StartingPrice"]
            total_cash_gen += df.loc[i, "CashGen_R4"]

    squad_df = pd.DataFrame(rows)
    total_adj_points_with_captain = base_points + captain_bonus_points

    # ---------- SORTING ----------
    # Sort by position (DEF, MID, RUC, FWD), then Starter before Bench, then StartingPrice desc
    position_order = {"DEF": 1, "MID": 2, "RUC": 3, "FWD": 4}
    role_order = {"Starter": 0, "Bench": 1}

    squad_df["PosSort"] = squad_df["FantasyPosition"].map(position_order).fillna(99)
    squad_df["RoleSort"] = squad_df["Role"].map(role_order).fillna(9)

    squad_df = squad_df.sort_values(
        ["PosSort", "RoleSort", "StartingPrice"],
        ascending=[True, True, False]
    ).reset_index(drop=True)

    squad_df = squad_df.drop(columns=["PosSort", "RoleSort"])

    return squad_df, total_cost, total_adj_points_with_captain, total_cash_gen


def save_output(squad_df, total_cost, total_adj_points, total_cash_gen):
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_DIR / OUTPUT_FILENAME

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        squad_df.to_excel(writer, sheet_name="Squad", index=False)

        summary_df = pd.DataFrame(
            {
                "Metric": [
                    "Total Cost (StartingPrice)",
                    "Total Adjusted Points R1–R4 (incl. Captain)",
                    "Total Cash Generation to End R4",
                    "Squad Size",
                    "Salary Cap",
                    "Cash Weight in Objective",
                ],
                "Value": [
                    total_cost,
                    total_adj_points,
                    total_cash_gen,
                    len(squad_df),
                    SALARY_CAP,
                    CASH_WEIGHT,
                ],
            }
        )
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

    print(f"Optimal starting team saved to: {output_path}")
    print(f"Total Cost: {total_cost:,.0f}")
    print(f"Total Adjusted Points R1–R4 (incl. captain): {total_adj_points:,.2f}")
    print(f"Total Cash Generation to End R4: {total_cash_gen:,.0f}")


def main():
    df = load_player_data()
    prob, start, bench, captain = build_and_solve_model(df)
    squad_df, total_cost, total_adj_points, total_cash_gen = extract_solution(
        df, start, bench, captain
    )
    save_output(squad_df, total_cost, total_adj_points, total_cash_gen)


if __name__ == "__main__":
    main()
