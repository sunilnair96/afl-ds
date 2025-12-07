# src/builddfsteam.py
from __future__ import annotations
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
from pulp import LpProblem, LpMaximize, LpVariable, LpBinary, lpSum, PULP_CBC_CMD, LpStatusOptimal

'''
# 1) Default (Projection)
python .\src\builddfsteam.py

# 2) Risk-adjusted (Projection − k × SD), with k = 0.6
python .\src\builddfsteam.py --objective proj-sd --k 0.6

# 3) Ceiling
python .\src\builddfsteam.py --objective ceiling

'''

ALLOWED_STATUSES = {"START", "NAMED IN TEAM TO PLAY", "CONFIRMED"}
SALARY_CAP = 100_000
SLOTS = {"DEF": 2, "MID": 4, "RK": 1, "FWD": 2}
POSITION_ORDER = ["DEF", "MID", "RK", "FWD"]

def load_data(csv_path: Path) -> pd.DataFrame:
    req_cols = {
        "ID","Name","Team","Opponent","Status","Price",
        "Projection","SD","Floor","Ceiling","Position","Position2",
    }
    df = pd.read_csv(csv_path)
    missing = req_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")


    df["StatusNorm"] = df["Status"].astype(str).str.strip().str.upper()
    df["Position"]   = df["Position"].astype(str).str.strip().str.upper()
    df["Position2"]  = df.get("Position2", "").astype(str).str.strip().str.upper().replace({"NAN": ""})

    df = df[df["StatusNorm"].isin(ALLOWED_STATUSES)].copy()

    # Deduplicate on ID (keep higher Projection)
    df = df.sort_values("Projection", ascending=False).drop_duplicates(subset=["ID"], keep="first")

    valid_pos = set(SLOTS.keys())
    df = df[(df["Position"].isin(valid_pos)) | (df["Position2"].isin(valid_pos))].copy()

    for c, cast, fill in [("Price", int, 0), ("Projection", float, 0.0), ("SD", float, 0.0), ("Ceiling", float, 0.0)]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(fill).astype(cast)

    def _elig(row):
        poss = []
        if row["Position"] in valid_pos: poss.append(row["Position"])
        if row["Position2"] in valid_pos and row["Position2"] != row["Position"]: poss.append(row["Position2"])
        return poss

    df["EligiblePositions"] = df.apply(_elig, axis=1)
    df = df[df["EligiblePositions"].map(len) > 0].copy()
    df.reset_index(drop=True, inplace=True)
    return df

def build_objective_vector(df: pd.DataFrame, objective: str, k: float) -> dict:
    """
    Returns a dict {player_id: score} according to chosen objective.
    - projection : uses df['Projection']
    - proj-sd    : uses df['Projection'] - k * df['SD']
    - ceiling    : uses df['Ceiling']
    """
    if objective == "projection":
        return {row.ID: float(row.Projection) for _, row in df.iterrows()}
    if objective == "proj-sd":
        return {row.ID: float(row.Projection) - k * float(row.SD) for _, row in df.iterrows()}
    if objective == "ceiling":
        return {row.ID: float(row.Ceiling) for _, row in df.iterrows()}
    raise ValueError(f"Unknown objective: {objective}")

def solve_lineup(df: pd.DataFrame, objective_scores: dict):
    players = list(df["ID"])
    price = {row.ID: int(row.Price) for _, row in df.iterrows()}
    elig  = {row.ID: list(row.EligiblePositions) for _, row in df.iterrows()}

    prob = LpProblem("DS_Lineup", LpMaximize)

    select = {p: LpVariable(f"sel_{p}", lowBound=0, upBound=1, cat=LpBinary) for p in players}
    assign = {(p, pos): LpVariable(f"as_{p}_{pos}", lowBound=0, upBound=1, cat=LpBinary)
              for p in players for pos in elig[p]}

    # Objective
    prob += lpSum(objective_scores[p] * select[p] for p in players), "Objective"

    # Salary cap
    prob += lpSum(price[p] * select[p] for p in players) <= SALARY_CAP, "SalaryCap"

    # Assignment: chosen player must fill exactly one eligible slot
    for p in players:
        prob += lpSum(assign.get((p, pos), 0) for pos in elig[p]) == select[p], f"AssignOnce_{p}"

    # Slot counts
    for pos, count in SLOTS.items():
        prob += lpSum(assign.get((p, pos), 0) for p in players) == count, f"Slots_{pos}"

    status = prob.solve(PULP_CBC_CMD(msg=False))
    if prob.status != LpStatusOptimal:
        return None, prob

    # Build lineup DF
    chosen_ids = [p for p in players if select[p].value() >= 0.99]
    rows_by_id = df.set_index("ID")
    chosen_rows = []
    for p in chosen_ids:
        chosen_pos = next((pos for pos in elig[p] if assign[(p, pos)].value() >= 0.99), "")
        r = rows_by_id.loc[p]
        chosen_rows.append({
            "Slot": chosen_pos,
            "ID": p,
            "Name": r["Name"],
            "Team": r["Team"],
            "Opponent": r["Opponent"],
            "Status": r["Status"],
            "Price": int(r["Price"]),
            "Projection": float(r["Projection"]),
            "SD": float(r["SD"]),
            "Ceiling": float(r["Ceiling"]),
            "PrimaryPos": r["Position"],
            "SecondaryPos": r["Position2"],
            "ObjectiveScore": float(objective_scores[p]),
        })
    import numpy as np
    lineup_df = pd.DataFrame(chosen_rows).sort_values(
        by=["Slot","Price","Projection"], ascending=[True, False, False]
    )
    totals = {
        "TotalPlayers": len(lineup_df),
        "TotalSalary": int(lineup_df["Price"].sum()) if not lineup_df.empty else 0,
        "TotalProjection": float(lineup_df["Projection"].sum()) if not lineup_df.empty else 0.0,
        "TotalCeiling": float(lineup_df["Ceiling"].sum()) if not lineup_df.empty else 0.0,
        "TotalObj": float(lineup_df["ObjectiveScore"].sum()) if not lineup_df.empty else 0.0,
        "SalaryCap": SALARY_CAP,
        "SalaryRemaining": SALARY_CAP - (int(lineup_df["Price"].sum()) if not lineup_df.empty else 0),
    }
    return (lineup_df, totals), prob

def main():
    ap = argparse.ArgumentParser(description="Build optimal DraftStars AFL team from ProjectionTable.csv")
    ap.add_argument("--csv", default=str(Path("data/raw/ProjectionTable.csv")),
                    help="Path to ProjectionTable.csv (default: data/raw/ProjectionTable.csv)")
    ap.add_argument("--outdir", default=str(Path("data/processed")),
                    help="Output directory for lineup CSV")
    ap.add_argument("--objective", choices=["projection","proj-sd","ceiling"], default="projection",
                    help="Objective: projection | proj-sd (Projection - k*SD) | ceiling")
    ap.add_argument("--k", type=float, default=0.5,
                    help="Risk penalty k for proj-sd objective (higher penalizes SD more)")
    args = ap.parse_args()

    csv_path = Path(args.csv); outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_data(csv_path)
    if df.empty:
        raise SystemExit("No eligible players after filtering statuses/positions.")

    obj_vec = build_objective_vector(df, args.objective, args.k)
    result, prob = solve_lineup(df, obj_vec)
    if result is None:
        raise SystemExit("No optimal solution found (check constraints or columns).")

    lineup_df, totals = result

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = outdir / f"dfsteam_{args.objective}_{ts}.csv"
    lineup_df.to_csv(out_csv, index=False)

    # Pretty print in slot order
    lineup_df = lineup_df.set_index("Slot").loc[POSITION_ORDER].reset_index()

    print("\n=== Optimal DraftStars Lineup ===")
    cols = ["Slot","Name","Team","Opponent","Price","Projection","SD","Ceiling","Status","ObjectiveScore"]
    print(lineup_df[cols].to_string(index=False))
    print("\nTotals:")
    print(f"  Players         : {totals['TotalPlayers']}")
    print(f"  Salary used     : {totals['TotalSalary']} / {totals['SalaryCap']}")
    print(f"  Salary remaining: {totals['SalaryRemaining']}")
    print(f"  Total projection: {totals['TotalProjection']:.2f}")
    print(f"  Total ceiling   : {totals['TotalCeiling']:.2f}")
    print(f"  Total objective : {totals['TotalObj']:.2f}")
    print(f"\nSaved to: {out_csv}")

if __name__ == "__main__":
    main()
