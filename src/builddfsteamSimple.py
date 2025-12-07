# src/builddfsteam.py
from __future__ import annotations
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
from pulp import LpProblem, LpMaximize, LpVariable, LpBinary, lpSum, PULP_CBC_CMD, LpStatusOptimal

ALLOWED_STATUSES = {"START", "NAMED IN TEAM TO PLAY", "CONFIRMED"}
SALARY_CAP = 100_000

# Exactly these many slots by position
SLOTS = {
    "DEF": 2,
    "MID": 4,
    "RK": 1,
    "FWD": 2,
}

POSITION_ORDER = ["DEF", "MID", "RK", "FWD"]  # for pretty printing


def load_data(csv_path: Path) -> pd.DataFrame:
    req_cols = {
        "ID",
        "Name",
        "Team",
        "Opponent",
        "Status",
        "Price",
        "Projection",
        "SD",
        "Floor",
        "Ceiling",
        "Position",
        "Position2",
    }
    df = pd.read_csv(csv_path)
    missing = req_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")

    # Normalize
    df["StatusNorm"] = df["Status"].astype(str).str.strip().str.upper()
    df["Position"] = df["Position"].astype(str).str.strip().str.upper()
    # Position2 may be blank/NaN
    df["Position2"] = df.get("Position2", "").astype(str).str.strip().str.upper().replace({"NAN": ""})

    # Filter allowed statuses
    df = df[df["StatusNorm"].isin(ALLOWED_STATUSES)].copy()

    # Drop duplicates on ID (keep best Projection if duplicates)
    if "ID" in df.columns:
        df = df.sort_values("Projection", ascending=False).drop_duplicates(subset=["ID"], keep="first")

    # Keep only rows with a recognized position
    valid_pos = set(SLOTS.keys())
    df = df[(df["Position"].isin(valid_pos)) | (df["Position2"].isin(valid_pos))].copy()

    # Clean numeric
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce").fillna(0).astype(int)
    df["Projection"] = pd.to_numeric(df["Projection"], errors="coerce").fillna(0.0)

    # Build eligible positions set per player (Position or Position2)
    def _elig(row):
        poss = set()
        p1 = row["Position"]
        p2 = row["Position2"] if isinstance(row["Position2"], str) else ""
        if p1 in valid_pos:
            poss.add(p1)
        if p2 in valid_pos and p2 != p1:
            poss.add(p2)
        return sorted(poss)

    df["EligiblePositions"] = df.apply(_elig, axis=1)
    # Remove any with no eligibility (shouldn't happen after filter)
    df = df[df["EligiblePositions"].map(len) > 0].copy()
    df.reset_index(drop=True, inplace=True)
    return df


def solve_lineup(df: pd.DataFrame):
    """
    ILP with assignment:
      - select[p] ∈ {0,1} whether player p is chosen
      - assign[p,pos] ∈ {0,1} whether player p fills position 'pos'
      Constraints:
        - For each p: sum_pos assign[p,pos] == select[p]
        - For each pos: sum_p assign[p,pos] == SLOTS[pos]
        - Salary: sum_p Price[p]*select[p] <= SALARY_CAP
      Objective: maximize sum_p Projection[p]*select[p]
    """
    players = list(df["ID"])
    price = {row.ID: int(row.Price) for _, row in df.iterrows()}
    proj = {row.ID: float(row.Projection) for _, row in df.iterrows()}
    elig = {row.ID: list(row.EligiblePositions) for _, row in df.iterrows()}

    prob = LpProblem("DS_Lineup_MaxProjection", LpMaximize)

    select = {p: LpVariable(f"sel_{p}", lowBound=0, upBound=1, cat=LpBinary) for p in players}
    assign = {(p, pos): LpVariable(f"as_{p}_{pos}", lowBound=0, upBound=1, cat=LpBinary)
              for p in players for pos in elig[p]}

    # Objective
    prob += lpSum(proj[p] * select[p] for p in players)

    # Salary cap
    prob += lpSum(price[p] * select[p] for p in players) <= SALARY_CAP, "SalaryCap"

    # Each player assigned to exactly one of their eligible positions if selected
    for p in players:
        prob += lpSum(assign.get((p, pos), 0) for pos in elig[p]) == select[p], f"AssignOnce_{p}"

    # Position slot counts
    for pos, count in SLOTS.items():
        prob += lpSum(assign.get((p, pos), 0) for p in players) == count, f"Slots_{pos}"

    # Solve
    status = prob.solve(PULP_CBC_CMD(msg=False))
    if prob.status != LpStatusOptimal:
        return None, prob

    chosen_ids = [p for p in players if select[p].value() >= 0.99]

    # Determine chosen position for each selected player (the one with assign=1)
    chosen_rows = []
    df_by_id = df.set_index("ID")
    for p in chosen_ids:
        pos_chosen = ""
        for pos in elig[p]:
            var = assign[(p, pos)]
            if var.value() >= 0.99:
                pos_chosen = pos
                break
        row = df_by_id.loc[p]
        chosen_rows.append({
            "Slot": pos_chosen,
            "ID": p,
            "Name": row["Name"],
            "Team": row["Team"],
            "Opponent": row["Opponent"],
            "Status": row["Status"],
            "Price": int(row["Price"]),
            "Projection": float(row["Projection"]),
            "PrimaryPos": row["Position"],
            "SecondaryPos": row["Position2"],
        })

    lineup_df = pd.DataFrame(chosen_rows).sort_values(
        by=["Slot", "Price", "Projection"], ascending=[True, False, False]
    )
    totals = {
        "TotalPlayers": len(lineup_df),
        "TotalSalary": int(lineup_df["Price"].sum()) if not lineup_df.empty else 0,
        "TotalProjection": float(lineup_df["Projection"].sum()) if not lineup_df.empty else 0.0,
        "SalaryCap": SALARY_CAP,
        "SalaryRemaining": SALARY_CAP - (int(lineup_df["Price"].sum()) if not lineup_df.empty else 0),
    }
    return (lineup_df, totals), prob


def main():
    ap = argparse.ArgumentParser(description="Build optimal DraftStars AFL team from ProjectionTable.csv")
    ap.add_argument("--csv", default=str(Path("data/raw/ProjectionTable.csv")),
                    help="Path to ProjectionTable.csv (default: data/raw/ProjectionTable.csv)")
    ap.add_argument("--outdir", default=str(Path("data/processed")), help="Output directory for lineup CSV")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_data(csv_path)
    if df.empty:
        raise SystemExit("No eligible players after filtering statuses/positions.")

    result, prob = solve_lineup(df)
    if result is None:
        raise SystemExit("No optimal solution found (infeasible under given constraints).")

    lineup_df, totals = result

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = outdir / f"dfsteam_{ts}.csv"
    lineup_df.to_csv(out_csv, index=False)

    # Print summary
    print("\n=== Optimal DraftStars Lineup ===")
    # order by our display order
    lineup_df = lineup_df.set_index("Slot").loc[POSITION_ORDER].reset_index()
    print(lineup_df[["Slot", "Name", "Team", "Opponent", "Price", "Projection", "Status"]].to_string(index=False))

    print("\nTotals:")
    print(f"  Players         : {totals['TotalPlayers']}")
    print(f"  Salary used     : {totals['TotalSalary']} / {totals['SalaryCap']}")
    print(f"  Salary remaining: {totals['SalaryRemaining']}")
    print(f"  Total projection: {totals['TotalProjection']:.2f}")

    print(f"\nSaved to: {out_csv}")


if __name__ == "__main__":
    main()
