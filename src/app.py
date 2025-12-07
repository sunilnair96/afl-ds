import os
from datetime import datetime, date

import pandas as pd
import streamlit as st
import pulp

# -------------------------------------------------
# CONFIG
# -------------------------------------------------

# If the Excel is in the same folder as app.py, this is enough.

DATA_FILE = r"C:\Users\sunil\Projects\2026ws\afl-ds\data\processed\afl_fantasy_roundwise_projection_2026.xlsx"

# Salary cap for starting 22
SALARY_CAP = 17_800_000

# 30-man squad: 22 starters + 8 bench
BENCH_SIZE = 8
BENCH_MAX_PRICE = 340_000


# Starting 22 structure – change if you want a different structure
POSITION_REQUIREMENTS = {
    "DEF": 6,
    "MID": 8,
    "RUC": 2,
    "FWD": 6,
}

# Bench structure: 2 DEF, 3 MID, 1 RUC, 2 FWD = 8 total
BENCH_REQUIREMENTS = {
    "DEF": 2,
    "MID": 3,
    "RUC": 1,
    "FWD": 2,
}


# Round columns in your file
ROUND_COLUMNS = [f"round {i}" for i in range(0, 25)]  # 0..24

# First 14 "master" columns – treated as read-only in UI
MASTER_COLS = [
    "Player",
    "Team",
    "Position",
    "Priced At",
    "AVG-2025",
    "GMS-2025",
    "AVG-2024",
    "GMS-2024",
    "AVG-2023",
    "GMS-2023",
    "EARLY",
    "MID",
    "DOB",
    "Pick No.",
]


# -------------------------------------------------
# HELPERS
# -------------------------------------------------


def load_data():
    """
    Loads the dataset.
    Priority:
    1. If a projections_*.xlsx exists in SAVE_DIR, load the newest one.
    2. Otherwise load the original master projection file.
    """

    # 1. Look for saved projection files
    saved_files = [
        f for f in os.listdir(SAVE_DIR)
        if f.startswith("projections_") and f.endswith(".xlsx")
    ]

    if saved_files:
        # Sort by timestamp embedded in filename
        saved_files.sort(reverse=True)
        latest_file = saved_files[0]
        load_path = os.path.join(SAVE_DIR, latest_file)
        st.info(f"Loaded latest saved projections: {latest_file}")
    else:
        # 2. No history found → load original
        load_path = DATA_FILE
        st.info("Loaded original projection file.")

    # Read the chosen file
    df = pd.read_excel(load_path)

    # Validate expected columns
    missing = [c for c in MASTER_COLS + ROUND_COLUMNS if c not in df.columns]
    if missing:
        st.error(f"Missing columns in Excel: {missing}")
        st.stop()

    # Starting price formula
    df["starting_price"] = (df["Priced At"] * 10250 / 100).round() * 100

    # Normalize positions
    def normalise_pos(pos_str: str):
        if not isinstance(pos_str, str):
            return ""
        parts = []
        for p in pos_str.upper().replace(" ", "").split("/"):
            if p == "RUCK":
                p = "RUC"
            parts.append(p)
        return "/".join(parts)

    df["Position"] = df["Position"].apply(normalise_pos)

    # Calculate ages
    def calc_age(dob_str):
        if pd.isna(dob_str):
            return None
        try:
            dob = datetime.strptime(str(dob_str), "%d/%m/%Y").date()
        except ValueError:
            try:
                dob = pd.to_datetime(dob_str, dayfirst=True).date()
            except Exception:
                return None
        ref_date = date(2026, 1, 1)
        return ref_date.year - dob.year - (
            (ref_date.month, ref_date.day) < (dob.month, dob.day)
        )

    df["Age"] = df["DOB"].apply(calc_age)

    # Initialise flags if missing
    if "Use In Team" not in df.columns:
        df["Use In Team"] = True
    if "Lock" not in df.columns:
        df["Lock"] = False

    return df


def eligible_lines(pos_str):
    """Return set of lines a player can play on: DEF/MID/RUC/FWD."""
    if not isinstance(pos_str, str):
        return set()
    return set(pos_str.split("/")) & {"DEF", "MID", "RUC", "FWD"}


def optimise_team(df, current_round: int):
    """
    Build a 30-player squad:
      - 22 starters (Best22) under salary cap, with position requirements
      - 8 bench players with fixed structure:
          2 DEF, 3 MID, 1 RUC, 2 FWD
        each with starting_price <= BENCH_MAX_PRICE.

    Objective: maximise projected score of the 22 starters over the next 3 rounds.
    """

    if current_round < 1 or current_round > 22:
        raise ValueError("current_round must be between 1 and 22")

    future_rounds = [
        r
        for r in [current_round, current_round + 1, current_round + 3]
        if f"round {r}" in df.columns
    ]
    if not future_rounds:
        raise ValueError("No valid future rounds found for optimisation.")

    # Use only players flagged as available
    mask = df["Use In Team"].fillna(False).astype(bool)
    work = df[mask].copy().reset_index(drop=True)

    if work.empty:
        raise ValueError("No players marked as available for team building.")

    # Ensure numeric and no NaNs
    work["starting_price"] = pd.to_numeric(work["starting_price"], errors="coerce").fillna(0)

    proj_cols = [f"round {r}" for r in future_rounds]
    for c in proj_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce").fillna(0)

    work["projected_total"] = work[proj_cols].sum(axis=1).fillna(0)

    n = len(work)

    # LP model
    model = pulp.LpProblem("AFL_Fantasy_30_Squad_Optimisation", pulp.LpMaximize)

    # -------------------------------------------------
    # Starter variables: y_start[i, line]
    # -------------------------------------------------
    y_start = {}
    for i in range(n):
        lines = eligible_lines(work.loc[i, "Position"])
        for line in lines:
            y_start[(i, line)] = pulp.LpVariable(
                f"y_start_{i}_{line}", lowBound=0, upBound=1, cat="Binary"
            )

    if not y_start:
        raise ValueError("No position-eligible players found for starters.")

    # -------------------------------------------------
    # Bench variables: b[i, line]
    #   - only for players with price <= BENCH_MAX_PRICE
    #   - each player can occupy at most one bench line
    # -------------------------------------------------
    b = {}
    for i in range(n):
        price_i = work.loc[i, "starting_price"]
        if price_i <= BENCH_MAX_PRICE:
            lines = eligible_lines(work.loc[i, "Position"])
            for line in lines:
                b[(i, line)] = pulp.LpVariable(
                    f"b_{i}_{line}", lowBound=0, upBound=1, cat="Binary"
                )

    if not b:
        raise ValueError(
            "No eligible bench players found (check BENCH_MAX_PRICE and Use In Team)."
        )

    # Make sure we have enough eligible cheap players for each bench line
    for line, req in BENCH_REQUIREMENTS.items():
        elig_vars = [b[(i, l)] for (i, l) in b.keys() if l == line]
        if len(elig_vars) < req:
            raise ValueError(
                f"Not enough players priced <= {BENCH_MAX_PRICE} eligible for bench line {line}. "
                f"Need {req}, have {len(elig_vars)}. Adjust data or constraints."
            )

    # -------------------------------------------------
    # Objective: maximise projected score of starters only
    # -------------------------------------------------
    model += pulp.lpSum(
        work.loc[i, "projected_total"] * y_start[(i, line)]
        for (i, line) in y_start.keys()
    )

    # -------------------------------------------------
    # Position requirements for starters (Best22)
    # -------------------------------------------------
    for line, required in POSITION_REQUIREMENTS.items():
        vars_for_line = [y_start[(i, l)] for (i, l) in y_start.keys() if l == line]
        if vars_for_line:
            model += pulp.lpSum(vars_for_line) == required, f"{line}_starter_count"
        else:
            raise ValueError(f"No players eligible for starter line {line}")

    # -------------------------------------------------
    # Bench composition: 2 DEF, 3 MID, 1 RUC, 2 FWD
    # -------------------------------------------------
    for line, required in BENCH_REQUIREMENTS.items():
        vars_for_line = [b[(i, l)] for (i, l) in b.keys() if l == line]
        model += pulp.lpSum(vars_for_line) == required, f"{line}_bench_count"

    # (Optional but consistent: total bench size)
    model += (
        pulp.lpSum(b.values()) == BENCH_SIZE,
        "bench_size_total",
    )

    # -------------------------------------------------
    # Each player: at most one role (starter OR bench OR not selected)
    # -------------------------------------------------
    for i in range(n):
        starter_vars = [y_start[(idx, l)] for (idx, l) in y_start.keys() if idx == i]
        bench_vars = [b[(idx, l)] for (idx, l) in b.keys() if idx == i]
        vars_for_player = starter_vars + bench_vars
        if vars_for_player:
            model += pulp.lpSum(vars_for_player) <= 1, f"one_role_{i}"

    # -------------------------------------------------
    # Salary cap applies to starters only
    # -------------------------------------------------
    model += (
        pulp.lpSum(
            work.loc[i, "starting_price"] * y_start[(i, line)]
            for (i, line) in y_start.keys()
        )
        <= SALARY_CAP,
        "salary_cap_starters",
    )

    # -------------------------------------------------
    # Locks: must be starters, not just on bench
    # -------------------------------------------------
    for i in range(n):
        if bool(work.loc[i, "Lock"]):
            lines = [l for (idx, l) in y_start.keys() if idx == i]
            if lines:
                model += pulp.lpSum(y_start[(i, l)] for l in lines) == 1, f"lock_starter_{i}"

    # -------------------------------------------------
    # Solve
    # -------------------------------------------------
    model.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus[model.status]
    if status != "Optimal":
        raise RuntimeError(f"Optimisation failed: {status}")

    # -------------------------------------------------
    # Collect results: starters + bench
    # -------------------------------------------------
    records = []

    # Starters
    for (i, line), var in y_start.items():
        if var.value() == 1:
            row = work.loc[i].copy()
            row["Selected Line"] = line
            row["Role"] = "Starter"
            records.append(row)

    # Bench
    for (i, line), var in b.items():
        if var.value() == 1:
            row = work.loc[i].copy()
            row["Selected Line"] = line
            row["Role"] = "Bench"
            records.append(row)

    team_df = pd.DataFrame(records)

    # Order: starters first, then bench
    role_order = {"Starter": 0, "Bench": 1}
    team_df["RoleOrder"] = team_df["Role"].map(role_order)
    team_df = team_df.sort_values(
        ["RoleOrder", "Selected Line", "starting_price"],
        ascending=[True, True, False],
    ).drop(columns=["RoleOrder"])

    return team_df, future_rounds


SAVE_DIR = r"C:\Users\sunil\Projects\2026ws\afl-ds\data\versions"

def save_version(full_df, team_df, future_rounds):
    os.makedirs(SAVE_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rounds_label = "_".join(str(r) for r in future_rounds)

    team_path = os.path.join(SAVE_DIR, f"team_{rounds_label}_{stamp}.xlsx")
    proj_path = os.path.join(SAVE_DIR, f"projections_{stamp}.xlsx")

    with pd.ExcelWriter(team_path, engine="openpyxl") as writer:
        team_df.to_excel(writer, index=False, sheet_name="Team")

    with pd.ExcelWriter(proj_path, engine="openpyxl") as writer:
        full_df.to_excel(writer, index=False, sheet_name="Projections")

    return team_path, proj_path


# -------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------

def main():
    st.title("AFL Fantasy 2026 – Projection & Team Builder")

    st.sidebar.header("Settings")

    # Choose the round to start from – optimiser uses this round and next 2
    current_round = st.sidebar.slider(
        "Optimise from round:",
        min_value=1,
        max_value=22,
        value=1,
        step=1,
        help="The optimiser will use this round plus the next two rounds.",
    )

    # -------------------------------------------------
    # LOAD DATA
    # -------------------------------------------------
    df = load_data()  # << df is defined here

    st.write("TOTAL rows loaded:", len(df))
    st.write("Rows with missing Team:", df["Team"].isna().sum())
    st.write("Rows with missing Position:", df["Position"].isna().sum())
    st.write("Rows with missing starting_price:", df["starting_price"].isna().sum())

    # -------------------------------------------------
    # FILTERS
    # -------------------------------------------------
    teams = sorted(df["Team"].dropna().unique())
    # Flatten DPP like DEF/MID into distinct options
    positions = sorted(
        pd.unique(
            sum([p.split("/") for p in df["Position"].dropna()], [])
        )
    )

    st.sidebar.subheader("Filters")

    selected_teams = st.sidebar.multiselect("Teams", options=teams, default=teams)
    selected_positions = st.sidebar.multiselect(
        "Positions", options=positions, default=positions
    )
    
    # 👇 New sidebar toggle
    hide_excluded = st.sidebar.checkbox(
        "Hide players excluded from team (Use In Team = False)",
        value=True,
    )


    # Age filter
    if df["Age"].notna().any():
        min_age = int(df["Age"].min())
        max_age = int(df["Age"].max())
    else:
        min_age, max_age = 18, 40

    age_range = st.sidebar.slider(
        "Age range",
        min_value=min_age,
        max_value=max_age,
        value=(min_age, max_age),
    )

    # Price filter (based on starting_price we calculated in load_data)
    min_price = int(df["starting_price"].min())
    max_price = int(df["starting_price"].max())
    price_range = st.sidebar.slider(
        "Price range",
        min_value=min_price,
        max_value=max_price,
        value=(min_price, max_price),
        step=1000,
    )

    # Apply filters
    # Base filter for team/position/price
    base_filter = (
        df["Team"].isin(selected_teams)
        & df["Position"].apply(
            lambda s: any(p in selected_positions for p in s.split("/"))
        )
        & df["starting_price"].between(price_range[0], price_range[1])
    )

    # If hiding excluded players, add Use In Team = True
    if hide_excluded:
        base_filter &= df["Use In Team"]

    filtered = df[base_filter].copy()

    # Age filter (ignore players with unknown age)
    filtered = filtered[
        (filtered["Age"].isna())
        | (
            (filtered["Age"] >= age_range[0])
            & (filtered["Age"] <= age_range[1])
        )
    ]

    # -------------------------------------------------
    # EDITOR
    # -------------------------------------------------
    st.subheader("Edit projections & flags")

    edit_cols = (
        MASTER_COLS
        + ["Age", "starting_price", "Use In Team", "Lock"]
        + ROUND_COLUMNS
    )
    display_cols = [c for c in edit_cols if c in filtered.columns]

    edited = st.data_editor(
        filtered[display_cols],
        # num_rows="dynamic",
        use_container_width=True,
        disabled=[
            c for c in MASTER_COLS + ["Age", "starting_price"] if c in display_cols
        ],
        key="player_table",
    )

    # Merge edits back into df using Player + Team as key
    key_cols = ["Player", "Team"]

    # Merge edited flags & round projections – but keep existing values for hidden rows
    merge_cols = key_cols + ["Use In Team", "Lock"] + ROUND_COLUMNS
    merge_cols = [c for c in merge_cols if c in edited.columns]  # safety

    df = df.merge(
        edited[merge_cols],
        on=key_cols,
        how="left",
        suffixes=("", "_edit"),
    )

    # Prefer edited values where present; keep original for others
    for col in ["Use In Team", "Lock"] + ROUND_COLUMNS:
        edit_col = f"{col}_edit"
        if edit_col in df.columns:
            df[col] = df[edit_col].combine_first(df[col])
            df.drop(columns=[edit_col], inplace=True, errors="ignore")

    # Ensure flags never have NaN
    df["Use In Team"] = df["Use In Team"].fillna(True)   # or False, as you prefer
    df["Lock"] = df["Lock"].fillna(False)

    # -------------------------------------------------
    # OPTIMISATION SECTION
    # -------------------------------------------------
    st.markdown("---")
    st.subheader("Optimised starting team")

    optimise_btn = st.button("Build optimal team")

    if optimise_btn:
        try:
            team_df, future_rounds = optimise_team(df, current_round)
        except Exception as e:
            st.error(str(e))
        else:
            # Store in session_state so we can save later
            st.session_state["team_df"] = team_df
            st.session_state["future_rounds"] = future_rounds
            st.session_state["optimised_df"] = df.copy()  # full edited projections

    if "team_df" in st.session_state and "future_rounds" in st.session_state:
        team_df = st.session_state["team_df"]
        future_rounds = st.session_state["future_rounds"]
        full_df_for_save = st.session_state.get("optimised_df", df)

        proj_cols = [f"round {r}" for r in future_rounds]

        starters = team_df[team_df["Role"] == "Starter"].copy()
        bench = team_df[team_df["Role"] == "Bench"].copy()

        total_price_starters = int(starters["starting_price"].sum())
        total_proj_starters = int(starters[proj_cols].sum().sum())
        total_price_bench = int(bench["starting_price"].sum()) if not bench.empty else 0

        st.write(f"Optimised for rounds: {future_rounds}")
        st.write(f"Starters price: {total_price_starters:,} (cap {SALARY_CAP:,})")
        st.write(f"Bench total price (no cap, max {BENCH_MAX_PRICE:,} each): {total_price_bench:,}")
        st.write(
            f"Total projected points for starters over {len(future_rounds)} rounds: {total_proj_starters:,}"
        )

        show_cols = [
            "Player",
            "Team",
            "Position",
            "Role",
            "Selected Line",
            "starting_price",
        ] + proj_cols

        st.markdown("#### Starters (Best 22)")
        st.dataframe(starters[show_cols], use_container_width=True)

        st.markdown("#### Bench (8 players, max 340k each)")
        if not bench.empty:
            st.dataframe(bench[show_cols], use_container_width=True)
        else:
            st.write("No bench players selected.")

        if st.button("Save this version (team + projections)"):
            team_path, proj_path = save_version(full_df_for_save, team_df, future_rounds)
            st.success(f"Saved team to: {team_path}")
            st.success(f"Saved projections to: {proj_path}")


if __name__ == "__main__":
    main()
