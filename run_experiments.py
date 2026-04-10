"""
run_experiments.py – Sensitivity analysis for VRP ALNS+Tabu
Usage:
    venv/bin/python run_experiments.py --set 1   # Fleet size (K=2-6)
    venv/bin/python run_experiments.py --set 2   # Customer density (N=30,40,50,60)
    venv/bin/python run_experiments.py --set 3   # Capacity (1800,1700,1600)
"""
import sys
import math
import argparse
import statistics
sys.path.append('.')
from src.vrp_alns_tabu import alns_tabu_solve, load_data

# ── Config ────────────────────────────────────────────────────────────────────
MAX_ITER = 300    # iterations per ALNS run (300 ≈ 7s/run, good balance speed/quality)
NUM_RUNS = 3      # seeds per config  (42, 43, 44)

BASE_N   = 50     # baseline số cửa hàng
BASE_K   = 4      # baseline số xe
BASE_CAP = 1800   # baseline capacity (kg)

# ── Helpers ───────────────────────────────────────────────────────────────────

def subset_data(data, N_stores, capacity):
    n = N_stores + 1  # +1 depot
    d = data.copy()
    d["distance_matrix"]   = data["distance_matrix"][:n, :n]
    d["congestion_matrix"] = data["congestion_matrix"][:n, :n]
    d["demands"]           = data["demands"][:n]
    d["time_windows"]      = data["time_windows"][:n]
    d["store_names"]       = data["store_names"][:n]
    d["vehicle_capacity"]  = capacity
    return d


def calculate_metrics(routes_info, data):
    """Return (OTDR %, CES score)."""
    total_customers = sum(len(r["nodes"]) - 2 for r in routes_info)
    if total_customers == 0:
        return 0.0, 0.0

    on_time = 0
    total_edges = 0
    total_cong = 0

    for r in routes_info:
        for t in r["timing"]:
            if t["node"] != 0 and t["lateness"] <= 0.01:
                on_time += 1

        nodes = r["nodes"]
        for i in range(1, len(nodes)):
            cong = data["congestion_matrix"][nodes[i-1]][nodes[i]]
            total_cong += (cong - 1)
            total_edges += 1

    otdr = (on_time / total_customers) * 100
    ces  = total_cong / total_edges if total_edges > 0 else 0.0
    return otdr, ces


def run_single(data_template, N, K, cap, seed):
    """Run one ALNS solve. Returns (cost_M, otdr, ces, time) or None on failure."""
    data_sub = subset_data(data_template, N, cap)
    status, total_cost, routes_info, solve_time = alns_tabu_solve(
        data_sub, K, max_iterations=MAX_ITER, seed=seed, verbose=False
    )
    if total_cost is None or not math.isfinite(total_cost) or not routes_info:
        return None
    otdr, ces = calculate_metrics(routes_info, data_sub)
    return total_cost / 1_000_000, otdr, ces * 1000, solve_time


def run_config(base_data, N, K, cap):
    """Run NUM_RUNS seeds and return aggregated lists."""
    costs, otdrs, cess, times = [], [], [], []
    for seed in range(42, 42 + NUM_RUNS):
        print(f"  [seed={seed}] N={N}, K={K}, Cap={cap} ...", file=sys.stderr, flush=True)
        result = run_single(base_data, N, K, cap, seed)
        if result:
            c, o, e, t = result
            costs.append(c); otdrs.append(o); cess.append(e); times.append(t)
            print(f"           cost={c:,.1f}M  OTDR={o:.1f}%  CES={e:.1f}  t={t:.1f}s",
                  file=sys.stderr, flush=True)
        else:
            print(f"           → FAILED / Infeasible", file=sys.stderr, flush=True)
    return costs, otdrs, cess, times


def fmt(values):
    if not values:
        return "N/A"
    if len(values) == 1:
        return f"{values[0]:,.1f} ± 0.0"
    return f"{statistics.mean(values):,.1f} ± {statistics.stdev(values):,.1f}"


def print_header():
    print("| Parameter variation | Total cost (Mil VND) | OTDR (%) | CES score | Computation time (Sec.) |")
    print("| :--- | :--- | :--- | :--- | :--- |")


def print_row(label, costs, otdrs, cess, times):
    print(f"| {label} | {fmt(costs)} | {fmt(otdrs)} | {fmt(cess)} | {fmt(times)} |")



def run_set1(data):
    """Bộ 1 – Fleet size sensitivity (K = 2-6, N=50, Cap=1800)."""
    print_header()
    print("| **Fleet Size Sensitivity** | | | | |")
    baseline = None
    for K in [4,5,6,7]:
        label = f"{K} vehicles" + (" (baseline)" if K == BASE_K else "")
        res = run_config(data, BASE_N, K, BASE_CAP)
        print_row(label, *res)
        if K == BASE_K:
            baseline = res
    return baseline


def run_set2(data, baseline=None):
    """Bộ 2 – Capacity """
    print_header()
    print("| **Capacity Sensitivity** | | | | |")
    for cap in [2000, 1900, 1800, 1700, 1600]:
        label = f"{cap} kg capacity" + (" (baseline)" if cap == BASE_CAP else "")
        if cap == BASE_CAP and baseline:
            print_row(label, *baseline)
        else:
            res = run_config(data, BASE_N, BASE_K, cap)
            print_row(label, *res)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", type=int, choices=[1, 2], required=True,
                        help="Which sensitivity set to run: 1=Fleet, 2=Capacity")
    args = parser.parse_args()

    print(f"\nLoading data...", file=sys.stderr, flush=True)
    data = load_data()
    print(f"Loaded {len(data['distance_matrix'])-1} stores + depot\n", file=sys.stderr, flush=True)

    if args.set == 1:
        run_set1(data)
    elif args.set == 2:
        run_set2(data)

if __name__ == "__main__":
    main()
