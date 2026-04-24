"""
run_experiments.py – Demand Sensitivity Analysis for VRP T-ALNS-RRD

Cases:
  --case 1   ± 10%, 20% demand toàn bộ stores (global)
  --case 2   + 10%, 20% demand random một số stores (< 1/2 tổng 139 CH)
  --case 3   - 10%, 20% demand random một số stores (< 1/2 tổng 139 CH)
  --case 4   ± 10%, 20% demand random một số stores (> 1/2 tổng 139 CH)
  --case 0   Chạy tất cả case 1-4

Usage:
    venv/bin/python run_experiments.py --case 1
    venv/bin/python run_experiments.py --case 0   # all cases
"""
import sys
import os
import math
import argparse
import statistics
import random
import numpy as np

# ── Setup paths ──────────────────────────────────────────────────────────────
project_root = os.path.dirname(os.path.abspath(__file__))
os.chdir(project_root)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

from src.vrp_alns_tabu_rrd import (
    load_data,
    rrd_solve,
    calc_metrics,
    CostEvaluatorRRD,
)

# ── Config ────────────────────────────────────────────────────────────────────
ALNS_ITER     = 50      # ALNS iterations per solve
REASSIGN_ITER = 10      # Phase 2a reassignment iterations
SEED          = 42
W             = 110     # print width


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _run_scenario(data, demands_list, K_range, label="", seed_offset=0):
    """
    Solve with given demands for each K in K_range, return best result by
    composite score (same weighting as main solver).
    """
    original_demands = list(data["demands"])
    data["demands"] = list(demands_list)

    sweep = []
    for K in K_range:
        np.random.seed(SEED + seed_offset)
        random.seed(SEED + seed_offset)
        res = rrd_solve(
            data,
            num_vehicles=K,
            alns_iterations=ALNS_ITER,
            reassign_iter=REASSIGN_ITER,
            verbose=False,
        )
        m = calc_metrics(res["rrd_routes"], data)
        n_active = sum(1 for r in res["rrd_routes"] if len(r["nodes"]) > 2)
        sweep.append({
            "K": K, "cost": res["rrd_cost"], "otdr": m["otdr"],
            "ces": m["ces"], "n_active": n_active, "result": res,
        })

    # Composite score (OTDR dominant)
    costs = [s["cost"] for s in sweep]
    otdrs = [s["otdr"] for s in sweep]
    ces_v = [s["ces"]  for s in sweep]

    def _norm(vals, invert=False):
        lo, hi = min(vals), max(vals)
        span = hi - lo if hi > lo else 1.0
        if invert:
            return [(hi - v) / span for v in vals]
        return [(v - lo) / span for v in vals]

    nc  = _norm(costs)
    no  = _norm(otdrs, invert=True)
    nce = _norm(ces_v)
    for i, s in enumerate(sweep):
        s["_score"] = 0.3 * nc[i] + 0.6 * no[i] + 0.1 * nce[i]

    best = min(sweep, key=lambda x: x["_score"])

    # Restore original demands
    data["demands"] = original_demands
    return best


def _select_random_stores(num_stores, count, seed):
    """Return sorted list of random store indices (1-based) of given count."""
    rng = random.Random(seed)
    all_stores = list(range(1, num_stores + 1))
    return sorted(rng.sample(all_stores, min(count, num_stores)))


def _perturb_demands(baseline, store_indices, factor):
    """
    Return a copy of baseline demands with specified stores perturbed by factor.
    factor can be: 1.10 (+10%), 0.90 (-10%), etc.
    """
    perturbed = list(baseline)
    for sid in store_indices:
        perturbed[sid] = baseline[sid] * factor
    return perturbed


def _perturb_demands_mixed(baseline, store_indices, pct_abs, seed):
    """
    Return a copy of baseline demands with specified stores randomly
    perturbed by ±pct_abs (e.g. 0.10 for ±10%).
    Each store randomly gets + or -.
    """
    rng = random.Random(seed)
    perturbed = list(baseline)
    for sid in store_indices:
        sign = rng.choice([1, -1])
        perturbed[sid] = baseline[sid] * (1 + sign * pct_abs)
    return perturbed


def _print_scenario_table(title, description, data, baseline_demands, scenarios,
                          K_range, num_stores):
    """
    Generic function to run and print a sensitivity table.

    scenarios: list of tuples (label, demands_list, seed_offset)
    """
    total_base = sum(baseline_demands[1:])
    capacity = data["vehicle_capacity"]

    print(f"\n{'═'*W}")
    print(f"  {title}")
    print(f"  {description}")
    print(f"  {num_stores} stores  |  Fleet sweep: K = {K_range[0]}→{K_range[-1]}")
    print(f"  Total baseline demand = {total_base:,.1f} kg  |  Capacity = {capacity} kg")
    print(f"{'═'*W}")

    # Run baseline
    print(f"\n  ⏳  Running baseline ...", file=sys.stderr, flush=True)
    base_best = _run_scenario(data, baseline_demands, K_range,
                              label="Baseline", seed_offset=0)
    base_cost   = base_best["cost"]
    base_otdr   = base_best["otdr"]
    base_K      = base_best["K"]
    base_active = base_best["n_active"]

    print(f"\n  Baseline (0%)")
    print(f"    Total demand : {total_base:>12,.1f} kg")
    print(f"    Best K       : {base_K:>12}")
    print(f"    Active veh.  : {base_active:>12}")
    print(f"    Cost         : {base_cost:>12,.0f} VND")
    print(f"    OTDR         : {base_otdr:>11.2f}%")

    # Header
    print(f"\n  {'─'*100}")
    print(
        f"  {'Scenario':<16} {'Tot.Demand':<13} {'Best K':<8} {'Active':<8} "
        f"{'Cost (VND)':<18} {'Δ Cost':<16} {'Δ%':<10} {'OTDR':<9} {'ΔOTDR':<9}"
    )
    print(
        f"  {'─'*16} {'─'*13} {'─'*8} {'─'*8} "
        f"{'─'*18} {'─'*16} {'─'*10} {'─'*9} {'─'*9}"
    )

    results = []
    for idx, (label, demands_list, seed_offset) in enumerate(scenarios):
        print(f"  ⏳  Running {label} ...", file=sys.stderr, flush=True)
        best_s = _run_scenario(data, demands_list, K_range,
                               label=label, seed_offset=seed_offset)

        p_cost   = best_s["cost"]
        p_otdr   = best_s["otdr"]
        p_K      = best_s["K"]
        p_active = best_s["n_active"]
        tot_d    = sum(demands_list[1:])

        delta      = p_cost - base_cost
        delta_pct  = delta / base_cost * 100 if base_cost > 0 else 0
        delta_otdr = p_otdr - base_otdr
        cost_sign  = "+" if delta >= 0 else ""
        otdr_sign  = "+" if delta_otdr >= 0 else ""
        k_note     = f" (was {base_K})" if p_K != base_K else ""

        # Count perturbed stores
        n_changed = sum(1 for i in range(1, len(demands_list))
                        if abs(demands_list[i] - baseline_demands[i]) > 0.01)

        print(
            f"  {label:<16} {tot_d:<13,.1f} {p_K:<8}{k_note:>0} "
            f"{p_active:<8} {p_cost:<18,.0f} "
            f"{cost_sign}{delta:<15,.0f} {cost_sign}{delta_pct:<9.2f}% "
            f"{p_otdr:<9.2f}%{otdr_sign}{delta_otdr:<8.2f}%"
        )

        results.append({
            "label": label, "tot_demand": tot_d, "best_K": p_K,
            "n_active": p_active, "cost": p_cost, "delta": delta,
            "delta_pct": delta_pct, "otdr": p_otdr, "delta_otdr": delta_otdr,
            "n_changed": n_changed,
        })

    # Summary
    _print_summary(results, base_K, base_cost, base_otdr)
    return results


def _print_summary(results, base_K, base_cost, base_otdr):
    """Print sensitivity summary from results list."""
    if not results:
        return

    inc = [r for r in results if r["delta_pct"] > 0.01]
    dec = [r for r in results if r["delta_pct"] < -0.01]

    avg_inc_cost = sum(r["delta_pct"] for r in inc) / len(inc) if inc else 0
    avg_dec_cost = sum(r["delta_pct"] for r in dec) / len(dec) if dec else 0
    avg_inc_otdr = sum(r["delta_otdr"] for r in inc) / len(inc) if inc else 0
    avg_dec_otdr = sum(r["delta_otdr"] for r in dec) / len(dec) if dec else 0

    k_changes = [r for r in results if r["best_K"] != base_K]
    max_impact = max(results, key=lambda r: abs(r["delta_pct"]))

    print(f"\n  {'─'*100}")
    print(f"  SENSITIVITY SUMMARY")
    print(f"  {'─'*100}")
    if inc:
        print(f"  Avg cost change  (cost increase scenarios): +{avg_inc_cost:.2f}%")
    if dec:
        print(f"  Avg cost change  (cost decrease scenarios): {avg_dec_cost:.2f}%")
    print(
        f"  Most impactful scenario : {max_impact['label']}  →  "
        f"Δcost = {'+' if max_impact['delta_pct']>=0 else ''}{max_impact['delta_pct']:.2f}%"
        f"  |  OTDR = {max_impact['otdr']:.2f}%"
    )
    if k_changes:
        for kc in k_changes:
            print(
                f"  ⚠ Optimal K changed at {kc['label']}: "
                f"K = {base_K} → {kc['best_K']}  "
                f"(active vehicles: {kc['n_active']})"
            )
    else:
        print(f"  ✓ Optimal K = {base_K} remains stable across all scenarios")
    print(f"{'═'*W}")


# ═══════════════════════════════════════════════════════════════════════════════
#  CASE 1: Global demand ±10%, ±20%  (tất cả stores)
# ═══════════════════════════════════════════════════════════════════════════════

def run_case1(data, best_K):
    """Case 1: ±10%, ±20% demand cho TOÀN BỘ stores."""
    baseline = list(data["demands"])
    num_stores = len(baseline) - 1
    K_lo = max(1, best_K - 2)
    K_hi = best_K + 2
    K_range = range(K_lo, K_hi + 1)

    scenarios = []
    for idx, (label, factor) in enumerate([
        ("-20%", 0.80), ("-10%", 0.90),
        ("+10%", 1.10), ("+20%", 1.20),
    ]):
        perturbed = list(baseline)
        for sid in range(1, num_stores + 1):
            perturbed[sid] = baseline[sid] * factor
        scenarios.append((f"Global {label}", perturbed, idx + 1))

    return _print_scenario_table(
        title="CASE 1: GLOBAL DEMAND SENSITIVITY",
        description="Perturb ALL stores' demand simultaneously by ±10%, ±20%",
        data=data,
        baseline_demands=baseline,
        scenarios=scenarios,
        K_range=K_range,
        num_stores=num_stores,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  CASE 2: +10%, +20% random demand một số stores (< 1/2 tổng CH)
# ═══════════════════════════════════════════════════════════════════════════════

def run_case2(data, best_K):
    """Case 2: +10%, +20% demand cho RANDOM subset stores (< 1/2 of 139)."""
    baseline = list(data["demands"])
    num_stores = len(baseline) - 1
    half = num_stores // 2   # 69 stores (< 1/2 of 139)
    K_lo = max(1, best_K - 2)
    K_hi = best_K + 2
    K_range = range(K_lo, K_hi + 1)

    # Select random stores for each sub-scenario (different seeds)
    stores_10 = _select_random_stores(num_stores, half, seed=100)
    stores_20 = _select_random_stores(num_stores, half, seed=200)

    scenarios = [
        (f"+10% ({len(stores_10)} CH)",
         _perturb_demands(baseline, stores_10, 1.10), 10),
        (f"+20% ({len(stores_20)} CH)",
         _perturb_demands(baseline, stores_20, 1.20), 20),
    ]

    return _print_scenario_table(
        title="CASE 2: POSITIVE RANDOM DEMAND SENSITIVITY",
        description=f"Increase demand of RANDOM subset (< 1/2 = {half} stores) by +10%, +20%",
        data=data,
        baseline_demands=baseline,
        scenarios=scenarios,
        K_range=K_range,
        num_stores=num_stores,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  CASE 3: -10%, -20% random demand một số stores (< 1/2 tổng CH)
# ═══════════════════════════════════════════════════════════════════════════════

def run_case3(data, best_K):
    """Case 3: -10%, -20% demand cho RANDOM subset stores (< 1/2 of 139)."""
    baseline = list(data["demands"])
    num_stores = len(baseline) - 1
    half = num_stores // 2   # 69 stores
    K_lo = max(1, best_K - 2)
    K_hi = best_K + 2
    K_range = range(K_lo, K_hi + 1)

    stores_10 = _select_random_stores(num_stores, half, seed=300)
    stores_20 = _select_random_stores(num_stores, half, seed=400)

    scenarios = [
        (f"-10% ({len(stores_10)} CH)",
         _perturb_demands(baseline, stores_10, 0.90), 30),
        (f"-20% ({len(stores_20)} CH)",
         _perturb_demands(baseline, stores_20, 0.80), 40),
    ]

    return _print_scenario_table(
        title="CASE 3: NEGATIVE RANDOM DEMAND SENSITIVITY",
        description=f"Decrease demand of RANDOM subset (< 1/2 = {half} stores) by -10%, -20%",
        data=data,
        baseline_demands=baseline,
        scenarios=scenarios,
        K_range=K_range,
        num_stores=num_stores,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  CASE 4: ±10%, ±20% random demand một số stores (> 1/2 tổng CH)
# ═══════════════════════════════════════════════════════════════════════════════

def run_case4(data, best_K):
    """
    Case 4: ±10%, ±20% demand cho RANDOM subset stores (> 1/2 of 139).
    Mỗi store trong subset được gán ngẫu nhiên + hoặc -.
    """
    baseline = list(data["demands"])
    num_stores = len(baseline) - 1
    # Slightly more than half: ~75-80 stores (>= 70 + a bit)
    count = int(num_stores * 0.55)  # ~76 stores — slightly > half
    K_lo = max(1, best_K - 2)
    K_hi = best_K + 2
    K_range = range(K_lo, K_hi + 1)

    stores_10 = _select_random_stores(num_stores, count, seed=500)
    stores_20 = _select_random_stores(num_stores, count, seed=600)

    scenarios = [
        (f"±10% ({len(stores_10)} CH)",
         _perturb_demands_mixed(baseline, stores_10, 0.10, seed=510), 50),
        (f"±20% ({len(stores_20)} CH)",
         _perturb_demands_mixed(baseline, stores_20, 0.20, seed=610), 60),
    ]

    return _print_scenario_table(
        title="CASE 4: MIXED RANDOM DEMAND SENSITIVITY",
        description=f"Randomly ±perturb demand of RANDOM subset (> 1/2 ≈ {count} stores) by ±10%, ±20%",
        data=data,
        baseline_demands=baseline,
        scenarios=scenarios,
        K_range=K_range,
        num_stores=num_stores,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  FLEET SWEEP – find best K (reused from vrp_alns_tabu_rrd.py logic)
# ═══════════════════════════════════════════════════════════════════════════════

def find_best_K(data, K_min=3, K_max=13):
    """Run fleet sweep to find best K by composite score."""
    import time as time_mod

    num_customers = len(data["distance_matrix"]) - 1
    print(f"\n{'═'*W}")
    print(f"  FLEET SWEEP — Finding optimal K  (K = {K_min}→{K_max})")
    print(f"{'═'*W}")

    summary = []
    for K in range(K_min, K_max + 1):
        t0 = time_mod.time()
        np.random.seed(SEED)
        random.seed(SEED)
        result = rrd_solve(
            data,
            num_vehicles=K,
            alns_iterations=ALNS_ITER,
            reassign_iter=REASSIGN_ITER,
            verbose=False,
        )
        comp_time = time_mod.time() - t0
        m = calc_metrics(result["rrd_routes"], data)
        summary.append({
            "K": K, "total_cost": result["rrd_cost"],
            "otdr": m["otdr"], "ces": m["ces"],
            "on_time": m["on_time"], "time": comp_time,
        })
        print(
            f"  K={K:>2}  cost={result['rrd_cost']:>15,.0f} VND  "
            f"OTDR={m['otdr']:>6.2f}%  CES={m['ces']:>6.2f}  "
            f"time={comp_time:>7.1f}s"
        )

    # Composite score
    costs = [r["total_cost"] for r in summary]
    otdrs = [r["otdr"]       for r in summary]
    ces_v = [r["ces"]        for r in summary]

    def _norm(vals, invert=False):
        lo, hi = min(vals), max(vals)
        span = hi - lo if hi > lo else 1.0
        if invert:
            return [(hi - v) / span for v in vals]
        return [(v - lo) / span for v in vals]

    nc  = _norm(costs)
    no  = _norm(otdrs, invert=True)
    nce = _norm(ces_v)
    for i, r in enumerate(summary):
        r["_score"] = 0.3 * nc[i] + 0.6 * no[i] + 0.1 * nce[i]

    best = min(summary, key=lambda x: x["_score"])
    print(
        f"\n  ★  Best K = {best['K']}  →  {best['total_cost']:,.0f} VND  |  "
        f"OTDR = {best['otdr']:.2f}%  |  Score = {best['_score']:.4f}"
    )
    print(f"{'═'*W}\n")
    return best["K"]


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="VRP T-ALNS-RRD Demand Sensitivity Analysis"
    )
    parser.add_argument(
        "--case", type=int, choices=[0, 1, 2, 3, 4], required=True,
        help="Which case to run: 0=All, 1=Global±, 2=+Random<half, "
             "3=-Random<half, 4=±Random>half",
    )
    parser.add_argument(
        "--best-k", type=int, default=0,
        help="Skip fleet sweep and use this K as baseline "
             "(0 = run fleet sweep to find best K)",
    )
    args = parser.parse_args()

    print(f"\n{'═'*W}")
    print(f"  VRP T-ALNS-RRD — Demand Sensitivity Experiments")
    print(f"{'═'*W}")

    print(f"\n  Loading data ...", file=sys.stderr, flush=True)
    data = load_data()
    num_stores = len(data["distance_matrix"]) - 1
    print(f"  Loaded {num_stores} stores + depot")
    print(f"  Capacity = {data['vehicle_capacity']} kg")
    print(f"  Total baseline demand = {sum(data['demands'][1:]):,.1f} kg")

    # Determine best K
    if args.best_k > 0:
        best_K = args.best_k
        print(f"\n  Using provided best K = {best_K}")
    else:
        best_K = find_best_K(data)

    # Run requested case(s)
    cases_to_run = [args.case] if args.case != 0 else [1, 2, 3, 4]

    for case_num in cases_to_run:
        print(f"\n{'▓'*W}")
        print(f"  RUNNING CASE {case_num}")
        print(f"{'▓'*W}")

        if case_num == 1:
            run_case1(data, best_K)
        elif case_num == 2:
            run_case2(data, best_K)
        elif case_num == 3:
            run_case3(data, best_K)
        elif case_num == 4:
            run_case4(data, best_K)

    print(f"\n  ✅  All requested cases complete.\n")


if __name__ == "__main__":
    main()
