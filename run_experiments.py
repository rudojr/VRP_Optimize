"""
run_experiments.py – Sensitivity Analysis for VRP T-ALNS-RRD

Cases:
  --case 1   ± 10%, 20% demand toàn bộ stores (global)
  --case 2   + 10%, 20% demand random một số stores (< 1/2 tổng 139 CH)
  --case 3   - 10%, 20% demand random một số stores (< 1/2 tổng 139 CH)
  --case 4   ± 10%, 20% demand random một số stores (> 1/2 tổng 139 CH)
  --case 5   ± 10%, 20% vehicle capacity (sức chứa xe)
  --case 6   ± 1, 2 số xe (fleet size K)
  --case 0   Chạy tất cả case 1-6

Usage:
    venv/bin/python run_experiments.py --case 1
    venv/bin/python run_experiments.py --case 0          # all cases
    venv/bin/python run_experiments.py --case 5          # capacity sensitivity
    venv/bin/python run_experiments.py --case 6          # fleet size sensitivity
    venv/bin/python run_experiments.py --case 1 --best-k 10
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
ALNS_ITER     = 50
REASSIGN_ITER = 10
SEED          = 42
W             = 115


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _run_single(data, demands_list, K, seed_offset=0):
    """
    Solve with given demands and a FIXED K. No fleet sweep.
    Returns dict with cost, otdr, ces, n_active.
    """
    original_demands = list(data["demands"])
    data["demands"] = list(demands_list)

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

    data["demands"] = original_demands
    return {
        "K": K, "cost": res["rrd_cost"], "otdr": m["otdr"],
        "ces": m["ces"], "n_active": n_active, "result": res,
    }


def _select_random_stores(num_stores, count, seed):
    """Return sorted list of random store indices (1-based) of given count."""
    rng = random.Random(seed)
    all_stores = list(range(1, num_stores + 1))
    return sorted(rng.sample(all_stores, min(count, num_stores)))


def _perturb_demands(baseline, store_indices, factor):
    """Perturb demands of specified stores by a single factor."""
    perturbed = list(baseline)
    for sid in store_indices:
        perturbed[sid] = baseline[sid] * factor
    return perturbed


def _perturb_demands_mixed(baseline, store_indices, pct_abs, seed):
    """Perturb demands of specified stores by ±pct_abs randomly."""
    rng = random.Random(seed)
    perturbed = list(baseline)
    for sid in store_indices:
        sign = rng.choice([1, -1])
        perturbed[sid] = baseline[sid] * (1 + sign * pct_abs)
    return perturbed


def _print_scenario_table(title, description, data, baseline_demands,
                          scenarios, alloc_K, num_stores):
    """
    Run each scenario with a FIXED K (alloc_K), report Active vehicles
    as the operationally needed fleet size.

    Giải thích cơ chế:
    - alloc_K (Allocated K): số xe TỐI ĐA mà solver được phân bổ.
    - Active: số xe THỰC SỰ được sử dụng (có chở hàng).
    - Util%: Active / alloc_K — tỷ lệ sử dụng đội xe.
    - Recommended K = Active + 1 (dự phòng 1 xe).

    scenarios: list of (label, demands_list, seed_offset, n_changed_stores)
    """
    total_base = sum(baseline_demands[1:])
    capacity = data["vehicle_capacity"]

    print(f"\n{'═'*W}")
    print(f"  {title}")
    print(f"  {description}")
    print(f"  {num_stores} stores  |  Allocated K = {alloc_K}  |  Capacity = {capacity} kg")
    print(f"  Total baseline demand = {total_base:,.1f} kg")
    print(f"{'─'*W}")
    print(f"  Cơ chế:")
    print(f"    • Allocated K = {alloc_K}: số xe tối đa solver được dùng (từ fleet sweep baseline)")
    print(f"    • Active: số xe THỰC SỰ dispatched (có ≥1 khách)")
    print(f"    • Util%: Active / Allocated K — hiệu suất sử dụng đội xe")
    print(f"    • Rec.K: Recommended fleet = Active + 1 (dự phòng)")
    print(f"{'═'*W}")

    # ── Baseline ──────────────────────────────────────────────────────────
    print(f"\n  ⏳  Running baseline ...", file=sys.stderr, flush=True)
    base = _run_single(data, baseline_demands, alloc_K, seed_offset=0)
    base_util = base["n_active"] / alloc_K * 100

    print(f"\n  Baseline (0%)")
    print(f"    Total demand : {total_base:>12,.1f} kg")
    print(f"    Allocated K  : {alloc_K:>12}")
    print(f"    Active veh.  : {base['n_active']:>12}")
    print(f"    Utilization  : {base_util:>11.1f}%")
    print(f"    Cost         : {base['cost']:>12,.0f} VND")
    print(f"    OTDR         : {base['otdr']:>11.2f}%")

    # ── Table header ──────────────────────────────────────────────────────
    print(f"\n  {'─'*107}")
    print(
        f"  {'Scenario':<18} {'Demand':<12} {'#CH':<6} "
        f"{'Active':<8} {'Util%':<7} {'Rec.K':<7} "
        f"{'Cost (VND)':<16} {'Δ Cost':<14} {'Δ%':<9} "
        f"{'OTDR':<8} {'ΔOTDR':<7}"
    )
    print(
        f"  {'─'*18} {'─'*12} {'─'*6} "
        f"{'─'*8} {'─'*7} {'─'*7} "
        f"{'─'*16} {'─'*14} {'─'*9} "
        f"{'─'*8} {'─'*7}"
    )

    results = []
    for label, demands_list, seed_offset, n_changed in scenarios:
        print(f"  ⏳  Running {label} ...", file=sys.stderr, flush=True)
        s = _run_single(data, demands_list, alloc_K, seed_offset=seed_offset)

        tot_d      = sum(demands_list[1:])
        delta      = s["cost"] - base["cost"]
        delta_pct  = delta / base["cost"] * 100 if base["cost"] > 0 else 0
        delta_otdr = s["otdr"] - base["otdr"]
        util_pct   = s["n_active"] / alloc_K * 100
        rec_k      = min(s["n_active"] + 1, alloc_K)

        cs = "+" if delta >= 0 else ""
        os_ = "+" if delta_otdr >= 0 else ""

        print(
            f"  {label:<18} {tot_d:<12,.1f} {n_changed:<6} "
            f"{s['n_active']:<8} {util_pct:<6.0f}% {rec_k:<7} "
            f"{s['cost']:<16,.0f} {cs}{delta:<13,.0f} {cs}{delta_pct:<8.2f}% "
            f"{s['otdr']:<7.2f}% {os_}{delta_otdr:<6.2f}%"
        )

        results.append({
            "label": label, "tot_demand": tot_d, "n_changed": n_changed,
            "n_active": s["n_active"], "util_pct": util_pct, "rec_k": rec_k,
            "cost": s["cost"], "delta": delta, "delta_pct": delta_pct,
            "otdr": s["otdr"], "delta_otdr": delta_otdr,
        })

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n  {'─'*107}")
    print(f"  SENSITIVITY SUMMARY")
    print(f"  {'─'*107}")

    inc = [r for r in results if r["delta_pct"] > 0.01]
    dec = [r for r in results if r["delta_pct"] < -0.01]

    if inc:
        avg = sum(r["delta_pct"] for r in inc) / len(inc)
        print(f"  Avg cost Δ (demand increase): +{avg:.2f}%")
    if dec:
        avg = sum(r["delta_pct"] for r in dec) / len(dec)
        print(f"  Avg cost Δ (demand decrease): {avg:.2f}%")

    max_imp = max(results, key=lambda r: abs(r["delta_pct"]))
    print(
        f"  Most impactful : {max_imp['label']}  →  "
        f"Δcost={'+' if max_imp['delta_pct']>=0 else ''}{max_imp['delta_pct']:.2f}%"
        f"  |  OTDR={max_imp['otdr']:.2f}%"
    )

    # Fleet utilization insights
    min_active = min(r["n_active"] for r in results)
    max_active = max(r["n_active"] for r in results)
    print(f"\n  Fleet insights (Allocated K = {alloc_K}):")
    print(f"    Active vehicles range : {min_active} – {max_active}")
    if min_active < alloc_K:
        print(
            f"    → Khi demand giảm, chỉ cần {min_active} xe "
            f"(tiết kiệm {alloc_K - min_active} xe so với allocation)"
        )
    if max_active >= alloc_K:
        print(
            f"    → Khi demand tăng, cần tất cả {alloc_K} xe. "
            f"Có thể cần thêm xe nếu demand tăng hơn nữa."
        )

    otdr_drops = [r for r in results if r["delta_otdr"] < -0.01]
    if otdr_drops:
        for r in otdr_drops:
            print(
                f"    ⚠ {r['label']}: OTDR giảm {r['delta_otdr']:.2f}% "
                f"→ cần thêm xe hoặc tối ưu lộ trình"
            )
    else:
        print(f"    ✓ OTDR ổn định ở tất cả scenarios")

    print(f"{'═'*W}")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  CASE 1–4
# ═══════════════════════════════════════════════════════════════════════════════

def run_case1(data, best_K):
    """Case 1: ±10%, ±20% demand cho TOÀN BỘ stores."""
    bl = list(data["demands"])
    ns = len(bl) - 1
    scenarios = []
    for idx, (lbl, f) in enumerate([
        ("-20%", 0.80), ("-10%", 0.90), ("+10%", 1.10), ("+20%", 1.20),
    ]):
        p = list(bl)
        for sid in range(1, ns + 1):
            p[sid] = bl[sid] * f
        scenarios.append((f"Global {lbl}", p, idx + 1, ns))

    return _print_scenario_table(
        "CASE 1: GLOBAL DEMAND SENSITIVITY",
        "Perturb ALL stores' demand simultaneously by ±10%, ±20%",
        data, bl, scenarios, best_K, ns,
    )


def run_case2(data, best_K):
    """Case 2: +10%, +20% demand cho RANDOM subset stores (< 1/2)."""
    bl = list(data["demands"])
    ns = len(bl) - 1
    half = ns // 2
    s10 = _select_random_stores(ns, half, seed=100)
    s20 = _select_random_stores(ns, half, seed=200)
    scenarios = [
        (f"+10% rnd ({len(s10)}CH)", _perturb_demands(bl, s10, 1.10), 10, len(s10)),
        (f"+20% rnd ({len(s20)}CH)", _perturb_demands(bl, s20, 1.20), 20, len(s20)),
    ]
    return _print_scenario_table(
        "CASE 2: POSITIVE RANDOM DEMAND SENSITIVITY",
        f"Increase demand of RANDOM subset (< 1/2 = {half} stores) by +10%, +20%",
        data, bl, scenarios, best_K, ns,
    )


def run_case3(data, best_K):
    """Case 3: -10%, -20% demand cho RANDOM subset stores (< 1/2)."""
    bl = list(data["demands"])
    ns = len(bl) - 1
    half = ns // 2
    s10 = _select_random_stores(ns, half, seed=300)
    s20 = _select_random_stores(ns, half, seed=400)
    scenarios = [
        (f"-10% rnd ({len(s10)}CH)", _perturb_demands(bl, s10, 0.90), 30, len(s10)),
        (f"-20% rnd ({len(s20)}CH)", _perturb_demands(bl, s20, 0.80), 40, len(s20)),
    ]
    return _print_scenario_table(
        "CASE 3: NEGATIVE RANDOM DEMAND SENSITIVITY",
        f"Decrease demand of RANDOM subset (< 1/2 = {half} stores) by -10%, -20%",
        data, bl, scenarios, best_K, ns,
    )


def run_case4(data, best_K):
    """Case 4: ±10%, ±20% demand cho RANDOM subset stores (> 1/2)."""
    bl = list(data["demands"])
    ns = len(bl) - 1
    count = int(ns * 0.55)  # ~76 stores — slightly > half
    s10 = _select_random_stores(ns, count, seed=500)
    s20 = _select_random_stores(ns, count, seed=600)
    scenarios = [
        (f"±10% rnd ({len(s10)}CH)", _perturb_demands_mixed(bl, s10, 0.10, seed=510), 50, len(s10)),
        (f"±20% rnd ({len(s20)}CH)", _perturb_demands_mixed(bl, s20, 0.20, seed=610), 60, len(s20)),
    ]
    return _print_scenario_table(
        "CASE 4: MIXED RANDOM DEMAND SENSITIVITY",
        f"Randomly ±perturb demand of RANDOM subset (> 1/2 ≈ {count} stores) by ±10%, ±20%",
        data, bl, scenarios, best_K, ns,
    )


def run_case5(data, best_K):
    """Case 5: ±10%, ±20% vehicle capacity.

    Giữ nguyên demand và K, thay đổi sức chứa xe.
    Đánh giá tác động lên cost, OTDR, số xe active.
    """
    bl = list(data["demands"])
    ns = len(bl) - 1
    base_cap = data["vehicle_capacity"]
    total_base = sum(bl[1:])

    cap_scenarios = [
        ("-20%", 0.80), ("-10%", 0.90), ("+10%", 1.10), ("+20%", 1.20),
    ]

    print(f"\n{'═'*W}")
    print(f"  CASE 5: VEHICLE CAPACITY SENSITIVITY")
    print(f"  Perturb vehicle capacity by ±10%, ±20% (baseline = {base_cap} kg)")
    print(f"  {ns} stores  |  Allocated K = {best_K}  |  Total demand = {total_base:,.1f} kg")
    print(f"{'─'*W}")
    print(f"  Cơ chế:")
    print(f"    • Baseline capacity = {base_cap} kg")
    print(f"    • Giữ nguyên demand và K = {best_K}")
    print(f"    • Thay đổi capacity → ảnh hưởng load distribution, routing, cost")
    print(f"    • Min trips needed = ceil(total_demand / capacity)")
    print(f"{'═'*W}")

    # ── Baseline ──────────────────────────────────────────────────────────
    print(f"\n  ⏳  Running baseline ...", file=sys.stderr, flush=True)
    base = _run_single(data, bl, best_K, seed_offset=0)
    base_util = base["n_active"] / best_K * 100
    min_trips_base = math.ceil(total_base / base_cap)

    print(f"\n  Baseline (Capacity = {base_cap} kg)")
    print(f"    Total demand : {total_base:>12,.1f} kg")
    print(f"    Capacity     : {base_cap:>12} kg")
    print(f"    Min trips    : {min_trips_base:>12}")
    print(f"    Allocated K  : {best_K:>12}")
    print(f"    Active veh.  : {base['n_active']:>12}")
    print(f"    Utilization  : {base_util:>11.1f}%")
    print(f"    Cost         : {base['cost']:>12,.0f} VND")
    print(f"    OTDR         : {base['otdr']:>11.2f}%")

    # ── Table header ──────────────────────────────────────────────────────
    print(f"\n  {'─'*107}")
    print(
        f"  {'Scenario':<18} {'Capacity':<12} {'MinTrips':<10} "
        f"{'Active':<8} {'Util%':<7} "
        f"{'Cost (VND)':<16} {'Δ Cost':<14} {'Δ%':<9} "
        f"{'OTDR':<8} {'ΔOTDR':<7}"
    )
    print(
        f"  {'─'*18} {'─'*12} {'─'*10} "
        f"{'─'*8} {'─'*7} "
        f"{'─'*16} {'─'*14} {'─'*9} "
        f"{'─'*8} {'─'*7}"
    )

    results = []
    for idx, (lbl, factor) in enumerate(cap_scenarios):
        new_cap = int(base_cap * factor)
        label = f"Cap {lbl} ({new_cap}kg)"
        print(f"  ⏳  Running {label} ...", file=sys.stderr, flush=True)

        # Temporarily change capacity
        data["vehicle_capacity"] = new_cap
        s = _run_single(data, bl, best_K, seed_offset=idx + 100)
        data["vehicle_capacity"] = base_cap  # restore

        min_trips = math.ceil(total_base / new_cap)
        delta      = s["cost"] - base["cost"]
        delta_pct  = delta / base["cost"] * 100 if base["cost"] > 0 else 0
        delta_otdr = s["otdr"] - base["otdr"]
        util_pct   = s["n_active"] / best_K * 100

        cs = "+" if delta >= 0 else ""
        os_ = "+" if delta_otdr >= 0 else ""

        print(
            f"  {label:<18} {new_cap:<12,} {min_trips:<10} "
            f"{s['n_active']:<8} {util_pct:<6.0f}% "
            f"{s['cost']:<16,.0f} {cs}{delta:<13,.0f} {cs}{delta_pct:<8.2f}% "
            f"{s['otdr']:<7.2f}% {os_}{delta_otdr:<6.2f}%"
        )

        results.append({
            "label": label, "capacity": new_cap, "min_trips": min_trips,
            "n_active": s["n_active"], "util_pct": util_pct,
            "cost": s["cost"], "delta": delta, "delta_pct": delta_pct,
            "otdr": s["otdr"], "delta_otdr": delta_otdr,
        })

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n  {'─'*107}")
    print(f"  CAPACITY SENSITIVITY SUMMARY")
    print(f"  {'─'*107}")

    dec_cap = [r for r in results if r["capacity"] < base_cap]
    inc_cap = [r for r in results if r["capacity"] > base_cap]

    if dec_cap:
        avg_cost = sum(r["delta_pct"] for r in dec_cap) / len(dec_cap)
        avg_otdr = sum(r["delta_otdr"] for r in dec_cap) / len(dec_cap)
        print(f"  Avg impact (capacity decrease): cost {'+' if avg_cost >= 0 else ''}{avg_cost:.2f}%  |  OTDR {'+' if avg_otdr >= 0 else ''}{avg_otdr:.2f}%")
    if inc_cap:
        avg_cost = sum(r["delta_pct"] for r in inc_cap) / len(inc_cap)
        avg_otdr = sum(r["delta_otdr"] for r in inc_cap) / len(inc_cap)
        print(f"  Avg impact (capacity increase): cost {'+' if avg_cost >= 0 else ''}{avg_cost:.2f}%  |  OTDR {'+' if avg_otdr >= 0 else ''}{avg_otdr:.2f}%")

    max_imp = max(results, key=lambda r: abs(r["delta_pct"]))
    print(
        f"  Most impactful : {max_imp['label']}  →  "
        f"Δcost={'+' if max_imp['delta_pct']>=0 else ''}{max_imp['delta_pct']:.2f}%"
        f"  |  OTDR={max_imp['otdr']:.2f}%"
    )

    # Fleet insights
    min_active = min(r["n_active"] for r in results)
    max_active = max(r["n_active"] for r in results)
    print(f"\n  Fleet insights (Allocated K = {best_K}):")
    print(f"    Active vehicles range : {min_active} – {max_active}")
    if any(r["min_trips"] > best_K for r in results):
        problematic = [r for r in results if r["min_trips"] > best_K]
        for r in problematic:
            print(
                f"    ⚠ {r['label']}: cần tối thiểu {r['min_trips']} chuyến "
                f"nhưng chỉ có {best_K} xe → khả năng quá tải!"
            )

    otdr_drops = [r for r in results if r["delta_otdr"] < -0.01]
    if otdr_drops:
        for r in otdr_drops:
            print(
                f"    ⚠ {r['label']}: OTDR giảm {r['delta_otdr']:.2f}% "
                f"→ capacity thấp hơn gây ảnh hưởng chất lượng giao hàng"
            )
    else:
        print(f"    ✓ OTDR ổn định ở tất cả scenarios")

    print(f"{'═'*W}")
    return results


def run_case6(data, best_K):
    """Case 6: ±1, ±2 fleet size (số xe).

    Giữ nguyên demand và capacity, thay đổi số xe K.
    Đánh giá tác động lên cost, OTDR, utilization.
    """
    bl = list(data["demands"])
    ns = len(bl) - 1
    capacity = data["vehicle_capacity"]
    total_base = sum(bl[1:])

    # Tạo danh sách K cần test: best_K ± 1, ± 2 (loại K <= 0)
    k_deltas = [-2, -1, +1, +2]
    k_scenarios = [(delta, best_K + delta) for delta in k_deltas if best_K + delta >= 1]

    print(f"\n{'═'*W}")
    print(f"  CASE 6: FLEET SIZE (K) SENSITIVITY")
    print(f"  Perturb fleet size K by ±1, ±2 (baseline K = {best_K})")
    print(f"  {ns} stores  |  Capacity = {capacity} kg  |  Total demand = {total_base:,.1f} kg")
    print(f"{'─'*W}")
    print(f"  Cơ chế:")
    print(f"    • Baseline K = {best_K} (từ fleet sweep hoặc user-defined)")
    print(f"    • Giữ nguyên demand và capacity = {capacity} kg")
    print(f"    • Thay đổi K → ảnh hưởng load balancing, routing flexibility, cost")
    print(f"    • Ít xe hơn → mỗi xe chở nhiều hơn, có thể tăng delay")
    print(f"    • Nhiều xe hơn → linh hoạt hơn nhưng tốn chi phí vận hành")
    print(f"{'═'*W}")

    # ── Baseline ──────────────────────────────────────────────────────────
    print(f"\n  ⏳  Running baseline (K={best_K}) ...", file=sys.stderr, flush=True)
    base = _run_single(data, bl, best_K, seed_offset=0)
    base_util = base["n_active"] / best_K * 100

    print(f"\n  Baseline (K = {best_K})")
    print(f"    Total demand : {total_base:>12,.1f} kg")
    print(f"    Capacity     : {capacity:>12} kg")
    print(f"    Allocated K  : {best_K:>12}")
    print(f"    Active veh.  : {base['n_active']:>12}")
    print(f"    Utilization  : {base_util:>11.1f}%")
    print(f"    Cost         : {base['cost']:>12,.0f} VND")
    print(f"    OTDR         : {base['otdr']:>11.2f}%")

    # ── Table header ──────────────────────────────────────────────────────
    print(f"\n  {'─'*107}")
    print(
        f"  {'Scenario':<18} {'K':<6} {'ΔK':<6} "
        f"{'Active':<8} {'Util%':<7} "
        f"{'Cost (VND)':<16} {'Δ Cost':<14} {'Δ%':<9} "
        f"{'OTDR':<8} {'ΔOTDR':<7}"
    )
    print(
        f"  {'─'*18} {'─'*6} {'─'*6} "
        f"{'─'*8} {'─'*7} "
        f"{'─'*16} {'─'*14} {'─'*9} "
        f"{'─'*8} {'─'*7}"
    )

    results = []
    for idx, (delta_k, test_K) in enumerate(k_scenarios):
        sign_str = f"+{delta_k}" if delta_k > 0 else str(delta_k)
        label = f"K={test_K} ({sign_str})"
        print(f"  ⏳  Running {label} ...", file=sys.stderr, flush=True)

        s = _run_single(data, bl, test_K, seed_offset=idx + 200)

        delta      = s["cost"] - base["cost"]
        delta_pct  = delta / base["cost"] * 100 if base["cost"] > 0 else 0
        delta_otdr = s["otdr"] - base["otdr"]
        util_pct   = s["n_active"] / test_K * 100

        cs = "+" if delta >= 0 else ""
        os_ = "+" if delta_otdr >= 0 else ""

        print(
            f"  {label:<18} {test_K:<6} {sign_str:<6} "
            f"{s['n_active']:<8} {util_pct:<6.0f}% "
            f"{s['cost']:<16,.0f} {cs}{delta:<13,.0f} {cs}{delta_pct:<8.2f}% "
            f"{s['otdr']:<7.2f}% {os_}{delta_otdr:<6.2f}%"
        )

        results.append({
            "label": label, "K": test_K, "delta_k": delta_k,
            "n_active": s["n_active"], "util_pct": util_pct,
            "cost": s["cost"], "delta": delta, "delta_pct": delta_pct,
            "otdr": s["otdr"], "delta_otdr": delta_otdr,
        })

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n  {'─'*107}")
    print(f"  FLEET SIZE SENSITIVITY SUMMARY")
    print(f"  {'─'*107}")

    fewer = [r for r in results if r["delta_k"] < 0]
    more  = [r for r in results if r["delta_k"] > 0]

    if fewer:
        avg_cost = sum(r["delta_pct"] for r in fewer) / len(fewer)
        avg_otdr = sum(r["delta_otdr"] for r in fewer) / len(fewer)
        print(f"  Avg impact (fewer vehicles) : cost {'+' if avg_cost >= 0 else ''}{avg_cost:.2f}%  |  OTDR {'+' if avg_otdr >= 0 else ''}{avg_otdr:.2f}%")
    if more:
        avg_cost = sum(r["delta_pct"] for r in more) / len(more)
        avg_otdr = sum(r["delta_otdr"] for r in more) / len(more)
        print(f"  Avg impact (more vehicles)  : cost {'+' if avg_cost >= 0 else ''}{avg_cost:.2f}%  |  OTDR {'+' if avg_otdr >= 0 else ''}{avg_otdr:.2f}%")

    max_imp = max(results, key=lambda r: abs(r["delta_pct"]))
    print(
        f"  Most impactful : {max_imp['label']}  →  "
        f"Δcost={'+' if max_imp['delta_pct']>=0 else ''}{max_imp['delta_pct']:.2f}%"
        f"  |  OTDR={max_imp['otdr']:.2f}%"
    )

    # Optimal K recommendation
    all_options = results + [{"label": f"K={best_K} (base)", "K": best_K,
        "cost": base["cost"], "otdr": base["otdr"], "n_active": base["n_active"],
        "util_pct": base_util, "delta_pct": 0, "delta_otdr": 0}]

    # Best = highest OTDR, then lowest cost
    max_otdr_all = max(o["otdr"] for o in all_options)
    near_optimal = [o for o in all_options if o["otdr"] >= max_otdr_all - 0.5]
    recommended = min(near_optimal, key=lambda x: (x["cost"], x["K"]))

    print(f"\n  Fleet recommendation:")
    print(f"    Tested K range     : {min(r['K'] for r in results)} – {max(r['K'] for r in results)} (baseline = {best_K})")
    print(
        f"    ★ Recommended K = {recommended['K']}  →  "
        f"cost = {recommended['cost']:,.0f} VND  |  OTDR = {recommended['otdr']:.2f}%"
    )

    if recommended["K"] < best_K:
        saving = base["cost"] - recommended["cost"]
        print(
            f"    → Có thể GIẢM {best_K - recommended['K']} xe "
            f"(tiết kiệm {saving:,.0f} VND, {saving/base['cost']*100:.2f}%)"
        )
    elif recommended["K"] > best_K:
        extra_cost = recommended["cost"] - base["cost"]
        print(
            f"    → Nên THÊM {recommended['K'] - best_K} xe "
            f"(chi phí thêm {extra_cost:,.0f} VND nhưng OTDR tốt hơn)"
        )
    else:
        print(f"    → K = {best_K} hiện tại đã tối ưu")

    # OTDR drop warnings
    otdr_drops = [r for r in results if r["delta_otdr"] < -0.5]
    if otdr_drops:
        for r in otdr_drops:
            print(
                f"    ⚠ {r['label']}: OTDR giảm {r['delta_otdr']:.2f}% "
                f"→ giảm xe gây ảnh hưởng đáng kể đến chất lượng giao hàng"
            )
    else:
        print(f"    ✓ OTDR ổn định ở tất cả fleet sizes tested")

    print(f"{'═'*W}")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  FLEET SWEEP – find best K
# ═══════════════════════════════════════════════════════════════════════════════

def find_best_K(data, K_min=3, K_max=13):
    """
    Fleet sweep K_min→K_max.

    Chiến lược chọn Best K:
      1. Tìm max OTDR trong tất cả K
      2. Lọc các K có OTDR >= max_OTDR - 0.5%  (gần tối ưu)
      3. Trong nhóm đó, chọn K có cost THẤP NHẤT
      4. Tie-break: K nhỏ hơn (ít xe hơn = tiết kiệm hơn)

    Cách này đảm bảo Best K là SỐ XE NHỎ NHẤT mà vẫn giữ OTDR gần tối ưu,
    thay vì luôn chọn K cao nhất chỉ vì OTDR tốt hơn 0.01%.
    """
    import time as time_mod

    print(f"\n{'═'*W}")
    print(f"  FLEET SWEEP — Finding optimal K  (K = {K_min}→{K_max})")
    print(f"  Strategy: Smallest K with near-optimal OTDR (within 0.5%)")
    print(f"{'═'*W}")

    summary = []
    for K in range(K_min, K_max + 1):
        t0 = time_mod.time()
        np.random.seed(SEED)
        random.seed(SEED)
        result = rrd_solve(
            data, num_vehicles=K,
            alns_iterations=ALNS_ITER, reassign_iter=REASSIGN_ITER,
            verbose=False,
        )
        ct = time_mod.time() - t0
        m = calc_metrics(result["rrd_routes"], data)
        n_active = sum(1 for r in result["rrd_routes"] if len(r["nodes"]) > 2)
        summary.append({
            "K": K, "total_cost": result["rrd_cost"],
            "otdr": m["otdr"], "ces": m["ces"],
            "n_active": n_active, "time": ct,
        })
        util = n_active / K * 100
        print(
            f"  K={K:>2}  Active={n_active:>2}  Util={util:>5.0f}%  "
            f"cost={result['rrd_cost']:>15,.0f} VND  "
            f"OTDR={m['otdr']:>6.2f}%  CES={m['ces']:>6.2f}  "
            f"time={ct:>7.1f}s"
        )

    # Strategy: smallest K with near-optimal OTDR, then lowest cost
    max_otdr = max(s["otdr"] for s in summary)
    threshold = max_otdr - 0.5  # within 0.5% of best

    candidates = [s for s in summary if s["otdr"] >= threshold]
    if not candidates:
        candidates = summary

    # Among candidates, pick lowest cost; tie-break: lower K
    best = min(candidates, key=lambda x: (x["total_cost"], x["K"]))

    print(f"\n  Max OTDR across sweep = {max_otdr:.2f}%")
    print(f"  Threshold = {threshold:.2f}% (max - 0.5%)")
    print(f"  Candidates (OTDR >= {threshold:.2f}%): "
          f"K = {[s['K'] for s in candidates]}")
    print(
        f"\n  ★  Best K = {best['K']}  (Active={best['n_active']})  →  "
        f"{best['total_cost']:,.0f} VND  |  "
        f"OTDR = {best['otdr']:.2f}%"
    )
    print(f"{'═'*W}\n")
    return best["K"]


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="VRP T-ALNS-RRD Sensitivity Analysis"
    )
    parser.add_argument(
        "--case", type=int, choices=[0, 1, 2, 3, 4, 5, 6], required=True,
        help="0=All, 1=Global±, 2=+Random<half, 3=-Random<half, 4=±Random>half, 5=Capacity±, 6=FleetK±",
    )
    parser.add_argument(
        "--best-k", type=int, default=0,
        help="Skip fleet sweep and use this K (0 = auto fleet sweep)",
    )
    args = parser.parse_args()

    print(f"\n{'═'*W}")
    print(f"  VRP T-ALNS-RRD — Sensitivity Experiments")
    print(f"{'═'*W}")

    print(f"\n  Loading data ...", file=sys.stderr, flush=True)
    data = load_data()
    ns = len(data["distance_matrix"]) - 1
    print(f"  {ns} stores + depot  |  Cap = {data['vehicle_capacity']} kg"
          f"  |  Baseline demand = {sum(data['demands'][1:]):,.1f} kg")

    if args.best_k > 0:
        best_K = args.best_k
        print(f"\n  Using provided best K = {best_K}")
    else:
        best_K = find_best_K(data)

    case_funcs = {
        1: run_case1, 2: run_case2, 3: run_case3,
        4: run_case4, 5: run_case5, 6: run_case6,
    }
    cases = [args.case] if args.case != 0 else [1, 2, 3, 4, 5, 6]
    for c in cases:
        print(f"\n{'▓'*W}")
        print(f"  RUNNING CASE {c}")
        print(f"{'▓'*W}")
        case_funcs[c](data, best_K)

    print(f"\n  ✅  All requested cases complete.\n")


if __name__ == "__main__":
    main()
