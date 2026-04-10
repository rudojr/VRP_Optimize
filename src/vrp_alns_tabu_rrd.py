import copy
import math
import time as time_mod
from typing import List, Tuple, Dict, Optional

import numpy as np

# ── Re-use all infrastructure from vrp_alns_tabu ─────────────────────────────
from vrp_alns_tabu import (
    load_data,
    CostEvaluator,
    Route,
    Solution,
    alns_tabu_solve,
    local_search_2opt,
    print_solution,
    CONGESTION_LABELS,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  BASE POLICY  (rollout completion heuristic)
# ═══════════════════════════════════════════════════════════════════════════════

class _GreedyTWPolicy:
    """
    Greedy nearest-neighbor với time-window urgency priority.
    Dùng làm base policy trong rollout simulation.

    Priority: min(tw_end * 1e4 + distance + lateness * 1e6)
    → Ưu tiên earliest deadline + gần + không trễ
    """

    def __init__(self, evaluator: CostEvaluator):
        self.ev     = evaluator
        self._tw    = evaluator.tw
        self._dist  = evaluator.dist
        self._tt    = evaluator.travel_time   # đã nhân congestion factor
        self._cong  = evaluator.cong
        self._lam1  = evaluator.lam1
        self._lam2  = evaluator.lam2
        self._cpk   = evaluator.cpk
        self._s     = evaluator.s_time
        self._depot = evaluator.depot

    def simulate(
        self,
        from_node: int,
        from_time: float,
        unvisited: List[int],
    ) -> float:
        """
        Simulate completing the route greedily from (from_node, from_time).
        Returns estimated remaining cost (transport + late_penalty + congestion).
        """
        current_node = from_node
        current_time = from_time
        remaining    = list(unvisited)
        total_cost   = 0.0

        while remaining:
            best_next  = None
            best_score = float("inf")

            for candidate in remaining:
                tw_end   = float(self._tw[candidate][1])
                tw_start = float(self._tw[candidate][0])
                tt_seg   = self._tt[current_node][candidate]
                arrive   = current_time + self._s + tt_seg
                arrive   = max(arrive, tw_start)
                lateness = max(0.0, arrive - tw_end)

                score = (
                    tw_end * 1e4
                    + self._dist[current_node][candidate]
                    + lateness * 1e6
                )
                if score < best_score:
                    best_score = score
                    best_next  = candidate

            # Commit step
            cong_next = self._cong[current_node][best_next]
            tt_next   = self._tt[current_node][best_next]
            arrive    = current_time + self._s + tt_next
            arrive    = max(arrive, float(self._tw[best_next][0]))
            lateness  = max(0.0, arrive - float(self._tw[best_next][1]))

            total_cost += (
                self._dist[current_node][best_next] * self._cpk
                + self._lam2 * cong_next
                + self._lam1 * lateness
            )

            current_node = best_next
            current_time = arrive
            remaining.remove(best_next)

        # Return to depot
        cong_depot = self._cong[current_node][self._depot]
        dist_depot = self._dist[current_node][self._depot]
        total_cost += dist_depot * self._cpk + self._lam2 * cong_depot

        return total_cost


# ═══════════════════════════════════════════════════════════════════════════════
#  ROLLOUT DISPATCHER  (Phase 2b — Intra-route resequencing)
# ═══════════════════════════════════════════════════════════════════════════════

class RolloutDispatcher:
    """
    Với mỗi route, sắp xếp lại thứ tự visit tối ưu qua 1-step lookahead rollout.
    """

    def __init__(self, data: dict, evaluator: CostEvaluator):
        self.data   = data
        self.ev     = evaluator
        self.policy = _GreedyTWPolicy(evaluator)
        self.depot  = data["depot"]

    def rollout_next_stop(
        self,
        from_node: int,
        from_time: float,
        unvisited: List[int],
    ) -> Tuple[int, float]:
        """1-step lookahead: chọn next customer tối thiểu hóa step_cost + rollout_completion."""
        best_next = None
        best_cost = float("inf")

        for candidate in unvisited:
            tt_to    = self.ev.travel_time[from_node][candidate]
            cong_to  = self.ev.cong[from_node][candidate]
            service  = self.ev.s_time if from_node != self.depot else 0
            arrive   = from_time + service + tt_to
            arrive   = max(arrive, float(self.ev.tw[candidate][0]))
            lateness = max(0.0, arrive - float(self.ev.tw[candidate][1]))

            step_cost = (
                self.ev.dist[from_node][candidate] * self.ev.cpk
                + self.ev.lam2 * cong_to
                + self.ev.lam1 * lateness
            )

            remaining       = [c for c in unvisited if c != candidate]
            completion_cost = self.policy.simulate(candidate, arrive, remaining)
            total           = step_cost + completion_cost

            if total < best_cost:
                best_cost = total
                best_next = candidate

        return best_next, best_cost

    def dispatch_route(self, assigned_customers: List[int]) -> Tuple[Route, List[dict]]:
        """
        Given customers, return Route với rollout-optimised visit sequence.
        """
        if not assigned_customers:
            return Route(), []

        current_node = self.depot
        current_time = float(self.ev.tw[self.depot][0])
        unvisited    = list(assigned_customers)
        sequence     = []
        timing_log   = [{"node": self.depot, "arrival": current_time, "lateness": 0.0}]

        while unvisited:
            next_stop, sim_cost = self.rollout_next_stop(current_node, current_time, unvisited)

            tt_to    = self.ev.travel_time[current_node][next_stop]
            cong_to  = self.ev.cong[current_node][next_stop]
            service  = self.ev.s_time if current_node != self.depot else 0
            arrive   = current_time + service + tt_to
            arrive   = max(arrive, float(self.ev.tw[next_stop][0]))
            lateness = max(0.0, arrive - float(self.ev.tw[next_stop][1]))

            timing_log.append({
                "node":     next_stop,
                "arrival":  arrive,
                "lateness": lateness,
                "sim_cost": sim_cost,
            })

            sequence.append(next_stop)
            current_node = next_stop
            current_time = arrive
            unvisited.remove(next_stop)

        return Route(customers=sequence), timing_log

    def dispatch_all(self, solution: Solution) -> Tuple[Solution, list]:
        """Apply rollout resequencing to every route."""
        new_routes  = []
        timing_logs = []

        for route in solution.routes:
            new_route, t_log = self.dispatch_route(route.customers)
            new_routes.append(new_route)
            timing_logs.append(t_log)

        new_sol = Solution(routes=new_routes)
        self.ev.evaluate_solution(new_sol)
        return new_sol, timing_logs


# ═══════════════════════════════════════════════════════════════════════════════
#  ROLLOUT REASSIGNER  (Phase 2a — Inter-route customer reassignment)
# ═══════════════════════════════════════════════════════════════════════════════

class RolloutReassigner:
    """
    Inter-route customer reassignment với EXACT cost evaluation + rollout sequencing.

    Thuật toán:
    -----------
    for iteration in range(max_iter):
        improved = False
        for each customer c in each route s:
            cost_s_cur = exact_cost(route_s)
            cost_s_new = exact_cost(route_s - {c})  # sau khi xóa c
            for each target route t != s:
                if capacity OK:
                    best_insert_pos = rollout chọn vị trí chèn tốt nhất
                    cost_t_new = exact_cost(route_t + {c} at best_insert_pos)
                    delta = (cost_s_new + cost_t_new) - (cost_s_cur + cost_t_cur)
                    if delta < -eps → accept move
        if not improved → break

    Key insight: Dùng EXACT evaluation để tránh sai lệch từ greedy approximation.
    Rollout chỉ được dùng để TÌM vị trí chèn tốt hơn (không phải để evaluate delta).
    """

    def __init__(self, evaluator: CostEvaluator, max_iter: int = 10):
        self.ev       = evaluator
        self.max_iter = max_iter

    def _exact_cost(self, customers: List[int]) -> float:
        """Exact route cost evaluation."""
        if not customers:
            return 0.0
        return self.ev.evaluate_route(Route(customers=customers))["total"]

    def _best_insert_position(self, customers: List[int], new_cust: int) -> Tuple[int, float]:
        """
        Tìm vị trí chèn tốt nhất cho new_cust vào route [customers] bằng exact evaluation.
        Returns (best_pos, best_cost).
        """
        best_cost = float("inf")
        best_pos  = 0
        for pos in range(len(customers) + 1):
            trial = customers[:pos] + [new_cust] + customers[pos:]
            c     = self._exact_cost(trial)
            if c < best_cost:
                best_cost = c
                best_pos  = pos
        return best_pos, best_cost

    def optimize(self, solution: Solution, verbose: bool = False) -> Solution:
        """
        Iterative inter-route reassignment.
        Returns improved Solution.
        """
        # Deep copy
        sol = Solution(routes=[Route(customers=r.customers[:]) for r in solution.routes])
        self.ev.evaluate_solution(sol)

        eps        = 0.5   # min improvement threshold (VND)
        num_routes = len(sol.routes)

        for iteration in range(self.max_iter):
            improved   = False
            best_delta = 0.0
            best_move  = None  # (cust, s_idx, c_pos, t_idx, ins_pos)

            # Tìm move tốt nhất trong tất cả các cặp (customer, target_route)
            for s_idx in range(num_routes):
                custs_s = sol.routes[s_idx].customers
                if len(custs_s) == 0:
                    continue

                cost_s_cur = self._exact_cost(custs_s)

                for c_pos, cust in enumerate(custs_s):
                    src_without = [x for x in custs_s if x != cust]
                    cost_s_new  = self._exact_cost(src_without)

                    saving_from_src = cost_s_cur - cost_s_new  # >0 nếu xóa c giúp route s

                    for t_idx in range(num_routes):
                        if t_idx == s_idx:
                            continue

                        custs_t = sol.routes[t_idx].customers

                        # Kiểm tra capacity
                        new_load = (
                            self.ev.route_load(sol.routes[t_idx])
                            + self.ev.demands[cust]
                        )
                        if new_load > self.ev.Q:
                            continue

                        cost_t_cur = self._exact_cost(custs_t)

                        # Tìm vị trí chèn tốt nhất
                        ins_pos, cost_t_new = self._best_insert_position(custs_t, cust)

                        delta = (cost_s_new + cost_t_new) - (cost_s_cur + cost_t_cur)

                        if delta < best_delta - eps:
                            best_delta = delta
                            best_move  = (cust, s_idx, c_pos, t_idx, ins_pos)

            # Apply best move nếu có
            if best_move is not None:
                cust, s_idx, c_pos, t_idx, ins_pos = best_move
                custs_s = sol.routes[s_idx].customers
                custs_t = sol.routes[t_idx].customers

                # Remove from source
                sol.routes[s_idx].customers = [x for x in custs_s if x != cust]
                # Insert into target
                sol.routes[t_idx].customers = (
                    custs_t[:ins_pos] + [cust] + custs_t[ins_pos:]
                )
                self.ev.evaluate_solution(sol)
                improved = True

                if verbose:
                    print(f"    [Reassign iter {iteration+1}] "
                          f"cust {cust}: route {s_idx}→{t_idx}, "
                          f"delta={best_delta:,.0f} VND, "
                          f"new total={sol.cost:,.0f}")

            if not improved:
                if verbose:
                    print(f"    [Reassign] Converged after {iteration+1} iteration(s)")
                break

        self.ev.evaluate_solution(sol)
        return sol


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN SOLVER — T-ALNS-RRD v2
# ═══════════════════════════════════════════════════════════════════════════════

def rrd_solve(
    data: dict,
    num_vehicles: int,
    alns_iterations: int = 5,
    reassign_iter: int = 10,
    verbose: bool = True,
) -> dict:

    evaluator = CostEvaluator(data)

    if verbose:
        print(f"\n{'═'*70}")
        print(f"  Phase 1 — ALNS+Tabu Search  (K={num_vehicles}, {alns_iterations} iters)")
        print(f"{'═'*70}")

    alns_start = time_mod.time()
    alns_status, alns_total_cost, alns_routes_info, alns_solve_time = alns_tabu_solve(
        data,
        num_vehicles,
        max_iterations=alns_iterations,
        seed=42,
        verbose=verbose,
    )
    alns_elapsed = time_mod.time() - alns_start

   
    active_routes = [
        Route(customers=list(r["nodes"][1:-1]))  
        for r in alns_routes_info
    ]
    while len(active_routes) < num_vehicles:
        active_routes.append(Route())

    alns_sol = Solution(routes=active_routes)
    evaluator.evaluate_solution(alns_sol)

    if verbose:
        print(f"\n  ✓ ALNS+Tabu complete — cost = {alns_total_cost:,.0f} VND  ({alns_elapsed:.1f}s)")
        print(f"    Active routes: {len(alns_routes_info)}, customers: "
              f"{sum(len(r.customers) for r in alns_sol.routes if r.customers)}")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 2a — Rollout-guided Inter-Route Reassignment
    # ══════════════════════════════════════════════════════════════════════
    if verbose:
        print(f"\n{'═'*70}")
        print(f"  Phase 2a — Rollout-guided Inter-Route Reassignment")
        print(f"  (Exact cost eval, best-move search, max {reassign_iter} iters)")
        print(f"{'═'*70}")

    p2a_start    = time_mod.time()
    reassigner   = RolloutReassigner(evaluator, max_iter=reassign_iter)
    reassigned_sol = reassigner.optimize(alns_sol, verbose=verbose)
    p2a_time     = time_mod.time() - p2a_start

    if verbose:
        print(f"\n  ✓ Phase 2a complete — cost = {reassigned_sol.cost:,.0f} VND  ({p2a_time:.1f}s)")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 2b — Rollout Intra-Route Resequencing
    # ══════════════════════════════════════════════════════════════════════
    if verbose:
        print(f"\n{'═'*70}")
        print(f"  Phase 2b — Rollout Intra-Route Resequencing (1-step lookahead)")
        print(f"{'═'*70}")

    p2b_start   = time_mod.time()
    dispatcher  = RolloutDispatcher(data, evaluator)
    reseq_sol, timing_logs = dispatcher.dispatch_all(reassigned_sol)
    p2b_time    = time_mod.time() - p2b_start

    if verbose:
        print(f"\n  ✓ Phase 2b complete — cost = {reseq_sol.cost:,.0f} VND  ({p2b_time:.1f}s)")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 3 — 2-opt Polish
    # ══════════════════════════════════════════════════════════════════════
    if verbose:
        print(f"\n{'═'*70}")
        print(f"  Phase 3 — 2-opt Polish")
        print(f"{'═'*70}")

    p3_start  = time_mod.time()
    final_sol = local_search_2opt(reseq_sol, evaluator, max_no_improve=50)
    p3_time   = time_mod.time() - p3_start

    if verbose:
        print(f"\n  ✓ Phase 3 complete — cost = {final_sol.cost:,.0f} VND  ({p3_time:.1f}s)")

    rrd_cost       = final_sol.cost
    rrd_total_time = p2a_time + p2b_time + p3_time

    # ── Build routes_info cho output ─────────────────────────────────────
    rrd_routes_info = []
    for k, route in enumerate(final_sol.routes):
        if not route.customers:
            continue
        info  = evaluator.evaluate_route(route)
        nodes = [data["depot"]] + route.customers + [data["depot"]]
        rrd_routes_info.append({
            "vehicle":         k,
            "nodes":           nodes,
            "distance_km":     info["distance"],
            "load_kg":         info["load"],
            "transport_cost":  info["transport"],
            "late_penalty":    info["late_penalty"],
            "congestion_cost": info["congestion"],
            "total_cost":      info["total"],
            "timing":          info["timing"],
        })

    improvement_pct = (
        (alns_total_cost - rrd_cost) / alns_total_cost * 100
        if alns_total_cost > 0 else 0
    )

    return {
        "alns_status":      alns_status,
        "alns_cost":        alns_total_cost,
        "alns_routes":      alns_routes_info,
        "alns_time":        alns_elapsed,
        "rrd_cost":         rrd_cost,
        "rrd_routes":       rrd_routes_info,
        "rrd_time":         rrd_total_time,
        "improvement_pct":  improvement_pct,
        "timing_logs":      timing_logs,
        "phase_times": {
            "p1_alns":       alns_elapsed,
            "p2a_reassign":  p2a_time,
            "p2b_reseq":     p2b_time,
            "p3_2opt":       p3_time,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  COMPARISON OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════

def compare_and_print(data: dict, num_vehicles: int, result: dict):
    """Print side-by-side comparison: ALNS+Tabu vs T-ALNS-RRD v2"""
    W = 96
    print(f"\n{'═'*W}")
    print(f"  COMPARISON — ALNS+Tabu  vs  T-ALNS-RRD v2  (K={num_vehicles})")
    print(f"{'═'*W}")

    hdr = f"  {'Metric':<34} {'ALNS+Tabu':>18}  {'T-ALNS-RRD v2':>18}  {'Δ (RRD−ALNS)':>14}  {'%':>8}"
    print(hdr)
    print(f"  {'─'*34} {'─'*18}  {'─'*18}  {'─'*14}  {'─'*8}")

    def route_metrics(routes_info):
        total_c   = sum(r["total_cost"]      for r in routes_info)
        transport = sum(r["transport_cost"]  for r in routes_info)
        late      = sum(r["late_penalty"]    for r in routes_info)
        cong      = sum(r["congestion_cost"] for r in routes_info)
        n_cust    = sum(len(r["nodes"]) - 2  for r in routes_info)
        on_time   = sum(
            1 for r in routes_info
              for t in r["timing"]
              if t["node"] != data["depot"] and t["lateness"] <= 0.01
        )
        otdr = on_time / n_cust * 100 if n_cust else 0
        return total_c, transport, late, cong, otdr, len(routes_info)

    a_total, a_tr, a_late, a_cong, a_otdr, a_active = route_metrics(result["alns_routes"])
    r_total, r_tr, r_late, r_cong, r_otdr, r_active = route_metrics(result["rrd_routes"])

    def row(label, a, r, fmt="{:>18,.0f}"):
        if abs(a) < 1e-9:
            pct  = 0.0
            icon = "="
        else:
            pct  = (r - a) / a * 100
            icon = "✓" if (r - a) < -0.5 else ("=" if abs(r - a) < 0.5 else "✗")
        sign = "+" if (r - a) > 0 else ""
        print(
            f"  {label:<34} {fmt.format(a)}  {fmt.format(r)}  "
            f"{sign}{(r-a):>12,.0f}  {sign}{pct:>7.2f}% {icon}"
        )

    def row_pct(label, a, r):
        delta = r - a
        icon  = "✓" if delta > 0.01 else ("=" if abs(delta) < 0.01 else "✗")
        sign  = "+" if delta > 0 else ""
        print(
            f"  {label:<34} {a:>17.2f}%  {r:>17.2f}%  "
            f"{sign}{delta:>12.2f}%  {'':>8} {icon}"
        )

    row("Total cost (VND)",          a_total, r_total)
    row("  ↳ Transport cost",         a_tr,    r_tr)
    row("  ↳ Delay penalty cost",     a_late,  r_late)
    row("  ↳ Congestion cost",        a_cong,  r_cong)
    row_pct("OTDR (%)",               a_otdr,  r_otdr)
    row("Active vehicles",            a_active, r_active, fmt="{:>18,.0f}")

    print(f"  {'─'*34} {'─'*18}  {'─'*18}  {'─'*14}  {'─'*8}")

    # Phase timing breakdown
    pt = result.get("phase_times", {})
    print(f"\n  Timing breakdown:")
    print(f"    Phase 1  ALNS+Tabu      : {pt.get('p1_alns', result['alns_time']):>7.1f}s")
    print(f"    Phase 2a Reassignment   : {pt.get('p2a_reassign', 0):>7.1f}s")
    print(f"    Phase 2b Resequencing   : {pt.get('p2b_reseq', 0):>7.1f}s")
    print(f"    Phase 3  2-opt polish   : {pt.get('p3_2opt', 0):>7.1f}s")
    print(f"    Total RRD overhead      : {result['rrd_time']:>7.1f}s")

    imp  = result["improvement_pct"]
    sign = "↓" if imp > 0 else "↑"
    print(f"\n  Overall cost improvement  (T-ALNS-RRD vs ALNS+Tabu): {sign} {abs(imp):.2f}%")

    # Final verdict
    checks = {
        "Total cost":     r_total < a_total,
        "Transport":       r_tr    < a_tr,
        "Delay penalty":   r_late  < a_late,
        "Congestion":      r_cong  < a_cong,
    }
    beaten = [k for k, v in checks.items() if v]
    worse  = [k for k, v in checks.items() if not v]

    if not worse:
        print(f"\n  ✅  T-ALNS-RRD v2 beats ALNS+Tabu on ALL 4 cost metrics!")
    else:
        if beaten:
            print(f"\n  ✓  Better on  : {', '.join(beaten)}")
        print(f"  ✗  Worse on   : {', '.join(worse)}")

    print(f"{'═'*W}\n")


def print_rrd_solution(data: dict, num_vehicles: int, result: dict):
    """Print the T-ALNS-RRD routes."""
    print_solution(
        data,
        num_vehicles,
        "T-ALNS-RRD v2",
        result["rrd_cost"],
        result["rrd_routes"],
        result["rrd_time"],
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys, os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    sys.path.insert(0, project_root)
    sys.path.insert(0, os.path.join(project_root, "src"))

    data = load_data()
    num_customers = len(data["distance_matrix"]) - 1

    print(f"{'═'*60}")
    print(f"  T-ALNS-RRD v2 — VRP Optimizer")
    print(f"{'═'*60}")
    print(f"  Loaded    : {num_customers} stores + 1 depot")
    print(f"  Capacity  : {data['vehicle_capacity']} kg")
    print(f"  Cost/km   : {data['cost_per_km']:,} VND")
    print(f"  Late pen. : {data['late_penalty_per_min']:,} VND/min")
    print(f"  Cong pen. : {data['congestion_penalty']:,} VND/unit")
    print()

    K = 4   

    result = rrd_solve(
        data,
        num_vehicles=K,
        alns_iterations=5,
        reassign_iter=10,
        verbose=True,
    )

    print_rrd_solution(data, K, result)
    compare_and_print(data, K, result)
