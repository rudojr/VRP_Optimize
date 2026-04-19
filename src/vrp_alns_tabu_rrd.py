import copy
import math
import time as time_mod
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

# ── Re-use all infrastructure from vrp_alns_tabu ─────────────────────────────
from vrp_alns_tabu import (
    CostEvaluator,
    Route,
    Solution,
    alns_tabu_solve,
    local_search_2opt,
    print_solution,
    CONGESTION_LABELS,
    generate_congestion_matrix,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  COST EVALUATOR — RRD variant (uses CSV time_matrix_h directly)
# ═══════════════════════════════════════════════════════════════════════════════

class CostEvaluatorRRD(CostEvaluator):
    """
    Drop-in replacement for CostEvaluator that uses the pre-loaded 2-D
    travel-time matrix (unit: hours, read from 140_time_matrix.csv) instead of
    deriving travel time from distance ÷ speed × congestion.

    travel_time[i][j]  =  time_matrix_h[i][j] × 60   (minutes)

    All route-cost logic (transport, late-penalty, congestion penalty) is
    inherited unchanged from CostEvaluator.
    """

    def __init__(self, data: dict):
        super().__init__(data)                          # builds default travel_time
        time_h: np.ndarray = data["time_matrix_h"]     # shape (N, N), unit: hours
        # Override travel_time (minutes) with the real-world CSV values
        self.travel_time = time_h * 60.0               # → minutes


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
#  EVENT-DRIVEN ROLLOUT DISPATCHER  (Phase 2b — Option A)
#  Offline simulation of Algorithm 3 (paper) — urgency-triggered dispatch
# ═══════════════════════════════════════════════════════════════════════════════

class EventDrivenRolloutDispatcher:
    """
    Offline simulation của real-time event-driven dispatch (Algorithm 3, paper).

    Thay vì luôn rollout mọi bước (như RolloutDispatcher cũ), chỉ trigger
    rollout khi urgency vượt ngưỡng — còn lại follow ALNS pre-planned sequence.

    Urgency (xấp xỉ Eq.34):
        urgency(c) = max(0, 1 - slack(c) / tw_width(c))
        slack(c)   = tw_end(c) - estimated_arrival(c)

    Workflow tại mỗi bước dispatch:
        urgency, _ = _calc_urgency(current_node, current_time, unvisited)
        if urgency > threshold  →  ROLLOUT  (event triggered)
        else                    →  follow ALNS plan
    """

    def __init__(
        self,
        data: dict,
        evaluator: CostEvaluator,
        urgency_threshold: float = 0.65,
    ):
        self.data    = data
        self.ev      = evaluator
        self.depot   = data["depot"]
        self.u_thresh = urgency_threshold
        self.policy  = _GreedyTWPolicy(evaluator)   # base policy cho rollout completion

    # ── Urgency calculation (Eq. 34 approximation) ───────────────────────────

    def _calc_urgency(
        self,
        current_node: int,
        current_time: float,
        unvisited: List[int],
    ) -> Tuple[float, Optional[int]]:
        """
        Tính urgency cao nhất trong số các customers chưa ghé.
        Returns (max_urgency ∈ [0,1], most_urgent_customer).

        urgency → 0  : còn nhiều thời gian, không cần can thiệp
        urgency → 1  : sắp/đã trễ, cần rollout ngay
        """
        max_urgency = 0.0
        urgent_cust: Optional[int] = None

        for cust in unvisited:
            tw_start = float(self.ev.tw[cust][0])
            tw_end   = float(self.ev.tw[cust][1])
            tw_width = max(tw_end - tw_start, 1.0)

            service  = self.ev.s_time if current_node != self.depot else 0
            tt       = self.ev.travel_time[current_node][cust]
            est_arr  = current_time + service + tt
            est_arr  = max(est_arr, tw_start)      # chờ nếu đến sớm

            slack    = tw_end - est_arr             # âm = đã trễ
            urgency  = max(0.0, 1.0 - slack / tw_width)

            if urgency > max_urgency:
                max_urgency = urgency
                urgent_cust = cust

        return max_urgency, urgent_cust

    # ── 1-step rollout (chỉ gọi khi event triggered) ─────────────────────────

    def _rollout_next_stop(
        self,
        from_node: int,
        from_time: float,
        unvisited: List[int],
    ) -> Tuple[int, float]:
        """Chọn next stop tối ưu qua 1-step lookahead + greedy completion."""
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

    # ── Main dispatch loop ────────────────────────────────────────────────────

    def dispatch_route(
        self,
        planned_customers: List[int],
    ) -> Tuple[Route, List[dict]]:
        """
        Simulate event-driven execution của một route.

        planned_customers : thứ tự từ ALNS/Phase-2a ("kế hoạch")
        Returns           : (Route thực thi, timing_log kèm event marker)
        """
        if not planned_customers:
            return Route(), []

        current_node = self.depot
        current_time = float(self.ev.tw[self.depot][0])
        unvisited    = list(planned_customers)
        plan_queue   = list(planned_customers)   # bản sao kế hoạch ALNS
        sequence: List[int] = []
        timing_log   = [{
            "node": self.depot, "arrival": current_time,
            "lateness": 0.0, "event": "START",
        }]

        while unvisited:
            urgency, _ = self._calc_urgency(current_node, current_time, unvisited)

            if urgency > self.u_thresh:
                # ── EVENT triggered: rollout chọn next stop tốt nhất ────────
                next_stop, sim_cost = self._rollout_next_stop(
                    current_node, current_time, unvisited
                )
                event_type = f"ROLLOUT (u={urgency:.2f})"
                if next_stop in plan_queue:
                    plan_queue.remove(next_stop)
            else:
                # ── Không urgent: follow ALNS plan ──────────────────────────
                next_stop = None
                while plan_queue:
                    candidate = plan_queue.pop(0)
                    if candidate in unvisited:
                        next_stop = candidate
                        break
                if next_stop is None:          # fallback nếu plan hết
                    next_stop = unvisited[0]
                sim_cost   = 0.0
                event_type = f"PLAN    (u={urgency:.2f})"

            # Thực thi step
            service  = self.ev.s_time if current_node != self.depot else 0
            tt_to    = self.ev.travel_time[current_node][next_stop]
            arrive   = current_time + service + tt_to
            arrive   = max(arrive, float(self.ev.tw[next_stop][0]))
            lateness = max(0.0, arrive - float(self.ev.tw[next_stop][1]))

            timing_log.append({
                "node":     next_stop,
                "arrival":  arrive,
                "lateness": lateness,
                "sim_cost": sim_cost,
                "event":    event_type,
            })

            sequence.append(next_stop)
            current_node = next_stop
            current_time = arrive
            unvisited.remove(next_stop)

        return Route(customers=sequence), timing_log

    def dispatch_all(self, solution: Solution) -> Tuple[Solution, list]:
        """Apply event-driven dispatch to every route; return stats."""
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

    def rescue_dropped(
        self,
        solution: Solution,
        dropped_customers: List[int],
        verbose: bool = False,
    ) -> Solution:
        """
        Phase 2a+ — Dropped Customer Rescue.

        Chỉ chèn customer nếu:
          1. Capacity không vượt quá Q
          2. Customer được giao ĐÚNG GIỜ (lateness == 0) tại vị trí chèn
          3. Chọn vị trí có delta_cost nhỏ nhất trong các vị trí hợp lệ

        Ưu tiên dropped customer theo TW start sớm nhất.
        """
        if not dropped_customers:
            return solution

        sol = Solution(routes=[Route(customers=r.customers[:]) for r in solution.routes])
        self.ev.evaluate_solution(sol)

        remaining_dropped = sorted(dropped_customers, key=lambda c: self.ev.tw[c][0])

        rescued = 0
        for cust in remaining_dropped:
            best_delta   = float("inf")
            best_r_idx   = -1
            best_ins_pos = 0

            for r_idx, route in enumerate(sol.routes):
                # Kiểm tra capacity
                if self.ev.route_load(route) + self.ev.demands[cust] > self.ev.Q:
                    continue

                cost_before = self._exact_cost(route.customers)

                # Duyệt từng vị trí — CHỈ chấp nhận nếu customer giao đúng giờ
                for pos in range(len(route.customers) + 1):
                    trial = route.customers[:pos] + [cust] + route.customers[pos:]
                    info  = self.ev.evaluate_route(Route(customers=trial))
                    # timing[0]=depot, timing[pos+1]=customer vừa chèn
                    if info["timing"][pos + 1]["lateness"] > 0.01:
                        continue   # sẽ trễ → bỏ qua vị trí này
                    delta = info["total"] - cost_before
                    if delta < best_delta:
                        best_delta   = delta
                        best_r_idx   = r_idx
                        best_ins_pos = pos

            if best_r_idx >= 0:
                custs_t = sol.routes[best_r_idx].customers
                sol.routes[best_r_idx].customers = (
                    custs_t[:best_ins_pos] + [cust] + custs_t[best_ins_pos:]
                )
                self.ev.evaluate_solution(sol)
                rescued += 1

                if verbose:
                    print(f"    [Rescue] cust {cust} → route {best_r_idx} "
                          f"pos {best_ins_pos}, Δcost={best_delta:+,.0f} VND ✓ on-time")

        if verbose:
            print(f"    [Rescue] Rescued {rescued}/{len(dropped_customers)} dropped (on-time only)")

        self.ev.evaluate_solution(sol)
        return sol

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

def load_data():

    df_store = pd.read_csv("data/140_stores.csv")
    df_store["Demand_kg"] = (
        df_store["Demand"]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )

    def time_to_minutes(t_str):
        h, m = t_str.strip().split(":")
        return int(h) * 60 + int(m)

    df_store["tw_start"] = df_store["Time start"].apply(time_to_minutes)
    df_store["tw_end"]   = df_store["Time end"].apply(time_to_minutes)

    df_matrix = pd.read_csv("data/140_distant_matrix.csv")
    df_matrix = df_matrix.drop(columns=["From/to"])
    distance_matrix_m  = df_matrix.values.astype(float)   
    distance_matrix_km = distance_matrix_m / 1_000.0       

    df_time = pd.read_csv("data/140_time_matrix.csv")
    df_time = df_time.drop(columns=["From/to (Hour)"])
    time_matrix_h = df_time.values.astype(float)   

    num_nodes = distance_matrix_km.shape[0]
    congestion_matrix = generate_congestion_matrix(num_nodes, seed=42)

    store_names = ["Depot"] + df_store["Tên CH"].tolist()
    demands     = [0] + df_store["Demand_kg"].tolist()

    depot_tw     = (6 * 60, 18 * 60)
    time_windows = [depot_tw] + list(
        zip(df_store["tw_start"].tolist(), df_store["tw_end"].tolist())
    )

    DEPOT_COORDS = (10.730399, 106.5991)
    COST_PER_M   = 18                     
    COST_PER_KM  = COST_PER_M * 1_000    
    NUM_VEHICLES = 13
    CAPACITY     = 1_800                 

    data = {
        "distance_matrix":   distance_matrix_km,  
        "distance_matrix_m": distance_matrix_m,   
        "time_matrix_h":     time_matrix_h,        
        "congestion_matrix": congestion_matrix,
        "demands":           demands,
        "time_windows":      time_windows,
        "depot":             0,
        "depot_coords":      DEPOT_COORDS,
        "store_names":       store_names,
        "vehicle_capacity":     CAPACITY,
        "num_vehicles":         NUM_VEHICLES,
        "cost_per_m":           COST_PER_M,
        "cost_per_km":          COST_PER_KM,
        "late_penalty_per_min": 50_000,
        "congestion_penalty":   5_000,
        "service_time_min":     10,
        "avg_speed_kmh":        30,   # km/h — CostEvaluator compat
    }
    return data


def rrd_solve(
    data: dict,
    num_vehicles: int,
    alns_iterations: int = 50,
    reassign_iter: int = 10,
    verbose: bool = True,
) -> dict:

    evaluator = CostEvaluatorRRD(data)

    if verbose:
        print(f"\n{'═'*70}")
        print(f"  Phase 1 — ALNS+Tabu Search  (K={num_vehicles}, {alns_iterations} iters)")
        print(f"{'═'*70}")

    alns_start = time_mod.time()

    # Patch Phase 1 so alns_tabu_solve also uses the real-world time matrix
    import vrp_alns_tabu as _alns_mod
    _orig_evaluator_cls = _alns_mod.CostEvaluator
    _alns_mod.CostEvaluator = CostEvaluatorRRD
    try:
        alns_status, alns_total_cost, alns_routes_info, alns_solve_time = alns_tabu_solve(
            data,
            num_vehicles,
            max_iterations=alns_iterations,
            seed=42,
            verbose=verbose,
        )
    finally:
        _alns_mod.CostEvaluator = _orig_evaluator_cls   # always restore

    alns_elapsed = time_mod.time() - alns_start

    # Xác định dropped customers (có trong data nhưng không có trong bất kỳ route ALNS nào)
    all_customers = set(range(1, len(data["distance_matrix"])))
    served_by_alns = set(
        c
        for r in alns_routes_info
        for c in r["nodes"][1:-1]
    )
    dropped_customers = sorted(all_customers - served_by_alns)

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
        if dropped_customers:
            print(f"    ⚠ Dropped customers: {len(dropped_customers)} → "
                  f"will attempt rescue in Phase 2a+")

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

    # Phase 2a+ : giải cứu dropped customers trước khi reassign
    if dropped_customers:
        if verbose:
            print(f"\nPhase 2a+ — Dropped Customer Rescue ({len(dropped_customers)} stores)")
        alns_sol = reassigner.rescue_dropped(alns_sol, dropped_customers, verbose=verbose)
        if verbose:
            print(f"After rescue — cost = {alns_sol.cost:,.0f} VND")

    reassigned_sol = reassigner.optimize(alns_sol, verbose=verbose)
    p2a_time     = time_mod.time() - p2a_start

    if verbose:
        print(f"\nPhase 2a complete — cost = {reassigned_sol.cost:,.0f} VND  ({p2a_time:.1f}s)")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 2b — Event-Driven Rollout Dispatch (Option A)
    # ══════════════════════════════════════════════════════════════════════
    urgency_threshold = 0.65   # trigger rollout khi slack < 35% TW width

    if verbose:
        print(f"\n{'═'*70}")
        print(f"  Phase 2b — Event-Driven Rollout Dispatch (Algorithm 3 offline sim)")
        print(f"  urgency_threshold = {urgency_threshold}  "
              f"(rollout triggered khi slack < {(1-urgency_threshold)*100:.0f}% TW width)")
        print(f"{'═'*70}")

    p2b_start  = time_mod.time()
    dispatcher = EventDrivenRolloutDispatcher(data, evaluator, urgency_threshold)
    reseq_sol, timing_logs = dispatcher.dispatch_all(reassigned_sol)
    p2b_time   = time_mod.time() - p2b_start

    # Đếm số lần rollout được trigger (events)
    n_rollout = sum(
        1 for logs in timing_logs
          for entry in logs
          if entry.get("event", "").startswith("ROLLOUT")
    )
    n_plan = sum(
        1 for logs in timing_logs
          for entry in logs
          if entry.get("event", "").startswith("PLAN")
    )

    if verbose:
        print(f"\n  ✓ Phase 2b complete — cost = {reseq_sol.cost:,.0f} VND  ({p2b_time:.1f}s)")
        print(f"    Dispatch events : {n_rollout} ROLLOUT  |  {n_plan} FOLLOW-PLAN")
        if n_rollout + n_plan > 0:
            print(f"    Rollout rate    : {n_rollout/(n_rollout+n_plan)*100:.1f}%")

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
        print(f"\nPhase 3 complete — cost = {final_sol.cost:,.0f} VND  ({p3_time:.1f}s)")

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

    # Tổng số khách hàng thực tế (không kể depot)
    total_customers = len(data["distance_matrix"]) - 1

    def route_metrics(routes_info):
        total_c   = sum(r["total_cost"]      for r in routes_info)
        transport = sum(r["transport_cost"]  for r in routes_info)
        late      = sum(r["late_penalty"]    for r in routes_info)
        cong      = sum(r["congestion_cost"] for r in routes_info)
        # Số khách được ghé thăm thực tế
        n_served  = sum(len(r["nodes"]) - 2  for r in routes_info)
        on_time   = sum(
            1 for r in routes_info
              for t in r["timing"]
              if t["node"] != data["depot"] and t["lateness"] <= 0.01
        )
        # OTDR (partial): tỷ lệ on-time trong số stores ĐƯỢC phục vụ
        otdr_partial = on_time / n_served * 100 if n_served else 0
        # True OTDR: tính dropped stores là "không đúng giờ" (không giao được)
        otdr_true    = on_time / total_customers * 100 if total_customers else 0
        # Service rate: tỷ lệ stores được phục vụ
        service_rate = n_served / total_customers * 100 if total_customers else 0
        return total_c, transport, late, cong, otdr_partial, otdr_true, service_rate, n_served, len(routes_info)

    a_total, a_tr, a_late, a_cong, a_otdr, a_otdr_true, a_svc, a_served, a_active = route_metrics(result["alns_routes"])
    r_total, r_tr, r_late, r_cong, r_otdr, r_otdr_true, r_svc, r_served, r_active = route_metrics(result["rrd_routes"])

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
    print(f"  {'─'*34} {'─'*18}  {'─'*18}  {'─'*14}  {'─'*8}")
    row(f"Served stores (/{total_customers})", a_served, r_served, fmt="{:>18,.0f}")
    row_pct("  ↳ Service rate (%)",    a_svc,   r_svc)
    print(f"  {'─'*34} {'─'*18}  {'─'*18}  {'─'*14}  {'─'*8}")
    row_pct("OTDR — served only (%)",  a_otdr,  r_otdr)
    row_pct("OTDR — TRUE (all stores)", a_otdr_true, r_otdr_true)
    row("Active vehicles",            a_active, r_active, fmt="{:>18,.0f}")

    print(f"  {'─'*34} {'─'*18}  {'─'*18}  {'─'*14}  {'─'*8}")

    pt = result.get("phase_times", {})
    # print(f"\n  Timing breakdown:")
    # print(f"    Phase 1  ALNS+Tabu      : {pt.get('p1_alns', result['alns_time']):>7.1f}s")
    # print(f"    Phase 2a Reassignment   : {pt.get('p2a_reassign', 0):>7.1f}s")
    # print(f"    Phase 2b Resequencing   : {pt.get('p2b_reseq', 0):>7.1f}s")
    # print(f"    Phase 3  2-opt polish   : {pt.get('p3_2opt', 0):>7.1f}s")
    # print(f"    Total RRD overhead      : {result['rrd_time']:>7.1f}s")

    imp  = result["improvement_pct"]
    sign = "↓" if imp > 0 else "↑"
    # print(f"\n  Overall cost improvement  (T-ALNS-RRD vs ALNS+Tabu): {sign} {abs(imp):.2f}%")

    checks = {
        "Total cost":     r_total < a_total,
        "Transport":       r_tr    < a_tr,
        "Delay penalty":   r_late  < a_late,
        "Congestion":      r_cong  < a_cong,
    }
    beaten = [k for k, v in checks.items() if v]
    worse  = [k for k, v in checks.items() if not v]

    # if not worse:
    #     print(f"\n  ✅  T-ALNS-RRD v2 beats ALNS+Tabu on ALL 4 cost metrics!")
    # else:
    #     if beaten:
    #         print(f"\n  ✓  Better on  : {', '.join(beaten)}")
    #     print(f"  ✗  Worse on   : {', '.join(worse)}")

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


def calc_metrics(routes_info: list, data: dict) -> dict:
    N     = len(data["distance_matrix"]) - 1  
    depot = data["depot"]
    tw    = data["time_windows"]
    cong  = data["congestion_matrix"]

    on_time = 0
    ces_sum = 0.0

    for r in routes_info:
        for t in r["timing"]:
            node = t["node"]
            if node == depot:
                continue
            if t["lateness"] <= 0.01:
                on_time += 1

        nodes = r["nodes"]   
        for idx in range(len(nodes) - 1):
            i, j = nodes[idx], nodes[idx + 1]
            ces_sum += cong[i][j]

    otdr = on_time / N * 100.0 if N > 0 else 0.0

    return {"otdr": otdr, "ces": ces_sum, "on_time": on_time, "N": N}


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN  — Fleet sweep K = 3 → 13, max_iter = 100
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys, os, io, datetime
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    sys.path.insert(0, project_root)
    sys.path.insert(0, os.path.join(project_root, "src"))

    # ── Tee: ghi đồng thời ra stdout và buffer để lưu log ────────────────
    class _Tee:
        """Write to both the original stdout and an in-memory buffer."""
        def __init__(self, original):
            self._orig = original
            self._buf  = io.StringIO()
        def write(self, msg):
            self._orig.write(msg)
            self._buf.write(msg)
        def flush(self):
            self._orig.flush()
        def getvalue(self):
            return self._buf.getvalue()

    _tee = _Tee(sys.stdout)
    sys.stdout = _tee

    data = load_data()
    num_customers = len(data["distance_matrix"]) - 1

    print(f"\n{'═'*72}")
    print(f"  T-ALNS-RRD  ·  Tabu + ALNS + Rollout VRP Optimizer")
    print(f"{'═'*72}")
    print(f"  Stores     : {num_customers}  |  Depot   : {data['depot_coords']}")
    print(f"  Capacity   : {data['vehicle_capacity']} kg  |  Cost/m  : {data['cost_per_m']} VND")
    print(f"  Max iter   : 500  |  Fleet sweep : K = 3 → 13")
    print(f"{'═'*72}\n")

    summary = []

    for K in range(3, 14):
        t0 = time_mod.time()
        result = rrd_solve(
            data,
            num_vehicles=K,
            alns_iterations=500,
            reassign_iter=10,
            verbose=False,
        )
        comp_time = time_mod.time() - t0

        m = calc_metrics(result["rrd_routes"], data)
        summary.append({
            "K":          K,
            "total_cost": result["rrd_cost"],
            "otdr":       m["otdr"],
            "ces":        m["ces"],
            "on_time":    m["on_time"],
            "time":       comp_time,
        })

        # Live progress line printed after each K completes
        print(
            f"  K={K:>2}  cost={result['rrd_cost']:>15,.0f} VND  "
            f"OTDR={m['otdr']:>6.2f}%  CES={m['ces']:>6.2f}  "
            f"time={comp_time:>7.1f}s"
        )

    # ── Final summary table ───────────────────────────────────────────────
    print(f"\n{'═'*74}")
    print(f"  RESULTS — Tabu-ALNS-Rollout  (max_iter=100, N={num_customers} stores)")
    print(f"{'═'*74}")
    print(
        f"  {'K':>3}  {'Total Cost (VND)':>18}  "
        f"{'OTDR (%)':>10}  {'CES Score':>10}  {'Time (s)':>9}"
    )
    print(f"  {'─'*3}  {'─'*18}  {'─'*10}  {'─'*10}  {'─'*9}")

    best_cost = min(r["total_cost"] for r in summary)
    for r in summary:
        star = " ★" if abs(r["total_cost"] - best_cost) < 1 else "  "
        print(
            f"  {r['K']:>3}  {r['total_cost']:>18,.0f}  "
            f"{r['otdr']:>9.2f}%  {r['ces']:>10.2f}  "
            f"{r['time']:>8.1f}s{star}"
        )

    print(f"{'═'*74}")
    best_r = min(summary, key=lambda x: x["total_cost"])
    print(
        f"\n  ★  Best K = {best_r['K']}  →  "
        f"{best_r['total_cost']:,.0f} VND  |  "
        f"OTDR = {best_r['otdr']:.2f}%  |  "
        f"CES = {best_r['ces']:.2f}\n"
    )

    # ── Lưu toàn bộ output vào file log ─────────────────────────────────
    sys.stdout = _tee._orig          # khôi phục stdout gốc
    log_path   = os.path.join(project_root, "log_result_alns_tabu_rrd.txt")
    timestamp  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header     = (
        f"{'═'*74}\n"
        f"  LOG — T-ALNS-RRD  |  Run: {timestamp}\n"
        f"{'═'*74}\n"
    )
    with open(log_path, "w", encoding="utf-8") as _f:
        _f.write(header)
        _f.write(_tee.getvalue())
    print(f"\n  📄  Kết quả đã được lưu vào: {log_path}")