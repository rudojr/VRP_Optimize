import copy
import math
import time as time_mod
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

MAX_ITER = 50

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



class CostEvaluatorRRD(CostEvaluator):


    def __init__(self, data: dict):
        super().__init__(data)                         
        time_h: np.ndarray = data["time_matrix_h"]    
        self.travel_time = time_h * 60.0              



class _GreedyTWPolicy:
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
        n_mc_sims: int = 5,
    ):
        self.data      = data
        self.ev        = evaluator
        self.depot     = data["depot"]
        self.u_thresh  = urgency_threshold
        self.n_mc_sims = n_mc_sims          # Monte Carlo rollout iterations
        self.policy    = _GreedyTWPolicy(evaluator)   # base policy cho rollout completion

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
    ) -> Tuple[int, float, float]:
        """
        Chọn next stop tối ưu qua 1-step lookahead + greedy completion.
        Returns (best_next, best_cost, second_best_cost).
        """
        best_next = None
        best_cost = float("inf")
        second_best_cost = float("inf")

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
                second_best_cost = best_cost
                best_cost = total
                best_next = candidate
            elif total < second_best_cost:
                second_best_cost = total

        return best_next, best_cost, second_best_cost


    def dispatch_route(
        self,
        planned_customers: List[int],
    ) -> Tuple[Route, List[dict]]:

        if not planned_customers:
            return Route(), []

        current_node = self.depot
        current_time = float(self.ev.tw[self.depot][0])
        unvisited    = list(planned_customers)
        plan_queue   = list(planned_customers)  
        sequence: List[int] = []
        cum_load     = 0.0                      
        timing_log   = [{
            "node": self.depot, "arrival": current_time,
            "lateness": 0.0, "event": "START",
        }]

        while unvisited:
            urgency, _ = self._calc_urgency(current_node, current_time, unvisited)

            extra = {}  

            if urgency > self.u_thresh:
                # ── EVENT triggered: Monte Carlo rollout ──────────────────
                t0_perf = time_mod.perf_counter()

                orig_tt = self.ev.travel_time.copy()
                mc_votes: Dict[int, int] = {}
                mc_best_costs: Dict[int, List[float]] = {}
                mc_2nd_costs: List[float] = []
                actual_sims = 0

                for mc_i in range(self.n_mc_sims):
                    # Perturbation nhẹ cho MC (trừ sim đầu tiên dùng tt gốc)
                    if mc_i > 0:
                        noise = np.random.normal(0, 0.05, orig_tt.shape)
                        self.ev.travel_time = np.clip(
                            orig_tt * (1 + noise), 0, None
                        )
                        self.policy._tt = self.ev.travel_time

                    mc_next, mc_cost, mc_2nd = self._rollout_next_stop(
                        current_node, current_time, unvisited
                    )
                    mc_votes[mc_next] = mc_votes.get(mc_next, 0) + 1
                    mc_best_costs.setdefault(mc_next, []).append(mc_cost)
                    mc_2nd_costs.append(mc_2nd)
                    actual_sims += 1

                # Restore original travel times
                self.ev.travel_time = orig_tt
                self.policy._tt     = orig_tt

                # Consensus: pick the most-voted candidate
                next_stop  = max(mc_votes, key=mc_votes.get)
                sim_cost   = float(np.mean(mc_best_costs[next_stop]))
                second_best_cost = float(np.mean(mc_2nd_costs))

                response_ms  = (time_mod.perf_counter() - t0_perf) * 1000
                n_candidates = len(unvisited)

                # ── Cost impact: rollout vs plan (or vs 2nd-best) ────────
                plan_next = None
                for pq in list(plan_queue):
                    if pq in unvisited:
                        plan_next = pq
                        break

                cost_impact_pct = 0.0
                if plan_next is not None and plan_next != next_stop:
                    # Rollout chose differently from plan → compare costs
                    svc_p    = self.ev.s_time if current_node != self.depot else 0
                    tt_p     = self.ev.travel_time[current_node][plan_next]
                    cong_p   = self.ev.cong[current_node][plan_next]
                    arrive_p = max(current_time + svc_p + tt_p,
                                   float(self.ev.tw[plan_next][0]))
                    late_p   = max(0.0, arrive_p - float(self.ev.tw[plan_next][1]))
                    step_p   = (self.ev.dist[current_node][plan_next] * self.ev.cpk
                                + self.ev.lam2 * cong_p
                                + self.ev.lam1 * late_p)
                    rem_p    = [c for c in unvisited if c != plan_next]
                    compl_p  = self.policy.simulate(plan_next, arrive_p, rem_p)
                    plan_tot = step_p + compl_p
                    if plan_tot > 0:
                        cost_impact_pct = (plan_tot - sim_cost) / plan_tot * 100
                elif second_best_cost < float("inf") and second_best_cost > 0:
                    # Rollout confirmed plan → margin vs 2nd-best alternative
                    cost_impact_pct = (second_best_cost - sim_cost) / second_best_cost * 100

                cong_arc = int(self.ev.cong[current_node][next_stop])
                load_pct = cum_load / self.ev.Q * 100 if self.ev.Q > 0 else 0

                extra = {
                    "response_ms":     response_ms,
                    "n_candidates":    n_candidates,
                    "n_mc_sims":       actual_sims,
                    "urgency_val":     urgency,
                    "cong_arc":        cong_arc,
                    "route_load_pct":  load_pct,
                    "cost_impact_pct": cost_impact_pct,
                }

                event_type = f"ROLLOUT (u={urgency:.2f})"
                if next_stop in plan_queue:
                    plan_queue.remove(next_stop)
            else:
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

            cum_load += self.ev.demands[next_stop]

            entry = {
                "node":     next_stop,
                "arrival":  arrive,
                "lateness": lateness,
                "sim_cost": sim_cost,
                "event":    event_type,
            }
            entry.update(extra)
            timing_log.append(entry)

            sequence.append(next_stop)
            current_node = next_stop
            current_time = arrive
            unvisited.remove(next_stop)

        return Route(customers=sequence), timing_log

    def dispatch_all(self, solution: Solution) -> Tuple[Solution, list]:
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
    def __init__(self, evaluator: CostEvaluator, max_iter: int = 10):
        self.ev       = evaluator
        self.max_iter = max_iter

    def _exact_cost(self, customers: List[int]) -> float:
        if not customers:
            return 0.0
        return self.ev.evaluate_route(Route(customers=customers))["total"]

    def _best_insert_position(self, customers: List[int], new_cust: int) -> Tuple[int, float]:
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


# Peak-hour slots (minutes): 07-09h and 17-19h
_PEAK_START1, _PEAK_END1 = 7 * 60, 9 * 60
_PEAK_START2, _PEAK_END2 = 17 * 60, 19 * 60
_HIGH_CONG_RHO = 3          # ρ threshold for "high-congestion arc"


def _is_peak(minute: float) -> bool:
    return (_PEAK_START1 <= minute < _PEAK_END1) or (_PEAK_START2 <= minute < _PEAK_END2)


def calc_metrics(routes_info: list, data: dict) -> dict:
    N     = len(data["distance_matrix"]) - 1  
    depot = data["depot"]
    tw    = data["time_windows"]
    cong  = data["congestion_matrix"]

    on_time     = 0
    ces_sum     = 0.0
    ces_peak    = 0.0
    ces_offpeak = 0.0
    total_arcs  = 0
    hc_arcs     = 0

    for r in routes_info:
        timing_map = {t["node"]: t["service_start"] for t in r["timing"]}

        for t in r["timing"]:
            node = t["node"]
            if node == depot:
                continue
            if t["lateness"] <= 0.01:
                on_time += 1

        nodes = r["nodes"]
        for idx in range(len(nodes) - 1):
            i, j     = nodes[idx], nodes[idx + 1]
            rho      = cong[i][j]
            arr_i    = timing_map.get(i, 0.0)
            ces_sum += rho
            total_arcs += 1
            if rho >= _HIGH_CONG_RHO:
                hc_arcs += 1
            if _is_peak(arr_i):
                ces_peak    += rho
            else:
                ces_offpeak += rho

    otdr = on_time / N * 100.0 if N > 0 else 0.0

    return {
        "otdr":        otdr,
        "ces":         ces_sum * 5,          # scale × 5
        "ces_peak":    ces_peak * 5,
        "ces_offpeak": ces_offpeak * 5,
        "hc_arcs":     hc_arcs,
        "total_arcs":  total_arcs,
        "on_time":     on_time,
        "N":           N,
    }




def print_ces_report(m: dict, W: int = 90) -> None:
    print(f"{'═'*W}")
    print(f"  CES (Congestion Exposure Score)  -- Lower is better")
    print(f"{'═'*W}")
    print(f"  {'CES Score':<30}: {m['ces']:>10.0f}")
    print(f"  {'Peak-hour CES':<30}: {m['ces_peak']:>10.0f}  (07-09h, 17-19h)")
    print(f"  {'Off-peak CES':<30}: {m['ces_offpeak']:>10.0f}")
    print(
        f"  {'High-congestion arcs':<30}: {m['hc_arcs']:>10}  "
        f"(rho >= {_HIGH_CONG_RHO}) out of {m['total_arcs']} total arcs"
    )
    print(f"{'═'*W}")


def print_event_response_metrics(
    result: dict,
    best_result: dict | None = None,
    W: int = 90,
) -> None:

    timing_logs = result.get("timing_logs", [])

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
    n_total   = n_rollout + n_plan
    event_rate = n_rollout / n_total * 100 if n_total > 0 else 0.0

    imp_pct   = result.get("improvement_pct", 0.0)

    rollout_ontime = 0
    rollout_total  = 0
    for logs in timing_logs:
        for entry in logs:
            if entry.get("event", "").startswith("ROLLOUT"):
                rollout_total += 1
                if entry.get("lateness", 1.0) <= 0.01:
                    rollout_ontime += 1
    adapt_success = rollout_ontime / rollout_total * 100 if rollout_total > 0 else 100.0

    # Latency impact: average sim_cost per rollout decision (proxy for extra ms)
    sim_costs = [
        entry.get("sim_cost", 0.0)
        for logs in timing_logs
        for entry in logs
        if entry.get("event", "").startswith("ROLLOUT")
    ]
    avg_sim_cost = sum(sim_costs) / len(sim_costs) if sim_costs else 0.0
    latency_ms   = avg_sim_cost / 1e6

    print(f"{'═'*W}")
    print(f"  EVENT RESPONSE METRICS  (Rollout Dispatch -- Phase 2b)")
    print(f"{'═'*W}")
    print(
        f"  {'Event response rate':<30}: {event_rate:>9.2f} %  "
        f"({n_rollout} rollout / {n_total} total)"
    )
    sign = "-" if imp_pct >= 0 else "+"
    print(f"  {'Cost reduction (ALNS->RRD)':<30}: {sign}{abs(imp_pct):>8.2f} %")
    print(f"  {'Adaptation success':<30}: {adapt_success:>9.2f} %")
    print(f"  {'Latency impact':<30}: {latency_ms:>9.2f} ms/decision")
    print(f"{'═'*W}")


def print_robustness_analysis(
    data: dict,
    result: dict,
    sigmas: tuple = (0.1, 0.2, 0.3, 0.5),
    n_trials: int = 5,
    W: int = 90,
) -> None:
    """
    Robustness Analysis: perturb travel times with Gaussian noise (sigma × mean)
    and re-evaluate the best solution's cost.
    """
    import random, math as _math

    evaluator = CostEvaluatorRRD(data)
    base_routes = result["rrd_routes"]
    base_cost   = result["rrd_cost"]

    scenario_results = []
    for sigma in sigmas:
        trial_costs = []
        for _ in range(n_trials):
            orig_tt = evaluator.travel_time          
            noise   = np.random.normal(0, sigma, orig_tt.shape)
            perturbed = np.clip(orig_tt * (1 + noise), 0, None)

            evaluator.travel_time = perturbed
            total = 0.0
            for r in base_routes:
                route_obj = Route(customers=r["nodes"][1:-1])
                info = evaluator.evaluate_route(route_obj)
                total += info["total"]
            trial_costs.append(total)
            evaluator.travel_time = orig_tt  # restore

        avg_cost    = sum(trial_costs) / len(trial_costs)
        cost_inc    = (avg_cost - base_cost) / base_cost * 100 if base_cost > 0 else 0.0
        label       = (
            "Low" if sigma <= 0.1 else
            "Medium" if sigma <= 0.2 else
            "High" if sigma <= 0.3 else
            "Extreme"
        )
        scenario_results.append((f"{label} uncertainty (s={sigma})", avg_cost, cost_inc))

    avg_inc_pct    = sum(r[2] for r in scenario_results) / len(scenario_results)
    robustness_idx = max(0.0, 1.0 - avg_inc_pct / 100)

    print(f"{'═'*W}")
    print(f"  ROBUSTNESS ANALYSIS  (Travel-time perturbation, {n_trials} trials each)")
    print(f"{'═'*W}")
    print(f"  {'Scenario':<32} {'Avg Cost (VND)':>16}  {'Cost Increase':>13}")
    print(f"  {'─'*32} {'─'*16}  {'─'*13}")
    for label, avg_c, inc in scenario_results:
        sign = "+" if inc >= 0 else "-"
        print(f"  {label:<32} {avg_c:>16,.0f}  {sign}{abs(inc):>11.2f}%")
    print(f"  {'─'*32} {'─'*16}  {'─'*13}")
    print(f"  {'Robustness Index':<32}: {robustness_idx:>7.4f}  (1.0 = perfectly robust)")
    print(f"{'═'*W}")


def print_resilience_metrics(
    data: dict,
    result: dict,
    disruption_sigma: float = 0.4,
    W: int = 90,
) -> None:
    """
    Resilience Metrics: single severe disruption (sigma=0.4) on best solution.
    Compare OTDR before vs after disruption.
    """
    evaluator   = CostEvaluatorRRD(data)
    base_routes = result["rrd_routes"]
    N           = len(data["distance_matrix"]) - 1
    depot       = data["depot"]

    # ── Baseline on-time count ────────────────────────────────────────────
    base_on_time = sum(
        1
        for r in base_routes
        for t in r["timing"]
        if t["node"] != depot and t["lateness"] <= 0.01
    )
    base_otdr = base_on_time / N * 100 if N > 0 else 0.0

    # ── Disruption: perturb travel times ─────────────────────────────────
    orig_tt   = evaluator.travel_time.copy()
    noise     = np.random.normal(0, disruption_sigma, orig_tt.shape)
    perturbed = np.clip(orig_tt * (1 + noise), 0, None)
    evaluator.travel_time = perturbed

    dis_on_time  = 0
    max_lateness = 0.0
    failed       = 0
    timing_start = time_mod.time()

    for r in base_routes:
        route_obj = Route(customers=r["nodes"][1:-1])
        info      = evaluator.evaluate_route(route_obj)
        for t in info["timing"]:
            if t["node"] == depot:
                continue
            lat = t["lateness"]
            if lat <= 0.01:
                dis_on_time += 1
            else:
                failed += 1
                if lat > max_lateness:
                    max_lateness = lat

    recovery_ms  = (time_mod.time() - timing_start) * 1000
    evaluator.travel_time = orig_tt  # restore

    dis_otdr    = dis_on_time / N * 100 if N > 0 else 0.0
    otdr_drop   = dis_otdr - base_otdr              # negative = drop
    adapt_freq  = failed / N * 100 if N > 0 else 0.0

    print(f"{'═'*W}")
    print(f"  RESILIENCE METRICS  (Single severe disruption, sigma={disruption_sigma})")
    print(f"{'═'*W}")
    sign = "+" if otdr_drop >= 0 else "-"
    print(f"  {'OTDR drop':<30}: {sign}{abs(otdr_drop):>8.2f} %")
    print(f"  {'Max delay increase':<30}: {max_lateness:>9.1f} min")
    print(f"  {'Failed deliveries':<30}: {failed:>10}")
    print(f"  {'Recovery time':<30}: {recovery_ms:>9.1f} ms")
    print(f"  {'Adaptation frequency':<30}: {adapt_freq:>9.2f} %")
    print(f"{'═'*W}")


def print_best_k_routes(data: dict, result: dict, best_K: int = 0, W: int = 100) -> None:
    """
    In chi tiết tuyến đường từng xe sau khi tìm ra K best.
    Mỗi xe hiển thị: thứ tự stops, tên CH, thời gian đến, TW, trễ, demand, tải lũy kế.
    """
    store_names = data["store_names"]
    tw          = data["time_windows"]
    demands     = data["demands"]
    cong        = data["congestion_matrix"]
    Q           = data["vehicle_capacity"]
    depot       = data["depot"]

    routes_info  = result["rrd_routes"]
    n_active     = len(routes_info)
    k_label      = best_K if best_K > 0 else n_active

    print(f"{'═'*W}")
    print(f"  DETAILED ROUTES — Best K = {k_label}  ({n_active} active vehicles)")
    print(f"{'═'*W}")

    total_transport = 0
    total_late      = 0
    total_cong      = 0
    total_served    = 0

    for r in routes_info:
        nodes  = r["nodes"]
        timing = r["timing"]
        v_id   = r["vehicle"] + 1

        print(f"\n{'─'*W}")
        print(f"  Vehicle {v_id}  |  Load: {r['load_kg']:.1f} / {Q} kg  "
              f"|  Distance: {r['distance_km']:.2f} km  |  Cost: {r['total_cost']:,.0f} VND")
        print(f"{'─'*W}")
        print(
            f"  {'#':<4} {'Store':<28} {'Arrive':<8} {'TW':<14} "
            f"{'Late':<8} {'Demand':<10} {'CumLoad':<10} {'Congestion':<15}"
        )
        print(
            f"  {'─'*4} {'─'*28} {'─'*8} {'─'*14} "
            f"{'─'*8} {'─'*10} {'─'*10} {'─'*15}"
        )

        cum_load = 0.0
        n_customers_in_route = 0

        for idx, t in enumerate(timing):
            node   = t["node"]
            name   = store_names[node][:27]
            s_min  = t["service_start"]
            late   = t["lateness"]

            arrival_str = f"{int(s_min) // 60}:{int(s_min) % 60:02d}"
            tw_s, tw_e  = tw[node]
            tw_str      = f"{tw_s//60}:{tw_s%60:02d}-{tw_e//60}:{tw_e%60:02d}"
            late_str    = f"+{late:.0f}m" if late > 0.01 else "✓"

            # Demand and cumulative load
            d = demands[node]
            cum_load += d
            demand_str  = f"{d:.1f}" if node != depot else "—"
            cumload_str = f"{cum_load:.1f}" if node != depot else "—"

            # Congestion from previous node
            if idx > 0:
                prev = nodes[idx - 1]
                cong_level = cong[prev][node]
                cong_str = CONGESTION_LABELS.get(cong_level, "?")
            else:
                cong_str = "—"

            if node != depot:
                n_customers_in_route += 1

            print(
                f"  {idx:<4} {name:<28} {arrival_str:<8} {tw_str:<14} "
                f"{late_str:<8} {demand_str:<10} {cumload_str:<10} {cong_str:<15}"
            )

        total_transport += r["transport_cost"]
        total_late      += r["late_penalty"]
        total_cong      += r["congestion_cost"]
        total_served    += n_customers_in_route

        # Route summary
        print(f"  ├─ Transport : {r['transport_cost']:>12,.0f} VND")
        if r["late_penalty"] > 0:
            print(f"  ├─ Late pen. : {r['late_penalty']:>12,.0f} VND")
        print(f"  ├─ Congestion: {r['congestion_cost']:>12,.0f} VND")
        print(f"  └─ Subtotal  : {r['total_cost']:>12,.0f} VND")

    # Grand summary
    grand_total = total_transport + total_late + total_cong
    print(f"\n{'═'*W}")
    print(f"  ROUTE SUMMARY — K = {k_label}, {n_active} active vehicles, {total_served} stores served")
    print(f"{'═'*W}")
    print(f"  Transport cost    : {total_transport:>15,.0f} VND")
    print(f"  Delay penalty     : {total_late:>15,.0f} VND")
    print(f"  Congestion cost   : {total_cong:>15,.0f} VND")
    print(f"  ──────────────────────────────────────")
    print(f"  GRAND TOTAL       : {grand_total:>15,.0f} VND")

    # List dropped stores
    all_customers = set(range(1, len(data["distance_matrix"])))
    visited = set()
    for r in routes_info:
        for node in r["nodes"]:
            if node != depot:
                visited.add(node)
    dropped = all_customers - visited
    if dropped:
        dropped_names = [store_names[n] for n in sorted(dropped)]
        print(f"\n  ⚠ Dropped stores ({len(dropped)}): {', '.join(dropped_names)}")
    else:
        print(f"\n  ✅ All {len(all_customers)} stores visited!")
    print(f"{'═'*W}")


def print_demand_sensitivity(
    data: dict,
    best_K: int,
    alns_iterations: int = 50,
    reassign_iter: int = 10,
    W: int = 100,
) -> None:
    """
    Sensitivity analysis: chạy lại solver với demand ±10% cho MỖI store.
    So sánh kết quả (cost, OTDR) với baseline để đánh giá tác động.
    """
    import random as _rnd

    baseline_demands = list(data["demands"])
    num_stores = len(baseline_demands) - 1  # bỏ depot

    print(f"{'═'*W}")
    print(f"  DEMAND SENSITIVITY ANALYSIS  (±10% per store)")
    print(f"  Baseline demands → perturb each store's demand by +10% and −10%")
    print(f"  K = {best_K}  |  {num_stores} stores")
    print(f"{'═'*W}")

    # ── Run baseline ──────────────────────────────────────────────────────
    base_result = rrd_solve(
        data,
        num_vehicles=best_K,
        alns_iterations=alns_iterations,
        reassign_iter=reassign_iter,
        verbose=False,
    )
    base_cost = base_result["rrd_cost"]
    base_m    = calc_metrics(base_result["rrd_routes"], data)
    base_otdr = base_m["otdr"]

    print(f"\n  Baseline        : cost = {base_cost:>15,.0f} VND  |  OTDR = {base_otdr:.2f}%")
    print(f"  {'─'*94}")
    print(
        f"  {'Store':<28} {'Demand':<10} {'Perturbation':<14} "
        f"{'Cost (VND)':<18} {'Δ Cost':<16} {'Δ%':<9} {'OTDR':<8}"
    )
    print(
        f"  {'─'*28} {'─'*10} {'─'*14} "
        f"{'─'*18} {'─'*16} {'─'*9} {'─'*8}"
    )

    store_names = data["store_names"]
    results_sens = []

    for store_id in range(1, num_stores + 1):
        orig_demand = baseline_demands[store_id]

        for pct_label, factor in [("−10%", 0.90), ("+10%", 1.10)]:
            # Perturb demand for this store only
            perturbed_demands = list(baseline_demands)
            perturbed_demands[store_id] = orig_demand * factor
            data["demands"] = perturbed_demands

            p_result = rrd_solve(
                data,
                num_vehicles=best_K,
                alns_iterations=alns_iterations,
                reassign_iter=reassign_iter,
                verbose=False,
            )
            p_cost = p_result["rrd_cost"]
            p_m    = calc_metrics(p_result["rrd_routes"], data)
            p_otdr = p_m["otdr"]

            delta     = p_cost - base_cost
            delta_pct = delta / base_cost * 100 if base_cost > 0 else 0.0
            sign      = "+" if delta >= 0 else ""

            sname = store_names[store_id][:27]
            print(
                f"  {sname:<28} {orig_demand:<10.1f} {pct_label:<14} "
                f"{p_cost:<18,.0f} {sign}{delta:<15,.0f} {sign}{delta_pct:<8.2f}% {p_otdr:<8.2f}%"
            )

            results_sens.append({
                "store_id":   store_id,
                "store_name": store_names[store_id],
                "original":   orig_demand,
                "perturbed":  perturbed_demands[store_id],
                "label":      pct_label,
                "cost":       p_cost,
                "delta":      delta,
                "delta_pct":  delta_pct,
                "otdr":       p_otdr,
            })

    # Restore original demands
    data["demands"] = baseline_demands

    # ── Summary ───────────────────────────────────────────────────────────
    plus_deltas  = [r["delta_pct"] for r in results_sens if r["label"] == "+10%"]
    minus_deltas = [r["delta_pct"] for r in results_sens if r["label"] == "−10%"]

    avg_plus   = sum(plus_deltas) / len(plus_deltas) if plus_deltas else 0
    avg_minus  = sum(minus_deltas) / len(minus_deltas) if minus_deltas else 0
    max_impact = max(results_sens, key=lambda r: abs(r["delta_pct"]))

    print(f"\n  {'─'*94}")
    print(f"  SENSITIVITY SUMMARY")
    print(f"  Avg cost change  (+10% demand): {'+' if avg_plus>=0 else ''}{avg_plus:.2f}%")
    print(f"  Avg cost change  (−10% demand): {'+' if avg_minus>=0 else ''}{avg_minus:.2f}%")
    print(
        f"  Most sensitive store: {max_impact['store_name']} "
        f"({max_impact['label']}) → Δ = {'+' if max_impact['delta_pct']>=0 else ''}"
        f"{max_impact['delta_pct']:.2f}%"
    )
    print(f"{'═'*W}")


def _classify_event(entry: dict) -> str:
    """
    Phân loại một ROLLOUT event vào 1 trong 4 loại:
      - Time Violations      : có lateness tại node được chọn
      - Traffic Incidents     : cung (arc) có congestion ≥ 2
      - Capacity Violations   : tải route ≥ 70% capacity
      - Urgent Deliveries     : còn lại (urgency cao nhưng không rơi vào 3 loại trên)
    """
    lateness = entry.get("lateness", 0)
    cong_arc = entry.get("cong_arc", 0)
    load_pct = entry.get("route_load_pct", 0)

    if lateness > 0.01:
        return "Time Violations"
    if cong_arc >= 2:
        return "Traffic Incidents"
    if load_pct >= 70:
        return "Capacity Violations"
    return "Urgent Deliveries"


def print_realtime_event_performance(
    data: dict,
    result: dict,
    n_trials: int = 10,
    W: int = 110,
) -> None:
    """
    Table 8  —  T-ALNS-RRD real-time event response performance.

    Chỉ re-run Phase 2b (dispatch) với travel-time perturbation nhẹ (σ=0.08)
    để tạo biến thiên giữa các trial → tính mean ± std.
    Không chạy lại ALNS (Phase 1) nên rất nhanh.
    """
    categories = [
        "Traffic Incidents",
        "Urgent Deliveries",
        "Capacity Violations",
        "Time Violations",
    ]

    # Reconstruct solution object từ result
    routes   = [Route(customers=r["nodes"][1:-1]) for r in result["rrd_routes"]]
    base_sol = Solution(routes=routes)

    # ── Per-trial collection ──────────────────────────────────────────────
    trial_data = {
        cat: {"freq": [], "resp": [], "success": [], "cost_imp": [], "sims": []}
        for cat in categories
    }

    for trial in range(n_trials):
        np.random.seed(42 + trial * 13)

        evaluator = CostEvaluatorRRD(data)

        # Perturbation nhẹ giữa các trial để tạo ±std
        if trial > 0:
            noise = np.random.normal(0, 0.08, evaluator.travel_time.shape)
            evaluator.travel_time = np.clip(
                evaluator.travel_time * (1 + noise), 0, None
            )

        dispatcher = EventDrivenRolloutDispatcher(data, evaluator, 0.65)
        _, timing_logs = dispatcher.dispatch_all(base_sol)

        # Classify ROLLOUT events
        events_by_cat = {cat: [] for cat in categories}
        for logs in timing_logs:
            for entry in logs:
                if not entry.get("event", "").startswith("ROLLOUT"):
                    continue
                cat = _classify_event(entry)
                events_by_cat[cat].append(entry)

        # Per-category metrics cho trial này
        for cat in categories:
            evts = events_by_cat[cat]
            n = len(evts)
            trial_data[cat]["freq"].append(n)
            if n > 0:
                trial_data[cat]["resp"].append(
                    np.mean([e.get("response_ms", 0) for e in evts])
                )
                trial_data[cat]["success"].append(
                    sum(1 for e in evts if e.get("lateness", 1) <= 0.01) / n * 100
                )
                trial_data[cat]["cost_imp"].append(
                    np.mean([abs(e.get("cost_impact_pct", 0)) for e in evts])
                )
                trial_data[cat]["sims"].append(
                    np.mean([e.get("n_mc_sims", 0) for e in evts])
                )
            else:
                trial_data[cat]["resp"].append(0.0)
                trial_data[cat]["success"].append(100.0)
                trial_data[cat]["cost_imp"].append(0.0)
                trial_data[cat]["sims"].append(0.0)

    # ── Print table ───────────────────────────────────────────────────────
    print(f"{chr(9552)*W}")
    print(f"  Table 8.  T-ALNS-RRD real-time event response performance")
    print(f"  ({n_trials} trials, travel-time perturbation sigma=0.08)")
    print(f"{chr(9552)*W}")

    hdr = (
        f"  {'Event type':<22} {'Frequency':>14} {'Avg response (ms)':>20} "
        f"{'Success rate (%)':>20} {'Cost impact red.':>20} {'Rollout sims':>16}"
    )
    print(hdr)
    print(
        f"  {chr(9472)*22} {chr(9472)*14} {chr(9472)*20} "
        f"{chr(9472)*20} {chr(9472)*20} {chr(9472)*16}"
    )

    def _fmt(mean: float, std: float, decimals: int = 1, pct: bool = False) -> str:
        """Format mean±std with optional % suffix."""
        sfx = "%" if pct else ""
        return f"{mean:.{decimals}f}{chr(177)}{std:.{decimals}f}{sfx}"

    for cat in categories:
        d  = trial_data[cat]
        fm = _fmt(np.mean(d["freq"]),    np.std(d["freq"]))
        rm = _fmt(np.mean(d["resp"]),    np.std(d["resp"]))
        sm = _fmt(np.mean(d["success"]), np.std(d["success"]))
        cm = _fmt(np.mean(d["cost_imp"]),np.std(d["cost_imp"]), pct=True)
        nm = _fmt(np.mean(d["sims"]),    np.std(d["sims"]))

        print(
            f"  {cat:<22} {fm:>14} {rm:>20} "
            f"{sm:>20} {cm:>20} {nm:>16}"
        )

    # ── Overall row (weighted across categories per trial) ────────────────
    print(
        f"  {chr(9472)*22} {chr(9472)*14} {chr(9472)*20} "
        f"{chr(9472)*20} {chr(9472)*20} {chr(9472)*16}"
    )

    o_freq, o_resp, o_succ, o_cost, o_sims = [], [], [], [], []
    for t in range(n_trials):
        tf = sum(trial_data[cat]["freq"][t] for cat in categories)
        o_freq.append(tf)
        if tf > 0:
            w_r = sum(trial_data[c]["freq"][t] * trial_data[c]["resp"][t]
                      for c in categories)
            w_s = sum(trial_data[c]["freq"][t] * trial_data[c]["success"][t]
                      for c in categories)
            w_c = sum(trial_data[c]["freq"][t] * trial_data[c]["cost_imp"][t]
                      for c in categories)
            w_n = sum(trial_data[c]["freq"][t] * trial_data[c]["sims"][t]
                      for c in categories)
            o_resp.append(w_r / tf)
            o_succ.append(w_s / tf)
            o_cost.append(w_c / tf)
            o_sims.append(w_n / tf)
        else:
            o_resp.append(0); o_succ.append(100)
            o_cost.append(0); o_sims.append(0)

    fm = _fmt(np.mean(o_freq), np.std(o_freq))
    rm = _fmt(np.mean(o_resp), np.std(o_resp))
    sm = _fmt(np.mean(o_succ), np.std(o_succ))
    cm = _fmt(np.mean(o_cost), np.std(o_cost), pct=True)
    nm = _fmt(np.mean(o_sims), np.std(o_sims))

    print(
        f"  {'Overall':<22} {fm:>14} {rm:>20} "
        f"{sm:>20} {cm:>20} {nm:>16}"
    )
    print(f"{chr(9552)*W}")


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
    print(f"  Max iter   : {MAX_ITER}  |  Fleet sweep : K = 3 → 13")
    print(f"{'═'*72}\n")

    summary = []

    for K in range(3, 14):
        t0 = time_mod.time()
        result = rrd_solve(
            data,
            num_vehicles=K,
            alns_iterations=MAX_ITER,
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

    # ── Composite score: normalize 3 metrics, equal 1/3 weight each ─────
    #   cost  → minimize  (norm: 0=best, 1=worst)
    #   OTDR  → maximize  (norm inverted: 0=best, 1=worst)
    #   CES   → minimize  (norm: 0=best, 1=worst)
    costs  = [r["total_cost"] for r in summary]
    otdrs  = [r["otdr"]       for r in summary]
    ces_v  = [r["ces"]        for r in summary]

    def _norm(vals, invert=False):
        lo, hi = min(vals), max(vals)
        span = hi - lo if hi > lo else 1.0
        if invert:
            return [(hi - v) / span for v in vals]   # bigger original → 0 (better)
        return [(v - lo) / span for v in vals]        # smaller original → 0 (better)

    nc  = _norm(costs,  invert=False)
    no  = _norm(otdrs,  invert=True)    # high OTDR is good → invert
    nce = _norm(ces_v,  invert=False)

    # Weights: OTDR dominates (service quality), cost secondary, CES minor
    W_COST, W_OTDR, W_CES = 0.3, 0.6, 0.1
    for i, r in enumerate(summary):
        r["_score"] = W_COST * nc[i] + W_OTDR * no[i] + W_CES * nce[i]

    best_r = min(summary, key=lambda x: x["_score"])

    # ── Final summary table ───────────────────────────────────────────────
    print(f"\n{'═'*90}")
    print(f"  RESULTS — Tabu-ALNS-Rollout  (max_iter={MAX_ITER}, N={num_customers} stores)")
    print(f"  Ranking: composite score = 0.6×(1−OTDR_norm) + 0.3×cost_norm + 0.1×CES_norm  [CES scaled ×5]")
    print(f"{'═'*90}")
    print(
        f"  {'K':>3}  {'Total Cost (VND)':>18}  "
        f"{'OTDR (%)':>10}  {'CES Score':>10}  {'Score':>7}  {'Time (s)':>9}"
    )
    print(f"  {'─'*3}  {'─'*18}  {'─'*10}  {'─'*10}  {'─'*7}  {'─'*9}")

    for r in summary:
        star = " ★" if r is best_r else "  "
        print(
            f"  {r['K']:>3}  {r['total_cost']:>18,.0f}  "
            f"{r['otdr']:>9.2f}%  {r['ces']:>10.2f}  "
            f"{r['_score']:>7.4f}  {r['time']:>8.1f}s{star}"
        )

    print(f"{'═'*90}")
    print(
        f"\n  ★  Best balanced K = {best_r['K']}  →  "
        f"{best_r['total_cost']:,.0f} VND  |  "
        f"OTDR = {best_r['otdr']:.2f}%  |  "
        f"CES = {best_r['ces']:.2f}  |  "
        f"Score = {best_r['_score']:.4f}\n"
    )

    # ── Re-solve best K to get full result dict (for detailed metrics) ───
    print(f"\n  ⏳  Re-running best K={best_r['K']} for detailed metric reports ...")
    best_full_result = rrd_solve(
        data,
        num_vehicles=best_r["K"],
        alns_iterations=MAX_ITER,
        reassign_iter=10,
        verbose=False,
    )
    best_m = calc_metrics(best_full_result["rrd_routes"], data)

    print()
    print_ces_report(best_m)
    print()
    print_event_response_metrics(best_full_result)
    print()
    np.random.seed(42)
    print_robustness_analysis(data, best_full_result)
    print()
    np.random.seed(42)
    print_resilience_metrics(data, best_full_result)

    # ── Table 8: Real-time event response performance ────────────────────
    print()
    print_realtime_event_performance(data, best_full_result, n_trials=10)

    # ── In chi tiết tuyến đường từng xe (best K) ────────────────────────
    print()
    print_best_k_routes(data, best_full_result, best_K=best_r["K"])

    # ── Phân tích nhạy cảm demand ±10% ──────────────────────────────────
    print()
    print_demand_sensitivity(
        data,
        best_K=best_r["K"],
        alns_iterations=MAX_ITER,
        reassign_iter=10,
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