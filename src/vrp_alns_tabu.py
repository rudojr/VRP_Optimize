"""
ALNS + Tabu Search meta-heuristic for CVRPTW
─────────────────────────────────────────────
Giải cùng bài toán với vrp_milp.py để so sánh:
  • Cùng dữ liệu (store.csv, distant_matrix.csv)
  • Cùng hàm mục tiêu: transport + late_penalty + congestion
  • Cùng ràng buộc: capacity, time-window (soft upper → penalty)
"""

import copy
import math
import random
import time as time_mod
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

# ── Re-use data loading from vrp_milp ────────────────────────────────────────

def generate_congestion_matrix(num_nodes, seed=42):
    rng = np.random.default_rng(seed)
    congestion = rng.choice([1, 2, 3], size=(num_nodes, num_nodes), p=[0.6, 0.3, 0.1])
    np.fill_diagonal(congestion, 0)
    congestion = np.triu(congestion) + np.triu(congestion, 1).T
    return congestion


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
    distance_matrix = df_matrix.values

    num_nodes = distance_matrix.shape[0]
    congestion_matrix = generate_congestion_matrix(num_nodes, seed=42)

    store_names = ["Depot"] + df_store["Tên CH"].tolist()
    demands = [0] + df_store["Demand_kg"].tolist()

    depot_tw = (6 * 60, 18 * 60)
    time_windows = [depot_tw] + list(
        zip(df_store["tw_start"].tolist(), df_store["tw_end"].tolist())
    )

    data = {
        "distance_matrix": distance_matrix,
        "congestion_matrix": congestion_matrix,
        "demands": demands,
        "time_windows": time_windows,
        "depot": 0,
        "vehicle_capacity": 1800,
        "cost_per_km": 18_000,
        "late_penalty_per_min": 50_000,
        "congestion_penalty": 5_000,
        "service_time_min": 10,
        "avg_speed_kmh": 30,
        "store_names": store_names,
    }
    return data


CONGESTION_LABELS = {0: "—", 1: "Bình thường", 2: "Hơi tắc", 3: "Đặc biệt tắc"}


# ═══════════════════════════════════════════════════════════════════════════════
#  SOLUTION REPRESENTATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Route:
    """Một tuyến đường: danh sách customer nodes (KHÔNG bao gồm depot)."""
    customers: List[int] = field(default_factory=list)


@dataclass
class Solution:
    """Một lời giải gồm nhiều routes."""
    routes: List[Route] = field(default_factory=list)
    cost: float = float("inf")


# ═══════════════════════════════════════════════════════════════════════════════
#  COST EVALUATOR
# ═══════════════════════════════════════════════════════════════════════════════

class CostEvaluator:
    """Tính chi phí giống hệt MILP: transport + late_penalty + congestion."""

    def __init__(self, data: dict):
        self.dist = data["distance_matrix"]
        self.cong = data["congestion_matrix"]
        self.demands = data["demands"]
        self.tw = data["time_windows"]
        self.Q = data["vehicle_capacity"]
        self.cpk = data["cost_per_km"]
        self.lam1 = data["late_penalty_per_min"]
        self.lam2 = data["congestion_penalty"]
        self.s_time = data["service_time_min"]
        self.v_avg = data["avg_speed_kmh"]
        self.depot = data["depot"]

        n = len(self.dist)
        self.travel_time = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    self.travel_time[i][j] = (self.dist[i][j] / self.v_avg) * self.cong[i][j] * 60

    def route_load(self, route: Route) -> float:
        return sum(self.demands[c] for c in route.customers)

    def is_feasible_capacity(self, route: Route) -> bool:
        return self.route_load(route) <= self.Q

    def evaluate_route(self, route: Route) -> dict:
        """Tính chi tiết chi phí cho 1 route."""
        if not route.customers:
            return {
                "distance": 0, "transport": 0, "late_penalty": 0,
                "congestion": 0, "total": 0, "load": 0, "timing": [],
                "feasible_capacity": True,
            }

        nodes = [self.depot] + route.customers + [self.depot]
        total_dist = 0
        total_cong_cost = 0

        # Tính timing
        timing = []
        current_time = float(self.tw[self.depot][0])  # khởi hành từ depot

        timing.append({
            "node": self.depot,
            "service_start": current_time,
            "lateness": 0,
        })

        total_late_penalty = 0

        for idx in range(1, len(nodes)):
            i, j = nodes[idx - 1], nodes[idx]
            seg_dist = self.dist[i][j]
            seg_cong = self.cong[i][j]
            seg_travel = self.travel_time[i][j]

            total_dist += seg_dist
            total_cong_cost += self.lam2 * seg_cong

            # thời gian phục vụ tại node trước (trừ depot)
            service = self.s_time if i != self.depot else 0
            arrival = current_time + service + seg_travel

            # Chờ nếu đến sớm (hard lower bound)
            if j != self.depot:
                arrival = max(arrival, float(self.tw[j][0]))

            # Tính trễ (soft upper bound)
            lateness = 0
            if j != self.depot:
                lateness = max(0, arrival - float(self.tw[j][1]))
                total_late_penalty += self.lam1 * lateness

            timing.append({
                "node": j,
                "service_start": arrival,
                "lateness": lateness,
            })
            current_time = arrival

        transport_cost = total_dist * self.cpk
        load = self.route_load(route)

        return {
            "distance": total_dist,
            "transport": transport_cost,
            "late_penalty": total_late_penalty,
            "congestion": total_cong_cost,
            "total": transport_cost + total_late_penalty + total_cong_cost,
            "load": load,
            "timing": timing,
            "feasible_capacity": load <= self.Q,
        }

    def evaluate_solution(self, sol: Solution) -> float:
        """Tính tổng chi phí cho toàn bộ solution."""
        total = 0
        for route in sol.routes:
            info = self.evaluate_route(route)
            if not info["feasible_capacity"]:
                total += 1e12  # phạt rất nặng nếu vi phạm capacity
            total += info["total"]
        sol.cost = total
        return total

    def insertion_cost(self, route: Route, customer: int, position: int) -> float:
        """Chi phí tăng thêm khi chèn customer vào vị trí position trong route."""
        new_route = Route(customers=route.customers[:])
        new_route.customers.insert(position, customer)

        if not self.is_feasible_capacity(new_route):
            return float("inf")

        old_cost = self.evaluate_route(route)["total"]
        new_cost = self.evaluate_route(new_route)["total"]
        return new_cost - old_cost


# ═══════════════════════════════════════════════════════════════════════════════
#  INITIAL SOLUTION CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════════

def build_initial_solution(data: dict, num_vehicles: int, evaluator: CostEvaluator) -> Solution:
    """
    Xây dựng lời giải ban đầu bằng phương pháp chèn tuần tự
    dựa trên time-window (earliest deadline first) + greedy insertion.
    """
    n = len(data["distance_matrix"])
    customers = list(range(1, n))
    # Sắp xếp theo thời gian bắt đầu time window
    customers.sort(key=lambda c: (data["time_windows"][c][0], data["time_windows"][c][1]))

    sol = Solution(routes=[Route() for _ in range(num_vehicles)])
    unassigned = list(customers)

    for cust in unassigned[:]:
        best_cost = float("inf")
        best_route_idx = -1
        best_pos = -1

        for r_idx, route in enumerate(sol.routes):
            for pos in range(len(route.customers) + 1):
                cost = evaluator.insertion_cost(route, cust, pos)
                if cost < best_cost:
                    best_cost = cost
                    best_route_idx = r_idx
                    best_pos = pos

        if best_route_idx >= 0 and best_cost < float("inf"):
            sol.routes[best_route_idx].customers.insert(best_pos, cust)
            unassigned.remove(cust)

    # Nếu còn unassigned, ép vào route có tải nhẹ nhất
    for cust in unassigned:
        loads = [(evaluator.route_load(r), idx) for idx, r in enumerate(sol.routes)]
        loads.sort()
        for _, r_idx in loads:
            best_pos = 0
            best_c = float("inf")
            for pos in range(len(sol.routes[r_idx].customers) + 1):
                trial = Route(customers=sol.routes[r_idx].customers[:])
                trial.customers.insert(pos, cust)
                c = evaluator.evaluate_route(trial)["total"]
                if c < best_c:
                    best_c = c
                    best_pos = pos
            sol.routes[r_idx].customers.insert(best_pos, cust)
            break

    evaluator.evaluate_solution(sol)
    return sol


# ═══════════════════════════════════════════════════════════════════════════════
#  DESTROY OPERATORS (4 chiến lược phá)
# ═══════════════════════════════════════════════════════════════════════════════

def _remove_customers(sol: Solution, customers_to_remove: List[int]) -> Solution:
    """Tạo solution mới loại bỏ các customer được chọn."""
    new_sol = Solution(routes=[Route(customers=[c for c in r.customers if c not in customers_to_remove]) for r in sol.routes])
    return new_sol


def destroy_random(sol: Solution, num_remove: int, evaluator: CostEvaluator, **kwargs) -> Tuple[Solution, List[int]]:
    """Xóa ngẫu nhiên num_remove customers."""
    all_customers = [c for r in sol.routes for c in r.customers]
    k = min(num_remove, len(all_customers))
    removed = random.sample(all_customers, k)
    return _remove_customers(sol, removed), removed


def destroy_worst(sol: Solution, num_remove: int, evaluator: CostEvaluator, **kwargs) -> Tuple[Solution, List[int]]:
    """Xóa các customer có chi phí đóng góp (cost contribution) lớn nhất."""
    contributions = []
    for r_idx, route in enumerate(sol.routes):
        if not route.customers:
            continue
        base_cost = evaluator.evaluate_route(route)["total"]
        for c_idx, cust in enumerate(route.customers):
            trial = Route(customers=[c for c in route.customers if c != cust])
            trial_cost = evaluator.evaluate_route(trial)["total"]
            saving = base_cost - trial_cost  # chi phí giảm khi xóa customer này
            contributions.append((saving, cust))

    contributions.sort(reverse=True)
    # Randomize với xác suất p^rank
    p_worst = kwargs.get("p_worst", 3)
    removed = []
    candidates = [c for _, c in contributions]

    k = min(num_remove, len(candidates))
    while len(removed) < k and candidates:
        idx = int(len(candidates) * (random.random() ** p_worst))
        idx = min(idx, len(candidates) - 1)
        removed.append(candidates.pop(idx))

    return _remove_customers(sol, removed), removed


def destroy_shaw(sol: Solution, num_remove: int, evaluator: CostEvaluator, **kwargs) -> Tuple[Solution, List[int]]:
    """
    Shaw removal: xóa các customer liên quan (gần nhau về khoảng cách,
    demand, time window) để tạo cơ hội tái cấu trúc tốt hơn.
    """
    dist = evaluator.dist
    demands = evaluator.demands
    tw = evaluator.tw

    all_customers = [c for r in sol.routes for c in r.customers]
    if not all_customers:
        return copy.deepcopy(sol), []

    # Chuẩn hóa các thuộc tính
    max_dist = np.max(dist) if np.max(dist) > 0 else 1
    max_demand = max(demands) if max(demands) > 0 else 1
    max_tw_diff = max(abs(tw[c][0] - tw[c2][0]) for c in all_customers for c2 in all_customers) if len(all_customers) > 1 else 1
    max_tw_diff = max(max_tw_diff, 1)

    phi_d, phi_q, phi_t = 9, 3, 2  # trọng số

    def relatedness(c1, c2):
        return (phi_d * dist[c1][c2] / max_dist +
                phi_q * abs(demands[c1] - demands[c2]) / max_demand +
                phi_t * abs(tw[c1][0] - tw[c2][0]) / max_tw_diff)

    seed_cust = random.choice(all_customers)
    removed = [seed_cust]
    candidates = [c for c in all_customers if c != seed_cust]

    p_shaw = kwargs.get("p_shaw", 6)
    k = min(num_remove, len(all_customers))

    while len(removed) < k and candidates:
        ref = random.choice(removed)
        candidates.sort(key=lambda c: relatedness(ref, c))
        idx = int(len(candidates) * (random.random() ** p_shaw))
        idx = min(idx, len(candidates) - 1)
        removed.append(candidates.pop(idx))

    return _remove_customers(sol, removed), removed


def destroy_timewindow(sol: Solution, num_remove: int, evaluator: CostEvaluator, **kwargs) -> Tuple[Solution, List[int]]:
    """Xóa các customer có lateness lớn nhất (ưu tiên giải quyết trễ giờ)."""
    late_customers = []
    for route in sol.routes:
        info = evaluator.evaluate_route(route)
        for t in info["timing"]:
            node = t["node"]
            if node != evaluator.depot and t["lateness"] > 0:
                late_customers.append((t["lateness"], node))

    if not late_customers:
        # Nếu không có ai trễ, fallback về random
        return destroy_random(sol, num_remove, evaluator)

    late_customers.sort(reverse=True)
    k = min(num_remove, len(late_customers))
    removed = [c for _, c in late_customers[:k]]

    # Nếu chưa đủ, bổ sung random
    if len(removed) < num_remove:
        remaining = [c for r in sol.routes for c in r.customers if c not in removed]
        extra = min(num_remove - len(removed), len(remaining))
        if extra > 0:
            removed += random.sample(remaining, extra)

    return _remove_customers(sol, removed), removed


# ═══════════════════════════════════════════════════════════════════════════════
#  REPAIR OPERATORS (3 chiến lược sửa)
# ═══════════════════════════════════════════════════════════════════════════════

def repair_greedy(sol: Solution, removed: List[int], evaluator: CostEvaluator, **kwargs) -> Solution:
    """Chèn từng customer vào vị trí tốt nhất (best position, best route)."""
    new_sol = Solution(routes=[Route(customers=r.customers[:]) for r in sol.routes])
    uninserted = list(removed)
    random.shuffle(uninserted)

    for cust in uninserted:
        best_cost = float("inf")
        best_r = -1
        best_p = -1

        for r_idx, route in enumerate(new_sol.routes):
            for pos in range(len(route.customers) + 1):
                cost = evaluator.insertion_cost(route, cust, pos)
                if cost < best_cost:
                    best_cost = cost
                    best_r = r_idx
                    best_p = pos

        if best_r >= 0:
            new_sol.routes[best_r].customers.insert(best_p, cust)

    evaluator.evaluate_solution(new_sol)
    return new_sol


def repair_regret2(sol: Solution, removed: List[int], evaluator: CostEvaluator, **kwargs) -> Solution:
    """
    Regret-2 insertion: ưu tiên chèn customer có regret lớn nhất
    (chênh lệch chi phí giữa vị trí tốt nhất và tốt nhì).
    """
    new_sol = Solution(routes=[Route(customers=r.customers[:]) for r in sol.routes])
    uninserted = list(removed)

    while uninserted:
        regrets = []
        for cust in uninserted:
            costs = []
            for r_idx, route in enumerate(new_sol.routes):
                for pos in range(len(route.customers) + 1):
                    c = evaluator.insertion_cost(route, cust, pos)
                    costs.append((c, r_idx, pos))
            costs.sort(key=lambda x: x[0])

            if not costs:
                continue

            best = costs[0]
            second_best_cost = costs[1][0] if len(costs) > 1 else float("inf")
            regret = second_best_cost - best[0]
            regrets.append((regret, cust, best[1], best[2]))

        if not regrets:
            break

        # Chọn customer có regret lớn nhất
        regrets.sort(key=lambda x: (-x[0], x[2]))
        _, cust, r_idx, pos = regrets[0]
        new_sol.routes[r_idx].customers.insert(pos, cust)
        uninserted.remove(cust)

    evaluator.evaluate_solution(new_sol)
    return new_sol


def repair_random(sol: Solution, removed: List[int], evaluator: CostEvaluator, **kwargs) -> Solution:
    """Chèn vào vị trí ngẫu nhiên khả thi."""
    new_sol = Solution(routes=[Route(customers=r.customers[:]) for r in sol.routes])
    uninserted = list(removed)
    random.shuffle(uninserted)

    for cust in uninserted:
        feasible_insertions = []
        for r_idx, route in enumerate(new_sol.routes):
            for pos in range(len(route.customers) + 1):
                trial = Route(customers=route.customers[:])
                trial.customers.insert(pos, cust)
                if evaluator.is_feasible_capacity(trial):
                    c = evaluator.insertion_cost(route, cust, pos)
                    feasible_insertions.append((c, r_idx, pos))

        if feasible_insertions:
            # Chọn ngẫu nhiên trong top 50% tốt nhất
            feasible_insertions.sort(key=lambda x: x[0])
            top_k = max(1, len(feasible_insertions) // 2)
            _, r_idx, pos = random.choice(feasible_insertions[:top_k])
            new_sol.routes[r_idx].customers.insert(pos, cust)
        else:
            # Ép vào route ngẫu nhiên
            r_idx = random.randint(0, len(new_sol.routes) - 1)
            pos = random.randint(0, len(new_sol.routes[r_idx].customers))
            new_sol.routes[r_idx].customers.insert(pos, cust)

    evaluator.evaluate_solution(new_sol)
    return new_sol


# ═══════════════════════════════════════════════════════════════════════════════
#  TABU LIST
# ═══════════════════════════════════════════════════════════════════════════════

class TabuList:
    """
    Tabu list lưu các cặp (customer, route_index) bị cấm.
    Ngăn customer vừa bị xóa khỏi route k quay lại route k trong tenure bước.
    """

    def __init__(self, tenure: int = 10):
        self.tenure = tenure
        self.tabu: Dict[Tuple[int, int], int] = {}  # (customer, route_idx) → hết hạn ở iteration nào

    def add(self, customer: int, route_idx: int, current_iter: int):
        self.tabu[(customer, route_idx)] = current_iter + self.tenure

    def is_tabu(self, customer: int, route_idx: int, current_iter: int) -> bool:
        key = (customer, route_idx)
        if key in self.tabu:
            if self.tabu[key] > current_iter:
                return True
            else:
                del self.tabu[key]
        return False

    def cleanup(self, current_iter: int):
        """Xóa các entry đã hết hạn."""
        expired = [k for k, v in self.tabu.items() if v <= current_iter]
        for k in expired:
            del self.tabu[k]


# ═══════════════════════════════════════════════════════════════════════════════
#  ADAPTIVE WEIGHT MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

class AdaptiveWeights:
    """Quản lý trọng số thích ứng cho các operator."""

    def __init__(self, num_operators: int, reaction_factor: float = 0.1):
        self.n = num_operators
        self.weights = [1.0] * num_operators        # trọng số
        self.scores = [0.0] * num_operators          # điểm tích lũy trong segment
        self.usage_count = [0] * num_operators       # số lần sử dụng trong segment
        self.reaction_factor = reaction_factor       # tốc độ cập nhật

    def select(self) -> int:
        """Chọn operator theo roulette wheel (xác suất tỷ lệ trọng số)."""
        total = sum(self.weights)
        probs = [w / total for w in self.weights]
        return random.choices(range(self.n), weights=probs, k=1)[0]

    def update_score(self, idx: int, score: float):
        self.scores[idx] += score
        self.usage_count[idx] += 1

    def update_weights(self):
        """Cập nhật trọng số dựa trên performance trong segment vừa qua."""
        for i in range(self.n):
            if self.usage_count[i] > 0:
                avg_score = self.scores[i] / self.usage_count[i]
                self.weights[i] = (
                    self.weights[i] * (1 - self.reaction_factor)
                    + self.reaction_factor * avg_score
                )
                self.weights[i] = max(self.weights[i], 0.1)  # sàn tối thiểu
        # Reset scores
        self.scores = [0.0] * self.n
        self.usage_count = [0] * self.n


# ═══════════════════════════════════════════════════════════════════════════════
#  LOCAL SEARCH (Tabu-enhanced)
# ═══════════════════════════════════════════════════════════════════════════════

def local_search_2opt(sol: Solution, evaluator: CostEvaluator, max_no_improve: int = 50) -> Solution:
    """Intra-route 2-opt improvement."""
    improved = True
    iterations = 0
    while improved and iterations < max_no_improve:
        improved = False
        iterations += 1
        for route in sol.routes:
            if len(route.customers) < 3:
                continue
            best_cost = evaluator.evaluate_route(route)["total"]
            for i in range(len(route.customers) - 1):
                for j in range(i + 2, len(route.customers)):
                    new_customers = (
                        route.customers[:i]
                        + route.customers[i:j+1][::-1]
                        + route.customers[j+1:]
                    )
                    trial = Route(customers=new_customers)
                    cost = evaluator.evaluate_route(trial)["total"]
                    if cost < best_cost - 1e-6:
                        route.customers = new_customers
                        best_cost = cost
                        improved = True
    evaluator.evaluate_solution(sol)
    return sol


def local_search_relocate(sol: Solution, evaluator: CostEvaluator,
                          tabu: TabuList, current_iter: int) -> Solution:
    """
    Inter-route relocate: di chuyển 1 customer giữa các routes.
    Sử dụng Tabu list để tránh cycling.
    """
    best_sol = sol
    best_cost = sol.cost
    improved = True

    while improved:
        improved = False
        for r1_idx in range(len(sol.routes)):
            for c_pos in range(len(sol.routes[r1_idx].customers)):
                cust = sol.routes[r1_idx].customers[c_pos]
                for r2_idx in range(len(sol.routes)):
                    if r1_idx == r2_idx:
                        continue
                    # Check tabu
                    if tabu.is_tabu(cust, r2_idx, current_iter):
                        continue

                    for ins_pos in range(len(sol.routes[r2_idx].customers) + 1):
                        # Tạo solution thử
                        new_sol = Solution(routes=[Route(customers=r.customers[:]) for r in sol.routes])
                        new_sol.routes[r1_idx].customers.pop(c_pos)
                        new_sol.routes[r2_idx].customers.insert(ins_pos, cust)

                        if not evaluator.is_feasible_capacity(new_sol.routes[r2_idx]):
                            continue

                        evaluator.evaluate_solution(new_sol)
                        if new_sol.cost < best_cost - 1e-6:
                            best_sol = new_sol
                            best_cost = new_sol.cost
                            # Tabu: cấm customer quay lại route cũ
                            tabu.add(cust, r1_idx, current_iter)
                            improved = True

        if improved:
            sol = best_sol

    return best_sol


# ═══════════════════════════════════════════════════════════════════════════════
#  ALNS + TABU — MAIN SOLVER
# ═══════════════════════════════════════════════════════════════════════════════

def alns_tabu_solve(
    data: dict,
    num_vehicles: int,
    max_iterations: int = 5000,
    segment_size: int = 100,
    tabu_tenure: int = 15,
    sa_start_temp: float = 1000,
    sa_end_temp: float = 1,
    sa_cooling: float = 0.9995,
    destroy_fraction: Tuple[float, float] = (0.1, 0.4),
    seed: int = 42,
    verbose: bool = False,
) -> Tuple[str, float, list, float]:
    """
    ALNS + Tabu Search meta-heuristic.

    Returns: (status, total_cost, routes_info, solve_time)
    """
    random.seed(seed)
    np.random.seed(seed)

    start_time = time_mod.time()
    evaluator = CostEvaluator(data)
    n_customers = len(data["distance_matrix"]) - 1

    # ── Destroy & Repair operators ──
    destroy_ops = [destroy_random, destroy_worst, destroy_shaw, destroy_timewindow]
    repair_ops = [repair_greedy, repair_regret2, repair_random]

    destroy_names = ["Random", "Worst", "Shaw", "TimeWindow"]
    repair_names = ["Greedy", "Regret-2", "Random"]

    destroy_weights = AdaptiveWeights(len(destroy_ops))
    repair_weights = AdaptiveWeights(len(repair_ops))

    # ── Reward scores ──
    SCORE_NEW_BEST = 33
    SCORE_BETTER = 9
    SCORE_ACCEPTED = 3

    # ── Tabu list ──
    tabu = TabuList(tenure=tabu_tenure)

    # ── Initial solution ──
    current = build_initial_solution(data, num_vehicles, evaluator)
    best = Solution(
        routes=[Route(customers=r.customers[:]) for r in current.routes],
        cost=current.cost,
    )

    if verbose:
        print(f"  Initial solution cost: {current.cost:,.0f} VND")

    # ── Simulated Annealing temperature ──
    temperature = sa_start_temp

    # ── Main loop ──
    no_improve_count = 0
    best_iteration = 0

    for iteration in range(1, max_iterations + 1):
        # Số customer cần xóa
        lo = max(1, int(n_customers * destroy_fraction[0]))
        hi = max(lo + 1, int(n_customers * destroy_fraction[1]))
        num_remove = random.randint(lo, hi)

        # Chọn operators
        d_idx = destroy_weights.select()
        r_idx = repair_weights.select()

        # Destroy
        destroyed, removed = destroy_ops[d_idx](current, num_remove, evaluator)

        # Ghi nhận vào tabu: customer bị xóa từ route nào
        for route_idx, route in enumerate(current.routes):
            for cust in removed:
                if cust in route.customers:
                    tabu.add(cust, route_idx, iteration)

        # Repair (áp dụng tabu khi repair)
        candidate = repair_ops[r_idx](destroyed, removed, evaluator)

        # Local search
        candidate = local_search_2opt(candidate, evaluator, max_no_improve=20)

        # ── Acceptance (SA + aspiration) ──
        delta = candidate.cost - current.cost
        accepted = False
        score = 0

        if candidate.cost < best.cost - 1e-6:
            # New global best → luôn chấp nhận (aspiration)
            accepted = True
            score = SCORE_NEW_BEST
            best = Solution(
                routes=[Route(customers=r.customers[:]) for r in candidate.routes],
                cost=candidate.cost,
            )
            best_iteration = iteration
            no_improve_count = 0

            if verbose and iteration % 100 == 0:
                print(f"  Iter {iteration:>5}: NEW BEST = {best.cost:,.0f} VND "
                      f"[{destroy_names[d_idx]}+{repair_names[r_idx]}]")

        elif delta < 0:
            # Tốt hơn current
            accepted = True
            score = SCORE_BETTER
            no_improve_count = 0

        elif random.random() < math.exp(-delta / max(temperature, 1e-10)):
            # SA acceptance
            accepted = True
            score = SCORE_ACCEPTED
            no_improve_count += 1
        else:
            no_improve_count += 1

        if accepted:
            current = candidate

        # Cập nhật scores
        destroy_weights.update_score(d_idx, score)
        repair_weights.update_score(r_idx, score)

        # Cập nhật trọng số theo segment
        if iteration % segment_size == 0:
            destroy_weights.update_weights()
            repair_weights.update_weights()

            if verbose:
                elapsed = time_mod.time() - start_time
                print(f"  Iter {iteration:>5}: best={best.cost:,.0f}  "
                      f"current={current.cost:,.0f}  T={temperature:.1f}  "
                      f"time={elapsed:.1f}s")

        # Cooling
        temperature *= sa_cooling

        # Tabu cleanup
        if iteration % 50 == 0:
            tabu.cleanup(iteration)

        # Periodical relocate with tabu
        if iteration % 200 == 0:
            current = local_search_relocate(current, evaluator, tabu, iteration)
            if current.cost < best.cost - 1e-6:
                best = Solution(
                    routes=[Route(customers=r.customers[:]) for r in current.routes],
                    cost=current.cost,
                )
                best_iteration = iteration

        # Restart nếu stuck quá lâu
        if no_improve_count > 500:
            current = Solution(
                routes=[Route(customers=r.customers[:]) for r in best.routes],
                cost=best.cost,
            )
            temperature = sa_start_temp * 0.5
            no_improve_count = 0
            if verbose:
                print(f"  Iter {iteration:>5}: RESTART from best")

    solve_time = time_mod.time() - start_time

    if verbose:
        print(f"\n  Finished at iteration {max_iterations}")
        print(f"  Best found at iteration {best_iteration}")
        print(f"  Final best cost: {best.cost:,.0f} VND")
        print(f"  Solve time: {solve_time:.2f}s")
        print(f"\n  Destroy weights: {[f'{w:.2f}' for w in destroy_weights.weights]}")
        print(f"  Repair  weights: {[f'{w:.2f}' for w in repair_weights.weights]}")

    # ── Build output giống format MILP ──
    routes_info = []
    for k, route in enumerate(best.routes):
        if not route.customers:
            continue

        info = evaluator.evaluate_route(route)
        nodes = [data["depot"]] + route.customers + [data["depot"]]

        routes_info.append({
            "vehicle": k,
            "nodes": nodes,
            "distance_km": info["distance"],
            "load_kg": info["load"],
            "transport_cost": info["transport"],
            "late_penalty": info["late_penalty"],
            "congestion_cost": info["congestion"],
            "total_cost": info["total"],
            "timing": info["timing"],
        })

    total_cost = sum(r["total_cost"] for r in routes_info)

    status = "Heuristic"
    return status, total_cost, routes_info, solve_time


# ═══════════════════════════════════════════════════════════════════════════════
#  OUTPUT (cùng format với vrp_milp.py)
# ═══════════════════════════════════════════════════════════════════════════════

def print_solution(data, num_vehicles, status, total_cost, routes_info, solve_time):
    store_names = data["store_names"]
    tw = data["time_windows"]
    cong = data["congestion_matrix"]

    print(f"\n{'═' * 90}")
    print(f"  ALNS+TABU SOLUTION — K = {num_vehicles} vehicles")
    print(f"  Status: {status} | Solve time: {solve_time:.1f}s")
    print(f"{'═' * 90}")

    if total_cost is None:
        print("No feasible solution found!")
        return

    total_transport = 0
    total_late = 0
    total_cong = 0

    for r in routes_info:
        nodes = r["nodes"]
        print(f"\n{'─' * 90}")
        print(f"  Vehicle {r['vehicle'] + 1}")
        print(f"{'─' * 90}")
        print(f"  {'Stop':<5} {'Store':<30} {'Arrive':<8} {'TW':<15} {'Late':<8} {'Congestion':<15}")
        print(f"  {'─'*5} {'─'*30} {'─'*8} {'─'*15} {'─'*8} {'─'*15}")

        for idx, t in enumerate(r["timing"]):
            node = t["node"]
            name = store_names[node]
            s_min = t["service_start"]
            late = t["lateness"]

            arrival_str = f"{int(s_min) // 60}:{int(s_min) % 60:02d}"
            tw_s, tw_e = tw[node]
            tw_str = f"{tw_s//60}:{tw_s%60:02d}-{tw_e//60}:{tw_e%60:02d}"
            late_str = f"+{late:.0f}m" if late > 0.01 else ""

            if idx > 0:
                prev = nodes[idx - 1]
                cong_level = cong[prev][node]
                cong_str = CONGESTION_LABELS.get(cong_level, "?")
            else:
                cong_str = "—"

            print(f"  {idx:<5} {name:<30} {arrival_str:<8} {tw_str:<15} {late_str:<8} {cong_str:<15}")

        total_transport += r["transport_cost"]
        total_late += r["late_penalty"]
        total_cong += r["congestion_cost"]

        print(f"\n  Distance : {r['distance_km']:>8.2f} km")
        print(f"  Load     : {r['load_kg']:>8.1f} kg  /  {data['vehicle_capacity']} kg")
        print(f"  Transport: {r['transport_cost']:>12,.0f} VND")
        if r["late_penalty"] > 0:
            print(f"  Late     : {r['late_penalty']:>12,.0f} VND")
        print(f"  Congestion: {r['congestion_cost']:>12,.0f} VND")

    # --- Summary ---
    grand_total = total_transport + total_late + total_cong
    print(f"\n{'═' * 90}")
    print(f"  SUMMARY — K = {num_vehicles}")
    print(f"{'═' * 90}")
    print(f"  Active vehicles   : {len(routes_info)} / {num_vehicles}")
    print(f"  Transport cost    : {total_transport:>15,.0f} VND")
    print(f"  Late penalty (λ₁) : {total_late:>15,.0f} VND  ({data['late_penalty_per_min']:,} VND/min)")
    print(f"  Congestion (λ₂)   : {total_cong:>15,.0f} VND  ({data['congestion_penalty']:,} VND/unit)")
    print(f"  ────────────────────────────────────")
    print(f"  GRAND TOTAL       : {grand_total:>15,.0f} VND")
    print(f"  Solve time        : {solve_time:.1f}s")

    visited = set()
    for r in routes_info:
        for node in r["nodes"]:
            if node != 0:
                visited.add(node)
    all_customers = set(range(1, len(data["distance_matrix"])))
    dropped = all_customers - visited
    if dropped:
        dropped_names = [store_names[n] for n in sorted(dropped)]
        print(f"\n  Dropped stores ({len(dropped)}): {', '.join(dropped_names)}")
    else:
        print(f"\n  All stores visited!")

    print(f"{'═' * 90}")


def vehicle_sweep(data, k_min=3, k_max=7, time_limit_iter=5000):
    print("==" * 80)
    print("SOLVE ALNS + TABU SEARCH")
    print("==" * 80)

    results = []

    for k in range(k_min, k_max + 1):
        print("--" * 80)
        print(f"  Solving for K = {k} vehicles ...")
        print("--" * 80)

        status, total_cost, routes_info, solve_time = alns_tabu_solve(
            data, k, max_iterations=time_limit_iter, verbose=True,
        )

        if total_cost is not None and routes_info:
            print_solution(data, k, status, total_cost, routes_info, solve_time)

            total_transport = sum(r["transport_cost"] for r in routes_info)
            total_late = sum(r["late_penalty"] for r in routes_info)
            total_cong = sum(r["congestion_cost"] for r in routes_info)
            active = len(routes_info)

            results.append({
                "K": k,
                "status": status,
                "total_cost": total_cost,
                "transport": total_transport,
                "late_penalty": total_late,
                "congestion": total_cong,
                "active_vehicles": active,
                "solve_time": solve_time,
            })
        else:
            print(f"  K = {k}: No feasible solution found")
            results.append({
                "K": k,
                "status": "Infeasible",
                "total_cost": float("inf"),
                "transport": 0,
                "late_penalty": 0,
                "congestion": 0,
                "active_vehicles": 0,
                "solve_time": solve_time,
            })

    # ── Comparison table ──
    print("\n\n")
    print("==" * 80)
    print("VEHICLE SWEEP RESULTS — ALNS+TABU COMPARISON TABLE")
    print("==" * 80)

    header = (
        f"  {'K':>3}  {'Status':<10} {'Active':>6}  "
        f"{'Transport':>15}  {'Late Penalty':>15}  {'Congestion':>15}  "
        f"{'TOTAL COST':>15}  {'Time':>6}"
    )
    print(header)
    print("  " + "─" * 116)

    best_k = None
    best_cost = float("inf")

    for r in results:
        if r["total_cost"] < best_cost:
            best_cost = r["total_cost"]
            best_k = r["K"]

        marker = " ★" if r["total_cost"] == best_cost and r["total_cost"] < float("inf") else "  "
        if r["total_cost"] < float("inf"):
            print(
                f"  {r['K']:>3}  {r['status']:<10} {r['active_vehicles']:>6}  "
                f"{r['transport']:>15,.0f}  {r['late_penalty']:>15,.0f}  {r['congestion']:>15,.0f}  "
                f"{r['total_cost']:>15,.0f}  {r['solve_time']:>5.1f}s{marker}"
            )
        else:
            print(
                f"  {r['K']:>3}  {r['status']:<10} {'—':>6}  "
                f"{'—':>15}  {'—':>15}  {'—':>15}  "
                f"{'INFEASIBLE':>15}  {r['solve_time']:>5.1f}s  "
            )

    print("  " + "─" * 116)

    if best_k is not None:
        print(f"\n  🏆 BEST FOUND: K = {best_k}")
        best_r = [r for r in results if r["K"] == best_k][0]
        print(f"     Total cost: {best_cost:,.0f} VND")
        print(f"     Transport : {best_r['transport']:,.0f} VND")
        print(f"     Late pen. : {best_r['late_penalty']:,.0f} VND")
        print(f"     Congestion: {best_r['congestion']:,.0f} VND")
        print(f"     Active    : {best_r['active_vehicles']} vehicles")
    else:
        print("\n  No feasible solution found for any K!")

    print()
    return results, best_k


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    data = load_data()

    num_customers = len(data["distance_matrix"]) - 1
    print(f"Loaded {num_customers} stores + 1 depot")
    print(f"Capacity     : {data['vehicle_capacity']} kg")
    print(f"Cost/km      : {data['cost_per_km']:,} VND")
    print(f"Late penalty : {data['late_penalty_per_min']:,} VND/min  (λ₁)")
    print(f"Cong penalty : {data['congestion_penalty']:,} VND/unit (λ₂)")
    print(f"Service time : {data['service_time_min']} min")
    print(f"Avg speed    : {data['avg_speed_kmh']} km/h")
    print()

    results, best_k = vehicle_sweep(data, k_min=3, k_max=13, time_limit_iter=5000)
