import pandas as pd
import numpy as np
from pulp import (
    LpProblem, LpMinimize, LpVariable, LpBinary, LpContinuous,
    LpInteger, lpSum, PULP_CBC_CMD, LpStatus, value,
)
import time as time_mod


def generate_congestion_matrix(num_nodes, seed=42):
    #random dữ liệu kẹt xe
    #60% là đường bình thường
    #30% là đường hơi tắc
    #10% là đường đặc biệt tắc
    rng = np.random.default_rng(seed)
    congestion = rng.choice([1, 2, 3], size=(num_nodes, num_nodes), p=[0.6, 0.3, 0.1])
    np.fill_diagonal(congestion, 0)
    congestion = np.triu(congestion) + np.triu(congestion, 1).T
    return congestion


def load_data():
    #đọc data từ file csv
    df_store = pd.read_csv("data/store.csv")
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

    df_matrix = pd.read_csv("data/distant_matrix.csv")
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
        "vehicle_capacity": 1800,        # kg
        "cost_per_km": 18_000,           # VND/km
        "late_penalty_per_min": 50_000,  # λ₁  (VND/phút trễ)
        "congestion_penalty":   5_000,   # λ₂  (VND/congestion-unit)
        "service_time_min": 10,          # s_i — phút phục vụ/cửa hàng
        "avg_speed_kmh": 30,             # v_avg
        "store_names": store_names,
    }
    return data


CONGESTION_LABELS = {0: "—", 1: "Bình thường", 2: "Hơi tắc", 3: "Đặc biệt tắc"}


def build_and_solve_milp(data, num_vehicles, time_limit_sec=120, verbose=False):
    n = len(data["distance_matrix"])      # tổng số nodes (depot + customers)
    N = list(range(n))                     # node set {0, 1, ..., n-1}
    C = list(range(1, n))                  # customer set {1, ..., n-1}
    K = list(range(num_vehicles))          # vehicle set
    depot = data["depot"]

    dist   = data["distance_matrix"]
    cong   = data["congestion_matrix"]
    d      = data["demands"]
    Q      = data["vehicle_capacity"]
    tw     = data["time_windows"]
    cpk    = data["cost_per_km"]
    lam1   = data["late_penalty_per_min"]
    lam2   = data["congestion_penalty"]
    s_time = data["service_time_min"]
    v_avg  = data["avg_speed_kmh"]

    # tính thời gian di chuyển
    travel_time = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                travel_time[i][j] = (dist[i][j] / v_avg) * cong[i][j] * 60

    M = 24 * 60  # 1440 phút = 24 giờ

    #build model MILP
    #K là số xe
    prob = LpProblem(f"CVRPTW_MILP_K{num_vehicles}", LpMinimize)
    x = {}
    for i in N:
        for j in N:
            if i != j:
                for k in K:
                    x[i, j, k] = LpVariable(f"x_{i}_{j}_{k}", cat=LpBinary)

    S = {}
    for j in N:
        S[j] = LpVariable(f"S_{j}", lowBound=0, upBound=M, cat=LpContinuous)

    delta = {}
    for j in C:
        delta[j] = LpVariable(f"delta_{j}", lowBound=0, cat=LpContinuous)

    u = {}
    for i in C:
        u[i] = LpVariable(f"u_{i}", lowBound=1, upBound=n - 1, cat=LpInteger)

    # chi phí vận chuyển
    transport_cost = lpSum(
        cpk * dist[i][j] * x[i, j, k]
        for i in N for j in N if i != j for k in K
    )

    # chi phí trễ
    lateness_cost = lpSum(
        lam1 * delta[j] for j in C
    )

    #chi phí kẹt xe
    congestion_cost = lpSum(
        lam2 * cong[i][j] * x[i, j, k]
        for i in N for j in N if i != j for k in K
    )

    prob += transport_cost + lateness_cost + congestion_cost, "Total_Cost"

    # mỗi cửa hàng được phục vụ đúng 1 lần
    for j in C:
        prob += (
            lpSum(x[i, j, k] for i in N if i != j for k in K) == 1,
            f"serve_once_{j}",
        )

    # nếu xe k đi vào cửa hàng j thì xe k đi ra khỏi cửa hàng j
    for k in K:
        for j in C:
            prob += (
                lpSum(x[i, j, k] for i in N if i != j)
                == lpSum(x[j, i, k] for i in N if i != j),
                f"flow_{j}_{k}",
            )

    # mỗi xe rời khỏi kho tại most 1 lần
    for k in K:
        prob += (
            lpSum(x[depot, j, k] for j in C) <= 1,
            f"leave_depot_{k}",
        )

    # mỗi xe quay về kho tại most 1 lần
    for k in K:
        prob += (
            lpSum(x[j, depot, k] for j in C) <= 1,
            f"return_depot_{k}",
        )

    # rời khỏi kho == quay về kho
    for k in K:
        prob += (
            lpSum(x[depot, j, k] for j in C)
            == lpSum(x[j, depot, k] for j in C),
            f"depot_balance_{k}",
        )

    # tải trọng xe
    for k in K:
        prob += (
            lpSum(
                d[j] * lpSum(x[i, j, k] for i in N if i != j)
                for j in C
            ) <= Q,
            f"capacity_{k}",
        )

   
    for k in K:
        for i in N:
            for j in C:
                if i != j:
                    service = s_time if i != depot else 0
                    prob += (
                        S[j] >= S[i] + service + travel_time[i][j] - M * (1 - x[i, j, k]),
                        f"time_prop_{i}_{j}_{k}",
                    )

    # thời gian bắt đầu không sớm hơn thời gian bắt đầu của cửa hàng
    for j in C:
        prob += (
            S[j] >= tw[j][0],
            f"tw_lower_{j}",
        )

    # thời gian kết thúc không muộn hơn thời gian kết thúc của cửa hàng
    for j in C:
        prob += (
            delta[j] >= S[j] - tw[j][1],
            f"tw_upper_{j}",
        )

    # thời gian kho
    prob += S[depot] >= tw[depot][0], "depot_tw_lower"
    prob += S[depot] <= tw[depot][1], "depot_tw_upper"

    for k in K:
        for i in C:
            for j in C:
                if i != j:
                    prob += (
                        u[i] - u[j] + n * x[i, j, k] <= n - 1,
                        f"mtz_{i}_{j}_{k}",
                    )

    #tìm lời giải
    solver = PULP_CBC_CMD(
        timeLimit=time_limit_sec,
        msg=1 if verbose else 0,
        threads=4,
    )

    start_time = time_mod.time()
    prob.solve(solver)
    solve_time = time_mod.time() - start_time

    status = LpStatus[prob.status]

    if status not in ("Optimal", "Not Solved"):
        pass

    if prob.sol_status not in (1,): 
        test_val = value(x[depot, C[0], K[0]]) if (depot, C[0], K[0]) in x else None
        if test_val is None:
            return status, None, None, solve_time

   
    total_obj = value(prob.objective)

    routes_info = []
    for k in K:
        route_arcs = []
        for i in N:
            for j in N:
                if i != j and (i, j, k) in x:
                    val = value(x[i, j, k])
                    if val is not None and val > 0.5:
                        route_arcs.append((i, j))

        if not route_arcs:
            continue

        arc_dict = {a[0]: a[1] for a in route_arcs}
        if depot not in arc_dict:
            continue

        route = [depot]
        current = depot
        visited = set()
        while current in arc_dict and arc_dict[current] not in visited:
            nxt = arc_dict[current]
            route.append(nxt)
            visited.add(nxt)
            current = nxt
            if current == depot:
                break

        if len(route) <= 2:
            continue

        route_dist = sum(dist[route[i]][route[i+1]] for i in range(len(route)-1))
        route_load = sum(d[node] for node in route if node != depot)
        route_transport_cost = route_dist * cpk
        route_late_penalty = sum(
            lam1 * value(delta[j]) for j in route if j != depot and j in delta
        )
        route_cong_cost = sum(
            lam2 * cong[route[i]][route[i+1]]
            for i in range(len(route)-1)
        )

        timing = []
        for node in route:
            s_val = value(S[node])
            d_val = value(delta[node]) if node in delta else 0
            timing.append({
                "node": node,
                "service_start": s_val if s_val else 0,
                "lateness": d_val if d_val else 0,
            })

        routes_info.append({
            "vehicle": k,
            "nodes": route,
            "distance_km": route_dist,
            "load_kg": route_load,
            "transport_cost": route_transport_cost,
            "late_penalty": route_late_penalty,
            "congestion_cost": route_cong_cost,
            "total_cost": route_transport_cost + route_late_penalty + route_cong_cost,
            "timing": timing,
        })

    return status, total_obj, routes_info, solve_time

def print_solution(data, num_vehicles, status, total_cost, routes_info, solve_time):
    store_names = data["store_names"]
    tw = data["time_windows"]
    cong = data["congestion_matrix"]

    print(f"\n{'═' * 90}")
    print(f"  MILP SOLUTION — K = {num_vehicles} vehicles")
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
            print(f"Late   : {r['late_penalty']:>12,.0f} VND")
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
        print(f"\nDropped stores ({len(dropped)}): {', '.join(dropped_names)}")
    else:
        print(f"\nAll stores visited!")

    print(f"{'═' * 90}")


def vehicle_sweep(data, k_min=3, k_max=7, time_limit=120):
    print("=="*80)
    print("SOLVE MILP ALGORITHM")
    print("=="*80)

    results = []

    for k in range(k_min, k_max + 1):
        print("--"*80)
        print(f"  Solving for K = {k} vehicles ...")
        print("--"*80)

        status, total_cost, routes_info, solve_time = build_and_solve_milp(
            data, k, time_limit_sec=time_limit, verbose=False
        )

        if total_cost is not None:
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
            print(f"K = {k}: {status} — No feasible solution")
            results.append({
                "K": k,
                "status": status,
                "total_cost": float("inf"),
                "transport": 0,
                "late_penalty": 0,
                "congestion": 0,
                "active_vehicles": 0,
                "solve_time": solve_time,
            })

    print("\n\n")
    print("=="*80)
    print("VEHICLE SWEEP RESULTS — COMPARISON TABLE")
    print("=="*80)

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
        print(f"\n  🏆 OPTIMAL NUMBER OF VEHICLES: K = {best_k}")
        print(f"     Total cost: {best_cost:,.0f} VND")
        best_r = [r for r in results if r["K"] == best_k][0]
        print(f"     Transport : {best_r['transport']:,.0f} VND")
        print(f"     Late pen. : {best_r['late_penalty']:,.0f} VND")
        print(f"     Congestion: {best_r['congestion']:,.0f} VND")
        print(f"     Active    : {best_r['active_vehicles']} vehicles")
    else:
        print("\nNo feasible solution found for any K!")

    print()
    return results, best_k

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

    results, best_k = vehicle_sweep(data, k_min=3, k_max=13, time_limit=120)