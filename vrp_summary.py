"""
Sweep number of vehicles from 1 to 13 and print SUMMARY for each.
"""
import pandas as pd
import numpy as np
from ortools.constraint_solver import routing_enums_pb2, pywrapcp
from vrp_solver import load_data

SCALE = 100


def solve_for_n_vehicles(data, n_vehicles):
    """Solve VRP with n_vehicles, return summary dict or None."""
    num_nodes = len(data["distance_matrix"])
    depot = data["depot"]

    manager = pywrapcp.RoutingIndexManager(num_nodes, n_vehicles, depot)
    routing = pywrapcp.RoutingModel(manager)

    # Distance callback
    def distance_callback(from_index, to_index):
        f = manager.IndexToNode(from_index)
        t = manager.IndexToNode(to_index)
        return int(data["distance_matrix"][f][t] * SCALE)

    transit_cb = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb)

    routing.AddDimension(transit_cb, 0, 300_000, True, "Distance")
    dist_dim = routing.GetDimensionOrDie("Distance")
    dist_dim.SetGlobalSpanCostCoefficient(100)

    # Capacity
    def demand_callback(from_index):
        return int(data["demands"][manager.IndexToNode(from_index)])

    demand_cb = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_cb, 0,
        [int(data["vehicle_capacity"])] * n_vehicles,
        True, "Capacity",
    )

    # Time windows
    AVG_SPEED = 30
    SERVICE = 10

    def time_callback(from_index, to_index):
        f = manager.IndexToNode(from_index)
        t = manager.IndexToNode(to_index)
        travel = data["distance_matrix"][f][t] / AVG_SPEED * 60
        svc = SERVICE if f != depot else 0
        return int(travel + svc)

    time_cb = routing.RegisterTransitCallback(time_callback)
    routing.AddDimension(time_cb, 60, 1440, False, "Time")
    time_dim = routing.GetDimensionOrDie("Time")

    for loc in range(num_nodes):
        if loc == depot:
            continue
        idx = manager.NodeToIndex(loc)
        tw_s, tw_e = data["time_windows"][loc]
        time_dim.CumulVar(idx).SetRange(tw_s, tw_e)

    depot_tw_s, depot_tw_e = data["time_windows"][depot]
    for v in range(n_vehicles):
        si = routing.Start(v)
        ei = routing.End(v)
        time_dim.CumulVar(si).SetRange(depot_tw_s, depot_tw_e)
        time_dim.CumulVar(ei).SetRange(depot_tw_s, depot_tw_e)
        routing.AddVariableMinimizedByFinalizer(time_dim.CumulVar(si))
        routing.AddVariableMinimizedByFinalizer(time_dim.CumulVar(ei))

    # Disjunction
    PENALTY = 100_000_000
    for node in range(1, num_nodes):
        routing.AddDisjunction([manager.NodeToIndex(node)], PENALTY)

    # Search
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    params.time_limit.FromSeconds(15)
    params.log_search = False  # quiet

    solution = routing.SolveWithParameters(params)

    if not solution:
        return None

    # Extract summary
    total_dist_scaled = 0
    total_load = 0
    active = 0
    dropped = []

    for v in range(n_vehicles):
        index = routing.Start(v)
        route_nodes = []
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            route_nodes.append(node)
            prev = index
            index = solution.Value(routing.NextVar(index))
            total_dist_scaled += routing.GetArcCostForVehicle(prev, index, v)
        route_nodes.append(manager.IndexToNode(index))
        if len(route_nodes) > 2:
            active += 1
            for n in route_nodes:
                total_load += data["demands"][n]

    for node in range(1, num_nodes):
        idx = manager.NodeToIndex(node)
        if solution.Value(routing.NextVar(idx)) == idx:
            dropped.append(node)

    total_km = total_dist_scaled / SCALE
    total_cost = total_km * data["cost_per_km"]

    return {
        "n_vehicles": n_vehicles,
        "active_vehicles": active,
        "total_distance_km": round(total_km, 2),
        "total_load_kg": round(total_load, 1),
        "total_cost_vnd": int(total_cost),
        "avg_cost_per_vehicle": int(total_cost / max(active, 1)),
        "dropped_stores": len(dropped),
        "all_served": len(dropped) == 0,
    }


if __name__ == "__main__":
    data = load_data()

    print("=" * 100)
    print("  VRP SUMMARY — SWEEP NUMBER OF VEHICLES (1 → 13)")
    print("=" * 100)
    print(
        f"{'Vehicles':>10} | {'Active':>8} | {'Distance (km)':>15} | "
        f"{'Load (kg)':>12} | {'Total Cost (VND)':>18} | "
        f"{'Avg Cost/Vehicle':>18} | {'Dropped':>8} | {'Status':>12}"
    )
    print("-" * 100)

    results = []
    for n in range(1, 14):
        summary = solve_for_n_vehicles(data, n)
        if summary:
            results.append(summary)
            status = "✅ All served" if summary["all_served"] else f"⚠️ Drop {summary['dropped_stores']}"
            print(
                f"{summary['n_vehicles']:>10} | "
                f"{summary['active_vehicles']:>8} | "
                f"{summary['total_distance_km']:>15,.2f} | "
                f"{summary['total_load_kg']:>12,.1f} | "
                f"{summary['total_cost_vnd']:>18,} | "
                f"{summary['avg_cost_per_vehicle']:>18,} | "
                f"{summary['dropped_stores']:>8} | "
                f"{status:>12}"
            )
        else:
            print(f"{n:>10} | {'—':>8} | {'NO SOLUTION':>15} | {'—':>12} | {'—':>18} | {'—':>18} | {'—':>8} | ❌ Infeasible")

    print("=" * 100)

    # Find optimal
    feasible = [r for r in results if r["all_served"]]
    if feasible:
        best = min(feasible, key=lambda x: x["total_distance_km"])
        print(f"\n🏆 Optimal: {best['n_vehicles']} vehicles → {best['total_distance_km']:.2f} km → {best['total_cost_vnd']:,} VND")
