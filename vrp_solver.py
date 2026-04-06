"""
Vehicle Routing Problem (VRP) - Capacitated with Time Windows
=============================================================
Objective: Minimizing total distance
Solver: Google OR-Tools (CP-SAT based routing solver)

Data:
  - store.csv: danh sách cửa hàng (STT, Tên, Demand, Time Window, Lat/Lng)
  - distant_matrix.csv: ma trận khoảng cách (km)

Parameters:
  - cost_per_km: 18,000 VND
  - number_vehicles: 13
  - capacity: 1,800 kg (1.8 tấn)
"""

import pandas as pd
import numpy as np
from ortools.constraint_solver import routing_enums_pb2, pywrapcp

# ==============================================================================
# 1. LOAD & PREPARE DATA
# ==============================================================================

def load_data():
    """Load store data and distance matrix, return a data dict for OR-Tools."""

    # --- Load store info ---
    df_store = pd.read_csv("store.csv")

    # Parse demand: "111,640" -> 111.640 (kg)  — comma is decimal separator
    df_store["Demand_kg"] = (
        df_store["Demand"]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )

    # Parse time windows → minutes from midnight
    def time_to_minutes(t_str):
        h, m = t_str.strip().split(":")
        return int(h) * 60 + int(m)

    df_store["tw_start"] = df_store["Time start"].apply(time_to_minutes)
    df_store["tw_end"]   = df_store["Time end"].apply(time_to_minutes)

    # --- Load distance matrix ---
    df_matrix = pd.read_csv("distant_matrix.csv")
    df_matrix = df_matrix.drop(columns=["From/to"])
    distance_matrix = df_matrix.values  # numpy 2-D array (21 x 21), unit: km

    # --- Build data model ---
    # Node 0 = depot (index 0 in the matrix)
    # Nodes 1..20 = stores (indices 1..20 in the matrix)

    num_locations = distance_matrix.shape[0]  # 21

    # Demands: depot has 0 demand, stores have their demand
    demands = [0] + df_store["Demand_kg"].tolist()

    # Time windows: depot gets a wide window (e.g., 6:00 – 18:00)
    depot_tw = (6 * 60, 18 * 60)  # 360 .. 1080 minutes
    time_windows = [depot_tw] + list(
        zip(df_store["tw_start"].tolist(), df_store["tw_end"].tolist())
    )

    # Store names for printing
    store_names = ["Depot"] + df_store["Tên CH"].tolist()

    # Coordinates for optional map plotting
    # Depot coordinate: we use the matrix row 0.  If you have a depot lat/lng,
    # replace here.  For now set depot to centroid of stores.
    depot_lat = df_store["Vĩ độ (Latitude)"].mean()
    depot_lng = df_store["Kinh độ (Longitude)"].mean()
    coords = [(depot_lat, depot_lng)] + list(
        zip(
            df_store["Vĩ độ (Latitude)"].tolist(),
            df_store["Kinh độ (Longitude)"].tolist(),
        )
    )

    data = {
        "distance_matrix": distance_matrix,
        "demands": demands,
        "time_windows": time_windows,
        "num_vehicles": 13,
        "depot": 0,
        "vehicle_capacity": 1800,  # kg
        "cost_per_km": 18_000,     # VND / km
        "store_names": store_names,
        "coords": coords,
        "df_store": df_store,
    }   
    return data


# ==============================================================================
# 2. OR-TOOLS MODEL
# ==============================================================================

def solve_vrp(data):
    """Build and solve the CVRPTW model. Returns manager, routing, solution."""

    num_nodes = len(data["distance_matrix"])
    num_vehicles = data["num_vehicles"]
    depot = data["depot"]

    # --- Routing Index Manager ---
    manager = pywrapcp.RoutingIndexManager(num_nodes, num_vehicles, depot)

    # --- Routing Model ---
    routing = pywrapcp.RoutingModel(manager)

    # ---- Distance callback ----
    # OR-Tools works with integers → multiply km by 100 to keep 2-decimal precision
    SCALE = 100

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node   = manager.IndexToNode(to_index)
        return int(data["distance_matrix"][from_node][to_node] * SCALE)

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # ---- Distance dimension (for objective & optional span cost) ----
    routing.AddDimension(
        transit_callback_index,
        0,             # no slack on distance
        300_000,       # max distance per vehicle (in scaled units = 3000 km)
        True,          # start cumul to zero
        "Distance",
    )
    distance_dimension = routing.GetDimensionOrDie("Distance")
    # Minimize the global span of distances (balances routes somewhat)
    distance_dimension.SetGlobalSpanCostCoefficient(100)

    # ---- Capacity (demand) constraint ----
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return int(data["demands"][from_node])

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,                                                       # no slack
        [int(data["vehicle_capacity"])] * num_vehicles,          # capacities
        True,                                                    # start cumul at 0
        "Capacity",
    )

    # ---- Time window constraint ----
    # Approximate travel time: assume average speed 30 km/h  → time = dist / 30 * 60 min
    AVG_SPEED_KMH = 30
    SERVICE_TIME_MIN = 10  # 10 min service at each stop

    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node   = manager.IndexToNode(to_index)
        travel_time_min = data["distance_matrix"][from_node][to_node] / AVG_SPEED_KMH * 60
        # add service time at the *from* node (except depot)
        service = SERVICE_TIME_MIN if from_node != depot else 0
        return int(travel_time_min + service)

    time_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.AddDimension(
        time_callback_index,
        60,            # max waiting time (slack) at each node: 60 min
        1440,          # max cumulative time per vehicle: 24h = 1440 min
        False,         # don't force start cumul to zero (vehicles can depart later)
        "Time",
    )
    time_dimension = routing.GetDimensionOrDie("Time")

    # Apply time windows
    for location_idx in range(num_nodes):
        if location_idx == depot:
            continue
        index = manager.NodeToIndex(location_idx)
        tw_start, tw_end = data["time_windows"][location_idx]
        time_dimension.CumulVar(index).SetRange(tw_start, tw_end)

    # Depot time window (vehicle must depart/return within)
    depot_tw_start, depot_tw_end = data["time_windows"][depot]
    for v in range(num_vehicles):
        start_index = routing.Start(v)
        end_index   = routing.End(v)
        time_dimension.CumulVar(start_index).SetRange(depot_tw_start, depot_tw_end)
        time_dimension.CumulVar(end_index).SetRange(depot_tw_start, depot_tw_end)
        # Minimise each vehicle's makespan
        routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(start_index))
        routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(end_index))

    # ---- Allow dropping visits if infeasible (penalty) ----
    # Large penalty discourages dropping but prevents infeasibility
    PENALTY = 100_000_000
    for node in range(1, num_nodes):
        routing.AddDisjunction([manager.NodeToIndex(node)], PENALTY)

    # ---- Search parameters ----
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.FromSeconds(30)  # max solving time
    search_parameters.log_search = True

    # ---- Solve ----
    solution = routing.SolveWithParameters(search_parameters)

    return manager, routing, solution


# ==============================================================================
# 3. EXTRACT & PRINT RESULTS
# ==============================================================================

def print_solution(data, manager, routing, solution):
    """Print route details for each vehicle + summary statistics."""

    SCALE = 100
    store_names = data["store_names"]
    cost_per_km = data["cost_per_km"]

    print("=" * 80)
    print("  VRP SOLUTION — MINIMIZE TOTAL DISTANCE (CVRPTW)")
    print("=" * 80)

    total_distance_scaled = 0
    total_load = 0
    total_cost = 0
    active_vehicles = 0
    routes = []

    time_dimension = routing.GetDimensionOrDie("Time")

    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        route_nodes = []
        route_distance_scaled = 0
        route_load = 0

        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            route_nodes.append(node)
            route_load += data["demands"][node]

            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance_scaled += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id
            )

        # Add depot at the end
        route_nodes.append(manager.IndexToNode(index))

        # Skip empty routes (depot → depot)
        if len(route_nodes) <= 2:
            continue

        active_vehicles += 1
        route_distance_km = route_distance_scaled / SCALE
        route_cost = route_distance_km * cost_per_km
        total_distance_scaled += route_distance_scaled
        total_load += route_load
        total_cost += route_cost

        routes.append({
            "vehicle": vehicle_id,
            "nodes": route_nodes,
            "distance_km": route_distance_km,
            "load_kg": route_load,
            "cost_vnd": route_cost,
        })

        # Print route
        print(f"\n🚛 Vehicle {vehicle_id + 1}")
        print(f"   Route: ", end="")
        route_str_parts = []
        for i, node in enumerate(route_nodes):
            time_var = None
            name = store_names[node]
            route_str_parts.append(name)
        print(" → ".join(route_str_parts))
        print(f"   Distance : {route_distance_km:>8.2f} km")
        print(f"   Load     : {route_load:>8.1f} kg  /  {data['vehicle_capacity']} kg")
        print(f"   Cost     : {route_cost:>12,.0f} VND")

    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    total_distance_km = total_distance_scaled / SCALE
    print(f"  Active vehicles  : {active_vehicles} / {data['num_vehicles']}")
    print(f"  Total distance   : {total_distance_km:,.2f} km")
    print(f"  Total load       : {total_load:,.1f} kg")
    print(f"  Total cost       : {total_cost:,.0f} VND")
    print(f"  Avg cost/vehicle : {total_cost / max(active_vehicles, 1):,.0f} VND")

    # Check dropped nodes
    dropped = []
    for node in range(1, len(data["distance_matrix"])):
        idx = manager.NodeToIndex(node)
        if solution.Value(routing.NextVar(idx)) == idx:
            dropped.append(store_names[node])
    if dropped:
        print(f"\n  ⚠️  Dropped stores ({len(dropped)}): {', '.join(dropped)}")
    else:
        print(f"\n  ✅ All stores visited!")

    print("=" * 80)
    return routes


# ==============================================================================
# 4. MAP VISUALIZATION (optional — requires folium)
# ==============================================================================

def plot_routes_on_map(data, routes, output_file="vrp_routes_map.html"):
    """Generate an interactive Folium map showing all vehicle routes."""
    try:
        import folium
        from folium import plugins
    except ImportError:
        print("\n[INFO] Install folium for map visualization: pip install folium")
        return

    coords = data["coords"]
    store_names = data["store_names"]
    depot_coord = coords[0]

    # Create base map centered on depot
    m = folium.Map(location=depot_coord, zoom_start=13, tiles="CartoDB positron")

    # Color palette for routes
    colors = [
        "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
        "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
        "#dcbeff", "#9A6324", "#800000",
    ]

    # Depot marker
    folium.Marker(
        depot_coord,
        popup=f"<b>DEPOT</b>",
        icon=folium.Icon(color="black", icon="home", prefix="fa"),
    ).add_to(m)

    for r_idx, route in enumerate(routes):
        color = colors[r_idx % len(colors)]
        route_coords = [coords[n] for n in route["nodes"]]

        # Route polyline
        folium.PolyLine(
            route_coords,
            color=color,
            weight=4,
            opacity=0.8,
            popup=f"Vehicle {route['vehicle'] + 1}: {route['distance_km']:.2f} km",
        ).add_to(m)

        # Store markers (skip depot nodes at start/end)
        for node in route["nodes"][1:-1]:
            folium.CircleMarker(
                coords[node],
                radius=7,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.9,
                popup=(
                    f"<b>{store_names[node]}</b><br>"
                    f"Demand: {data['demands'][node]:.1f} kg<br>"
                    f"Vehicle: {route['vehicle'] + 1}"
                ),
            ).add_to(m)

    m.save(output_file)
    print(f"\n🗺️  Route map saved to: {output_file}")


# ==============================================================================
# 5. MAIN
# ==============================================================================

if __name__ == "__main__":
    # Load data
    data = load_data()

    print(f"Loaded {len(data['distance_matrix']) - 1} stores + 1 depot")
    print(f"Vehicles: {data['num_vehicles']}, Capacity: {data['vehicle_capacity']} kg")
    print(f"Cost/km : {data['cost_per_km']:,} VND")
    print()

    # Solve
    manager, routing, solution = solve_vrp(data)

    if solution:
        routes = print_solution(data, manager, routing, solution)
        plot_routes_on_map(data, routes)
    else:
        print("❌ No solution found!")
        print("   Consider: increasing vehicles, capacity, or relaxing time windows.")
