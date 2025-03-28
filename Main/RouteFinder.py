import math
import pickle
import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt
from pyproj import Transformer

# Import functions from your modules:
from ObtainRD import generate_road_dataset
from ConvertToGraph import dataset_to_graph


def assign_node_coordinates(G, road_dataset):
    """
    Ensure each node in graph G has 'x' and 'y' coordinates.
    If missing, assign coordinates from the corresponding edge geometry.
    """
    for idx, row in road_dataset.iterrows():
        u = row["u"]
        v = row["v"]
        geom = row["geometry"]
        if geom is not None:
            if u not in G.nodes or "x" not in G.nodes[u]:
                G.nodes[u]["x"], G.nodes[u]["y"] = geom.coords[0]
            if v not in G.nodes or "x" not in G.nodes[v]:
                G.nodes[v]["x"], G.nodes[v]["y"] = geom.coords[-1]
    return G


def nearest_node(G, x, y):
    """Find the nearest node in graph G to the point (x, y) using Euclidean distance."""
    min_dist = float("inf")
    nearest = None
    for node, data in G.nodes(data=True):
        if "x" in data and "y" in data:
            d = math.hypot(x - data["x"], y - data["y"])
            if d < min_dist:
                min_dist = d
                nearest = node
    return nearest


def route_length(G, route):
    """Compute the total route length (sum of edge lengths) for a given route (list of node IDs)."""
    total = 0
    for i in range(len(route) - 1):
        edge_data = G.get_edge_data(route[i], route[i + 1])
        if edge_data:
            total += edge_data.get("length", 0)
    return total


def main():
    # -----------------------------
    # 1. Get User Input for Routing
    # -----------------------------
    print("Input for dataset and routing:")
    start_lat = float(input("Enter start latitude (EPSG:4326): "))  # e.g., 48.8462
    start_lon = float(input("Enter start longitude (EPSG:4326): "))  # e.g., 2.3460
    finish_lat = float(input("Enter finish latitude (EPSG:4326): "))  # e.g., 48.8470
    finish_lon = float(input("Enter finish longitude (EPSG:4326): "))  # e.g., 2.3480
    max_length = float(input("Enter maximum route length in meters: "))  # e.g., 3000

    # -----------------------------
    # 2. Generate the Road Dataset and Convert to Graph
    # -----------------------------
    # Call the function from ObtainRD.py to generate the dataset.
    road_dataset = generate_road_dataset(start_lat, start_lon, finish_lat, finish_lon, radius=400)
    print("Road dataset generated.")

    # Convert dataset to graph using the function from ConvertToGraph.py.
    G_route = dataset_to_graph(road_dataset)
    print("Graph created with", G_route.number_of_nodes(), "nodes and", G_route.number_of_edges(), "edges.")

    # Ensure node coordinates are assigned.
    G_route = assign_node_coordinates(G_route, road_dataset)

    # Set the graph CRS if missing.
    if "crs" not in G_route.graph:
        G_route.graph["crs"] = "EPSG:32631"

    # -----------------------------
    # 3. Shift Cost Values to Avoid Negative Weights
    # -----------------------------
    # Instead of cost = -composite_score, we compute:
    # cost = max_composite - composite_score, ensuring all costs are nonnegative.
    # First, compute max composite score from the dataset.
    max_composite = max(data.get("composite_score", 0) for u, v, data in G_route.edges(data=True))
    for u, v, data in G_route.edges(data=True):
        composite = data.get("composite_score", 0)
        # Set cost so that higher composite gives lower cost.
        data["cost"] = max_composite - composite

    # -----------------------------
    # 4. Convert User Input Coordinates to Metric CRS (EPSG:32631)
    # -----------------------------
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32631", always_xy=True)
    start_x, start_y = transformer.transform(start_lon, start_lat)
    finish_x, finish_y = transformer.transform(finish_lon, finish_lat)

    # -----------------------------
    # 5. Find Nearest Nodes in the Graph
    # -----------------------------
    start_node = nearest_node(G_route, start_x, start_y)
    finish_node = nearest_node(G_route, finish_x, finish_y)
    print("Start node:", start_node, "Finish node:", finish_node)

    # -----------------------------
    # 6. Compute Best Route Using Bellman-Ford (supports negative weights, but we shifted costs)
    # -----------------------------
    try:
        best_route = nx.dijkstra_path(G_route, start_node, finish_node, weight="cost")
    except nx.NetworkXNoPath:
        print("No path found between the specified nodes.")
        return

    total_route_length = route_length(G_route, best_route)
    if total_route_length > max_length:
        print(f"Route length {total_route_length:.1f} m exceeds maximum allowed {max_length} m.")
    else:
        print("Best route (node sequence):", best_route)
        print("Total route length: {:.1f} m".format(total_route_length))

    # -----------------------------
    # 7. Visualize the Road Network and the Best Route
    # -----------------------------
    fig, ax = plt.subplots(figsize=(12, 12))
    # Plot all edges in light gray using edge geometries
    for u, v, data in G_route.edges(data=True):
        if "geometry" in data and data["geometry"] is not None:
            xs, ys = data["geometry"].xy
            ax.plot(xs, ys, color="lightgray", linewidth=1)
        else:
            x1, y1 = G_route.nodes[u].get("x"), G_route.nodes[u].get("y")
            x2, y2 = G_route.nodes[v].get("x"), G_route.nodes[v].get("y")
            if x1 is not None and x2 is not None:
                ax.plot([x1, x2], [y1, y2], color="lightgray", linewidth=1)

    # Extract and plot the best route in red.
    route_coords = [(G_route.nodes[node]["x"], G_route.nodes[node]["y"]) for node in best_route]
    xs, ys = zip(*route_coords)
    ax.plot(xs, ys, color="red", linewidth=3, label="Best Route")

    ax.legend()
    ax.set_title("Best Route on Road Network")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    plt.show()


if __name__ == '__main__':
    main()
