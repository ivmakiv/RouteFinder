import math
import heapq
import networkx as nx
import matplotlib.pyplot as plt
from pyproj import Transformer


# -----------------------------
# Helper Functions
# -----------------------------
def assign_node_coordinates(G, road_dataset):
    """
    Ensure each node in graph G has 'x' and 'y' coordinates.
    For each row in road_dataset (which should contain "u", "v", and "geometry"),
    assign:
      - For node 'u': the first coordinate of the edge geometry.
      - For node 'v': the last coordinate of the edge geometry.
    """
    for idx, row in road_dataset.iterrows():
        u = row["u"]
        v = row["v"]
        geom = row["geometry"]
        if geom is not None:
            if u in G.nodes and ("x" not in G.nodes[u] or "y" not in G.nodes[u]):
                G.nodes[u]["x"], G.nodes[u]["y"] = geom.coords[0]
            if v in G.nodes and ("x" not in G.nodes[v] or "y" not in G.nodes[v]):
                G.nodes[v]["x"], G.nodes[v]["y"] = geom.coords[-1]
    return G


def nearest_node(G, x, y):
    """
    Find the nearest node in graph G to the point (x, y) using Euclidean distance.
    Assumes each node in G has "x" and "y" attributes.
    """
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
    """
    Compute the total route length (sum of edge lengths) for a given route (list of node IDs).
    """
    total = 0
    for i in range(len(route) - 1):
        edge_data = G.get_edge_data(route[i], route[i + 1])
        if edge_data:
            total += edge_data.get("length", 0)
    return total


# -----------------------------
# Constrained Route-Finding Function
# -----------------------------
def find_route_constrained(G, start_node, finish_node, max_length):
    """
    Find a path from start_node to finish_node in graph G that minimizes the total cost,
    subject to the constraint that the total length does not exceed max_length.

    This uses a modified Dijkstra-like algorithm with a length constraint.

    Each entry in the priority queue is a tuple:
       (total_cost, total_length, current_node, path)

    Returns:
      - best_path: list of node IDs if found; otherwise, None.
      - total_cost: total cost of the best path.
      - total_length: total length of the best path.
    """
    heap = [(0, 0, start_node, [start_node])]
    # For each node, keep the best (lowest) cost for a given accumulated length.
    visited = {}
    rez = None, None, None

    while heap:
        total_cost, total_length, current, path = heapq.heappop(heap)
        # If we reached the destination within the allowed length, return.
        if current == finish_node and total_length <= max_length and (rez[1] is None or total_cost < rez[1]):
            rez = path, total_cost, total_length

        # If we've seen this node with a better (lower cost, shorter length) combination, skip.
        if current in visited:
            best_length, best_cost = visited[current]
            if total_length >= best_length and total_cost >= best_cost:
                continue

        # Record the best known combination for this node.
        visited[current] = (total_length, total_cost)

        # Expand neighbors.
        for neighbor in G.successors(current):
            edge_data = G.get_edge_data(current, neighbor)
            if edge_data is None:
                continue
            edge_length = edge_data.get("length", 0)
            edge_cost = edge_data.get("cost", 0)
            new_length = total_length + edge_length
            if new_length <= max_length and neighbor not in path:
                new_cost = total_cost + edge_cost
                new_path = path + [neighbor]
                heapq.heappush(heap, (new_cost, new_length, neighbor, new_path))

    return rez


# -----------------------------
# High-Level Route-Finding Function
# -----------------------------
def find_route(G, road_dataset, start_lat, start_lon, finish_lat, finish_lon, max_length):
    """
    Given a graph G (constructed from your road dataset) and start/finish coordinates (in EPSG:4326),
    this function:
      1. Ensures nodes have coordinates.
      2. Transforms the input coordinates from EPSG:4326 to the graph's CRS (assumed EPSG:32631).
      3. Finds the nearest nodes.
      4. Uses find_route_constrained() to search for a path that does not exceed max_length.

    Returns:
      - best_route: List of node IDs representing the best route (or None if not found)
      - total_length: Total length of the best route (meters)
      - status: A status message.
    """
    # Ensure nodes have coordinates.
    G = assign_node_coordinates(G, road_dataset)
    if "crs" not in G.graph:
        G.graph["crs"] = "EPSG:32631"

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32631", always_xy=True)
    start_x, start_y = transformer.transform(start_lon, start_lat)
    finish_x, finish_y = transformer.transform(finish_lon, finish_lat)

    start_node = nearest_node(G, start_x, start_y)
    finish_node = nearest_node(G, finish_x, finish_y)

    if start_node is None or finish_node is None:
        return None, None, "Nearest nodes not found."

    best_route, total_cost, total_length = find_route_constrained(G, start_node, finish_node, max_length)
    if best_route is None:
        return None, None, "No path found within the maximum length constraint."

    status = "Success"
    return best_route, total_length, status


# -----------------------------
# Visualization Function
# -----------------------------
def visualize_route(G, best_route, start_lon, start_lat, finish_lon, finish_lat):
    """
    Visualize the road network and overlay the best route.

    Parameters:
      - G: NetworkX graph (in metric CRS, e.g., EPSG:32631) with nodes having 'x' and 'y' attributes.
      - best_route: List of node IDs representing the best route.
      - start_lon, start_lat: Start coordinates (EPSG:4326).
      - finish_lon, finish_lat: Finish coordinates (EPSG:4326).
    """
    from pyproj import Transformer
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32631", always_xy=True)
    start_x, start_y = transformer.transform(start_lon, start_lat)
    finish_x, finish_y = transformer.transform(finish_lon, finish_lat)

    fig, ax = plt.subplots(figsize=(12, 12))

    # Plot all edges in light gray.
    for u, v, data in G.edges(data=True):
        if "geometry" in data and data["geometry"] is not None:
            xs, ys = data["geometry"].xy
            ax.plot(xs, ys, color="lightgray", linewidth=1)
        else:
            x1, y1 = G.nodes[u].get("x"), G.nodes[u].get("y")
            x2, y2 = G.nodes[v].get("x"), G.nodes[v].get("y")
            if x1 is not None and x2 is not None:
                ax.plot([x1, x2], [y1, y2], color="lightgray", linewidth=1)

    # Overlay the best route in red.
    route_coords = [(G.nodes[node]["x"], G.nodes[node]["y"]) for node in best_route]
    xs, ys = zip(*route_coords)
    ax.plot(xs, ys, color="red", linewidth=3, label="Best Route")

    # Mark the start point in green and finish point in blue.
    ax.scatter([start_x], [start_y], color="green", s=100, label="Start")
    ax.scatter([finish_x], [finish_y], color="blue", s=100, label="Finish")

    ax.legend()
    ax.set_title("Best Route on Road Network")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    plt.show()


# -----------------------------
# Main Execution for Testing
# -----------------------------
if __name__ == '__main__':
    # For testing purposes, you can use sample coordinates in Paris.
    # Example: Start: 48.8600, 2.3522; Finish: 48.8550, 2.3700; max_length: 3000 m.
    start_lat = float(input("Enter start latitude (EPSG:4326): "))  # e.g., 48.8600
    start_lon = float(input("Enter start longitude (EPSG:4326): "))  # e.g., 2.3522
    finish_lat = float(input("Enter finish latitude (EPSG:4326): "))  # e.g., 48.8550
    finish_lon = float(input("Enter finish longitude (EPSG:4326): "))  # e.g., 2.3700
    max_length = float(input("Enter maximum route length in meters: "))  # e.g., 3000

    # For testing: assume you can generate the road dataset and graph from your other modules.
    from ObtainRD import generate_road_dataset
    from ConvertToGraph import dataset_to_graph

    road_dataset = generate_road_dataset(start_lat, start_lon, finish_lat, finish_lon, radius=500)
    G = dataset_to_graph(road_dataset)
    # Ensure nodes have coordinates.
    G = assign_node_coordinates(G, road_dataset)
    if "crs" not in G.graph:
        G.graph["crs"] = "EPSG:32631"

    best_route, total_length, status = find_route(G, road_dataset, start_lat, start_lon, finish_lat, finish_lon,
                                                  max_length)
    print(status)
    if best_route is None:
        exit()
    print("Best route (node sequence):", best_route)
    print("Total route length: {:.1f} m".format(total_length))
    visualize_route(G, best_route, start_lon, start_lat, finish_lon, finish_lat)
