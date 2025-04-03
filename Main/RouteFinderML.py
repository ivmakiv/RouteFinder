import math
import heapq
import joblib
import networkx as nx
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from pyproj import Transformer

# ----------------------------------------------------
# Helper Functions: Assign coordinates and find nodes
# ----------------------------------------------------

def assign_node_coordinates(G, road_dataset):
    """
    Assign coordinates to nodes in the graph G based on the geometry in the road_dataset.
    This is required for distance calculations and visualization.
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
    Find the nearest node in the graph G to a given coordinate (x, y).
    Uses Euclidean distance.
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

# ----------------------------------------------------
# Data Generation for ML Training (Supervised Learning)
# ----------------------------------------------------

def generate_training_data(G, start_node, finish_node, max_length, find_route_constrained):
    """
    Generate training samples from the path found by the classical algorithm.
    Each sample corresponds to a possible move from a node to one of its neighbors,
    with features and labels for supervised learning.
    """
    X, y = [], []
    result = find_route_constrained(G, start_node, finish_node, max_length)
    if result[0] is None:
        return X, y  # No valid path

    path, total_cost, total_length = result
    for i in range(len(path) - 1):
        current = path[i]
        correct_next = path[i + 1]
        for neighbor in G.successors(current):
            edge_data = G.get_edge_data(current, neighbor)
            if edge_data is None:
                continue
            edge_cost = edge_data.get("cost", 0)
            edge_length = edge_data.get("length", 0)
            neighbor_data = G.nodes[neighbor]
            finish_data = G.nodes[finish_node]
            euclidean_to_finish = math.hypot(
                neighbor_data["x"] - finish_data["x"],
                neighbor_data["y"] - finish_data["y"]
            )
            features = [edge_cost, edge_length, euclidean_to_finish]
            label = 1 if neighbor == correct_next else 0
            X.append(features)
            y.append(label)
    return X, y


# ----------------------------------------------------
# Training the ML Classifier
# ----------------------------------------------------

def train_model(G, route_samples, find_route_constrained):
    """
    Train a Decision Tree classifier using simulated route samples.
    """
    X_all, y_all = [], []
    for start, finish, max_len in route_samples:
        X, y = generate_training_data(G, start, finish, max_len, find_route_constrained)
        X_all.extend(X)
        y_all.extend(y)

    if not X_all:
        print("No training data generated.")
        return None

    # Split and train a simple Decision Tree
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
    clf = DecisionTreeClassifier(max_depth=5)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))

    joblib.dump(clf, "pathfinder_model.pkl")
    return clf


# ----------------------------------------------------
# ML-Based Pathfinding Inference
# ----------------------------------------------------

def find_route_ml(G, start_node, finish_node, max_length, model):
    """
    Use a trained ML model to predict the best path from start_node to finish_node.
    At each step, the model selects the next node based on features.
    """
    current = start_node
    path = [current]
    total_length = 0
    total_cost = 0

    while current != finish_node:
        candidates = []
        for neighbor in G.successors(current):
            edge_data = G.get_edge_data(current, neighbor)
            if edge_data is None:
                continue

            edge_cost = edge_data.get("cost", 0)
            edge_length = edge_data.get("length", 0)
            neighbor_data = G.nodes[neighbor]
            finish_data = G.nodes[finish_node]

            euclidean_to_finish = math.hypot(
                neighbor_data["x"] - finish_data["x"],
                neighbor_data["y"] - finish_data["y"]
            )
            features = [edge_cost, edge_length, euclidean_to_finish]
            candidates.append((model.predict([features])[0], neighbor, edge_cost, edge_length))

        # Filter predicted good moves and pick the best
        valid_moves = [(n, c, l) for pred, n, c, l in candidates if pred == 1 and n not in path]
        if not valid_moves:
            break

        next_node, cost, length = sorted(valid_moves, key=lambda x: x[1])[0]

        if total_length + length > max_length:
            break

        total_length += length
        total_cost += cost
        path.append(next_node)
        current = next_node

    if current == finish_node:
        return path, total_cost, total_length
    return None, None, None


# ----------------------------------------------------
# Main Interface Function
# ----------------------------------------------------

def find_route(G, road_dataset, start_lat, start_lon, finish_lat, finish_lon, max_length):
    """
    High-level route-finding interface. Transforms coordinates, finds nearest nodes,
    loads the ML model, and performs ML-based routing.
    """
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

    # Load trained ML model
    model = joblib.load("pathfinder_model.pkl")
    best_route, total_cost, total_length = find_route_ml(G, start_node, finish_node, max_length, model)

    if best_route is None:
        return None, None, "No path found within the maximum length constraint."

    return best_route, total_length, "Success"


# ----------------------------------------------------
# Visualization
# ----------------------------------------------------

def visualize_route(G, best_route, start_lon, start_lat, finish_lon, finish_lat):
    """
    Visualize the graph and overlay the predicted best route in red.
    """
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32631", always_xy=True)
    start_x, start_y = transformer.transform(start_lon, start_lat)
    finish_x, finish_y = transformer.transform(finish_lon, finish_lat)

    fig, ax = plt.subplots(figsize=(12, 12))
    for u, v, data in G.edges(data=True):
        if "geometry" in data and data["geometry"] is not None:
            xs, ys = data["geometry"].xy
            ax.plot(xs, ys, color="lightgray", linewidth=1)
        else:
            x1, y1 = G.nodes[u].get("x"), G.nodes[u].get("y")
            x2, y2 = G.nodes[v].get("x"), G.nodes[v].get("y")
            if x1 is not None and x2 is not None:
                ax.plot([x1, x2], [y1, y2], color="lightgray", linewidth=1)

    route_coords = [(G.nodes[node]["x"], G.nodes[node]["y"]) for node in best_route]
    xs, ys = zip(*route_coords)
    ax.plot(xs, ys, color="red", linewidth=3, label="Best Route")
    ax.scatter([start_x], [start_y], color="green", s=100, label="Start")
    ax.scatter([finish_x], [finish_y], color="blue", s=100, label="Finish")
    ax.legend()
    ax.set_title("Best Route on Road Network")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    plt.show()


# ----------------------------------------------------
# Main Execution Block
# ----------------------------------------------------
if __name__ == '__main__':
    # Step 1: Get user input
    start_lat = float(input("Enter start latitude (EPSG:4326): "))
    start_lon = float(input("Enter start longitude (EPSG:4326): "))
    finish_lat = float(input("Enter finish latitude (EPSG:4326): "))
    finish_lon = float(input("Enter finish longitude (EPSG:4326): "))
    max_length = float(input("Enter maximum route length in meters: "))

    # Step 2: Import road data and convert to graph
    from ObtainRD import generate_road_dataset
    from ConvertToGraph import dataset_to_graph
    from RouteFinder import find_route_constrained  # classic algorithm for training

    road_dataset = generate_road_dataset(start_lat, start_lon, finish_lat, finish_lon, radius=500)
    G = dataset_to_graph(road_dataset)
    G = assign_node_coordinates(G, road_dataset)

    # Optional: Train the model with your own sample routes
    # sample_routes = [(start_node, finish_node, max_length), ...]
    # train_model(G, sample_routes, find_route_constrained)

    # Step 3: Find and display route using ML model
    best_route, total_length, status = find_route(G, road_dataset, start_lat, start_lon, finish_lat, finish_lon, max_length)
    print(status)
    if best_route is None:
        exit()
    print("Best route (node sequence):", best_route)
    print("Total route length: {:.1f} m".format(total_length))
    visualize_route(G, best_route, start_lon, start_lat, finish_lon, finish_lat)
