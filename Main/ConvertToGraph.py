import osmnx as ox
import geopandas as gpd
import networkx as nx
import pandas as pd
import math

# Import the function from ObtainRD.py
from ObtainRD import generate_road_dataset


# -----------------------------
# 1. Define a Composite Score Function
# -----------------------------
def compute_composite_score(row,
                            w_venue=1.0, w_traffic=1.0,
                            w_temp=1.0, w_feels=1.0,
                            w_pressure=1.0, w_humidity=1.0,
                            w_wind_speed=1.0, w_wind_deg=1.0,
                            w_length=0.001):
    """
    Compute a composite score for a road segment from its metrics.
    Uses:
      - Total venue counts: shop_count + tourism_count + amenity_count + public_transport_count
      - traffic_score
      - Weather metrics: temperature, feels_like, pressure, humidity, wind_speed, wind_deg
      - Length as a penalty (weighted by w_length)
    Adjust the weights as needed.
    """
    venues = ((row.get("shop_count") or 0) +
              (row.get("tourism_count") or 0) +
              (row.get("amenity_count") or 0) +
              (row.get("public_transport_count") or 0))
    traffic = row.get("traffic_score") or 0
    temperature = row.get("temperature") or 0
    feels_like = row.get("feels_like") or 0
    pressure = row.get("pressure") or 0
    humidity = row.get("humidity") or 0
    wind_speed = row.get("wind_speed") or 0
    wind_deg = row.get("wind_deg") or 0
    length = row.get("length") or 0
    # Higher composite score means higher exposure, but longer segments incur a penalty.
    composite = (w_venue * venues +
                 w_traffic * traffic +
                 w_temp * temperature +
                 w_feels * feels_like +
                 w_pressure * pressure +
                 w_humidity * humidity +
                 w_wind_speed * wind_speed +
                 w_wind_deg * wind_deg -
                 w_length * length)
    return composite


# -----------------------------
# 2. Convert Road Dataset to Graph with Composite Scores (and preserve geometry)
# -----------------------------
def dataset_to_graph(road_dataset):
    """
    Convert a GeoDataFrame (road dataset) into a NetworkX DiGraph.
    Computes a composite score for each segment and sets the edge cost as the negative
    composite score (so that maximizing exposure becomes minimizing cost).

    Input:
      road_dataset: GeoDataFrame with columns:
         - "u", "v", "length", venue counts, "traffic_score", weather metrics, etc.

    Output:
      A directed NetworkX graph where each edge has attributes:
         - composite_score
         - cost (negative composite_score)
         - length
         - geometry
    """
    # Ensure "length" column exists
    if "length" not in road_dataset.columns:
        road_dataset["length"] = road_dataset.geometry.length

    # Compute composite score for each segment
    road_dataset["composite_score"] = road_dataset.apply(lambda row: compute_composite_score(row), axis=1)
    print("Composite scores computed. Sample:")
    print(road_dataset[["shop_count", "tourism_count", "amenity_count", "public_transport_count",
                        "traffic_score", "temperature", "composite_score"]].head())

    # Create the graph using "u" and "v" columns.
    # (Ensure that your ObtainRD.py returns a dataset with "u" and "v" columns.)
    G = nx.from_pandas_edgelist(
        road_dataset,
        source="u",
        target="v",
        edge_attr=["composite_score", "length", "geometry"],
        create_using=nx.DiGraph()
    )

    # Update each edge: keep composite_score, length, and geometry, and also compute "cost" as -composite_score.
    for u, v, data in G.edges(data=True):
        new_attrs = {
            "composite_score": data.get("composite_score", 0),
            "cost": -data.get("composite_score", 0),
            "length": data.get("length", 0),
            "geometry": data.get("geometry")  # preserve the road segment geometry
        }
        G[u][v].clear()
        G[u][v].update(new_attrs)

    return G


# -----------------------------
# 3. Main Execution
# -----------------------------
if __name__ == '__main__':
    # Get user input (same as in ObtainRD.py)
    start_lat = float(input("Enter start latitude: "))  # e.g., 48.8462
    start_lon = float(input("Enter start longitude: "))  # e.g., 2.3460
    finish_lat = float(input("Enter finish latitude: "))  # e.g., 48.8470
    finish_lon = float(input("Enter finish longitude: "))  # e.g., 2.3480

    # Generate the road dataset using the function from ObtainRD.py.
    # (Radius is set to 2000 meters; adjust as needed.)
    road_dataset = generate_road_dataset(start_lat, start_lon, finish_lat, finish_lon, radius=200)

    # Convert the dataset to a graph with weighted (composite) scores and geometry.
    G_route = dataset_to_graph(road_dataset)
    print("Graph created with", G_route.number_of_nodes(), "nodes and", G_route.number_of_edges(), "edges.")

    # Save the graph for later use.
    import pickle

    with open("road_network_graph.gpickle", "wb") as f:
        pickle.dump(G_route, f)
    print("Graph saved to road_network_graph.gpickle")
