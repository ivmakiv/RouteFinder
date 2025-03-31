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
                            w_venue=1.0, w_traffic=20.0,
                            w_temp=0.1, w_feels=0.1,
                            w_pressure=0.0, w_humidity=-0.05,
                            w_wind_speed=-0.5, w_wind_deg=0.0):
    """
    Compute a composite score for a road segment from its metrics.

    Uses:
      - Total venue counts: shop_count + tourism_count + amenity_count + public_transport_count
      - traffic_score
      - Weather metrics: temperature, feels_like, pressure, humidity, wind_speed, wind_deg
      - Length is included as a penalty (weighted by w_length)

    A higher composite score indicates higher advert exposure.
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
    composite = (w_venue * venues +
                 w_traffic * traffic +
                 w_temp * temperature +
                 w_feels * feels_like +
                 w_pressure * pressure +
                 w_humidity * humidity +
                 w_wind_speed * wind_speed +
                 w_wind_deg * wind_deg
                )
    return composite


# -----------------------------
# 2. Convert Road Dataset to Graph with Positive Costs (and preserve geometry)
# -----------------------------
def dataset_to_graph(road_dataset):
    """
    Convert a GeoDataFrame (road dataset) into a NetworkX DiGraph.

    Steps:
      1. Ensure a 'length' column exists (compute it from geometry if needed).
      2. Compute a composite score for each road segment.
      3. Compute the maximum composite score.
      4. For each segment, define a positive cost as:
             cost = max_composite - composite_score
      5. Build a directed graph using the "u" and "v" columns,
         and for each edge store only:
             - cost (the positive cost)
             - length
             - geometry

    The resulting graph does not retain the separate composite_score field.
    """
    # Ensure "length" column exists.
    if "length" not in road_dataset.columns:
        road_dataset["length"] = road_dataset.geometry.length

    # Compute composite score for each segment.
    road_dataset["composite_score"] = road_dataset.apply(lambda row: compute_composite_score(row), axis=1)
    print("Composite scores computed. Sample:")
    print(road_dataset[["shop_count", "tourism_count", "amenity_count", "public_transport_count",
                        "traffic_score", "temperature", "composite_score"]].head())

    # Compute maximum composite score.
    max_composite = road_dataset["composite_score"].max()
    print("Max composite score:", max_composite)

    # Compute a positive cost for each segment.
    road_dataset["cost"] = -road_dataset["composite_score"]

    # Create the graph using "u" and "v" columns.
    # (Make sure your dataset has "u" and "v" columns from the OSMnx extraction.)
    G = nx.from_pandas_edgelist(
        road_dataset,
        source="u",
        target="v",
        edge_attr=["cost", "length", "geometry"],
        create_using=nx.DiGraph()
    )

    # Remove the composite_score field from the graph's edges.
    for u, v, data in G.edges(data=True):
        # Data now contains: cost, length, geometry.
        # Optionally, you can leave the "cost" field as is.
        pass  # No action needed if you already have "cost" computed.

    return G


# -----------------------------
# 3. Main Execution (for testing)
# -----------------------------
if __name__ == '__main__':
    # Get user input (for testing purposes)
    start_lat = float(input("Enter start latitude: "))  # e.g., 48.8462
    start_lon = float(input("Enter start longitude: "))  # e.g., 2.3460
    finish_lat = float(input("Enter finish latitude: "))  # e.g., 48.8470
    finish_lon = float(input("Enter finish longitude: "))  # e.g., 2.3480

    # Generate the road dataset using the function from ObtainRD.py.
    road_dataset = generate_road_dataset(start_lat, start_lon, finish_lat, finish_lon, radius=2000)

    # Convert the dataset to a graph with positive cost values.
    G_route = dataset_to_graph(road_dataset)
    print("Graph created with", G_route.number_of_nodes(), "nodes and", G_route.number_of_edges(), "edges.")

    # Save the graph for later use.
    import pickle

    with open("road_network_graph.gpickle", "wb") as f:
        pickle.dump(G_route, f)
    print("Graph saved to road_network_graph.gpickle")
