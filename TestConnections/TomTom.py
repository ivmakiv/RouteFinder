import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import requests, json
from shapely.geometry import Point

# Load configuration from config.json
with open("../config.json") as f:
    config = json.load(f)

# -----------------------------
# 1. Define the Area of Interest
# -----------------------------
place_name = "5th arrondissement, Paris, France"

# -----------------------------
# 2. Download Road Network and POIs
# -----------------------------
G = ox.graph_from_place(place_name, network_type="drive")
print(f"Graph has {len(G.nodes)} nodes and {len(G.edges)} edges")
gdf_edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
# Reproject road network to metric CRS (UTM zone 31N)
gdf_edges = gdf_edges.to_crs(epsg=32631)
# Buffer each road segment by 200 m (for our venue count analysis)
buffer_distance = 200  # meters
gdf_edges["buffer"] = gdf_edges.geometry.buffer(buffer_distance)

# Retrieve POIs from OSM using a broad tag selection
tags = {'amenity': True, 'shop': True, 'tourism': True}
gdf_pois = ox.features_from_place(place_name, tags)
# Reproject POIs to metric CRS
gdf_pois = gdf_pois.to_crs(epsg=32631)

# -----------------------------
# 3. (We already computed venue counts in previous code; now we integrate traffic)
# -----------------------------

def fetch_tomtom_traffic_data(point, zoom, api_key):
    """
    Fetches traffic flow data from the TomTom Traffic API.
    Parameters:
      point (str): "lat,lon" in degrees.
      zoom (int): Zoom level (e.g., 10).
      api_key (str): TomTom API key.
    Returns:
      dict: JSON response from the API.
    """
    url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/{zoom}/json"
    params = {
        "point": point,
        "key": api_key
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

# -----------------------------
# 4. Integrate Traffic Data for a Road Segment
# -----------------------------
# For demonstration, we'll take the first road segment.
segment0 = gdf_edges.iloc[4]

# Compute the centroid of the road segment (in metric CRS)
centroid_metric = segment0.geometry.centroid
# Reproject centroid to geographic CRS (EPSG:4326) to use in TomTom API
centroid_geo = gpd.GeoSeries([centroid_metric], crs="EPSG:32631").to_crs(epsg=4326).iloc[0]
# Format as "lat,lon"
point_str = f"{centroid_geo.y},{centroid_geo.x}"
print("Segment 0 centroid (lat,lon):", point_str)

# Use TomTom API to fetch traffic data for this point.
tomtom_api_key = config["tomtom"]["api_key"]
zoom = 12  # choose a zoom level appropriate for your needs
traffic_data = fetch_tomtom_traffic_data(point_str, zoom, tomtom_api_key)
print("TomTom traffic data for segment 0:")
print(json.dumps(traffic_data, indent=2))

# -----------------------------
# 5. Compute a Traffic Congestion Metric
# -----------------------------
# We'll define a simple congestion metric: freeFlowSpeed - currentSpeed.
flow_data = traffic_data.get("flowSegmentData", {})
current_speed = flow_data.get("currentSpeed")
free_flow_speed = flow_data.get("freeFlowSpeed")
if current_speed is not None and free_flow_speed is not None:
    congestion_metric = free_flow_speed - current_speed
else:
    congestion_metric = None

print("Congestion metric for segment 0 (freeFlowSpeed - currentSpeed):", congestion_metric)

from shapely.geometry import Point, LineString

# Store the congestion metric in the GeoDataFrame for segment 0.
gdf_edges.loc[gdf_edges.index[4], "congestion_metric"] = congestion_metric

# Extract coordinates and create a LineString representing the traffic segment.
coords_list = traffic_data["flowSegmentData"]["coordinates"]["coordinate"]
line_coords = [(coord["longitude"], coord["latitude"]) for coord in coords_list]
segment_line = LineString(line_coords)
# Create a GeoDataFrame for this TomTom traffic segment (initially in EPSG:4326)
gdf_segment = gpd.GeoDataFrame([{"geometry": segment_line}], crs="EPSG:4326")
# Reproject the traffic segment to the same metric CRS as the road network.
gdf_segment = gdf_segment.to_crs(epsg=32631)

# -----------------------------
# 5. Compute the Centroid of the First Road Segment
# -----------------------------
segment0 = gdf_edges.iloc[4]
centroid_metric = segment0.geometry.centroid

# -----------------------------
# 6. Visualize Everything on a Single Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(12, 12))

# Plot the full road network in black (thin lines)
gdf_edges.plot(ax=ax, color="black", linewidth=0.5, alpha=0.5)
# Plot the road segment buffers in light blue with blue edges
gdf_edges["buffer"].plot(ax=ax, color="lightblue", edgecolor="blue", alpha=0.3)
# Plot the POIs in red
gdf_pois.plot(ax=ax, color="red", markersize=5, alpha=0.8)
# Highlight the first road segment in green
gpd.GeoSeries(segment0.geometry).plot(ax=ax, color="green", linewidth=3)
# Plot the centroid of segment 0 as a large blue dot
ax.scatter([centroid_metric.x], [centroid_metric.y], color="blue", s=100)
# Plot the TomTom traffic segment (from the API) in magenta
gdf_segment.plot(ax=ax, edgecolor="magenta", linewidth=3)
plt.title("5th Arrondissement: Road Network, Buffers, POIs, and TomTom Traffic Segment")
plt.xlabel("Easting (m)")
plt.ylabel("Northing (m)")
plt.show()

