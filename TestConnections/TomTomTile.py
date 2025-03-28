import os
import math
import requests
import json
import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import Point, LineString
import mapbox_vector_tile

# -----------------------------
# 1. Load Configuration and Define Area
# -----------------------------
with open("../config.json") as f:
    config = json.load(f)

place_name = "Paris, France"
tomtom_api_key = config["tomtom"]["api_key"]

# -----------------------------
# 2. Download Road Network
# -----------------------------
G = ox.graph_from_place(place_name, network_type="drive")
print(f"Graph has {len(G.nodes)} nodes and {len(G.edges)} edges")
gdf_edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
print("First few road segments:")
print(gdf_edges.head())

# -----------------------------
# 3. Reproject Road Network to Metric CRS (UTM Zone 31N)
# -----------------------------
gdf_edges = gdf_edges.to_crs(epsg=32631)


# -----------------------------
# 4. Define Functions for TomTom Tile Requests and Tile Traffic Score Computation
# -----------------------------
def latlon_to_tile(lat, lon, zoom):
    """
    Converts latitude and longitude (EPSG:4326) to tile x,y indices at the given zoom level.
    """
    lat_rad = math.radians(lat)
    n = 2 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile


def fetch_tomtom_tile_pbf(zoom, x, y, api_key, flow_type="relative"):
    """
    Fetches a TomTom traffic vector tile in PBF format.
    flow_type: "relative" (for relative traffic levels) or "absolute"
    """
    url = f"https://api.tomtom.com/traffic/map/4/tile/flow/{flow_type}/{zoom}/{x}/{y}.pbf"
    params = {"key": api_key}
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.content


def compute_tile_traffic_score_for_segment(segment_geometry, zoom, api_key, flow_type="relative"):
    """
    Computes a traffic score for a road segment using TomTom's vector tile API.
    The function:
      1. Computes the segment's centroid (in metric CRS).
      2. Converts the centroid to geographic coordinates (EPSG:4326).
      3. Converts that lat/lon into tile x,y indices at the given zoom level.
      4. Requests the traffic tile in PBF format.
      5. Decodes the tile and extracts all features from the "Traffic flow" layer.
      6. Computes the average "traffic_level" value among all features and returns it.
         (For relative tiles, traffic_level is a fractional value between 0 and 1.)
    """
    # 1. Compute centroid in metric CRS.
    centroid_metric = segment_geometry.centroid
    # 2. Convert centroid to geographic CRS (EPSG:4326)
    centroid_geo = gpd.GeoSeries([centroid_metric], crs="EPSG:32631").to_crs(epsg=4326).iloc[0]
    lat, lon = centroid_geo.y, centroid_geo.x
    # 3. Convert lat/lon to tile coordinates.
    tile_x, tile_y = latlon_to_tile(lat, lon, zoom)
    # 4. Fetch the tile in PBF format.
    try:
        pbf_data = fetch_tomtom_tile_pbf(zoom, tile_x, tile_y, api_key, flow_type=flow_type)
    except Exception as e:
        print(f"Tile fetch error at tile ({tile_x}, {tile_y}): {e}")
        return 0
    # 5. Decode the tile.
    tile_decoded = mapbox_vector_tile.decode(pbf_data)
    # The layer containing traffic data is typically named "Traffic flow".
    traffic_layer = tile_decoded.get("Traffic flow")
    if traffic_layer is None:
        return 0
    features = traffic_layer.get("features", [])
    if not features:
        return 0
    # 6. Extract the "traffic_level" value from each feature.
    levels = []
    for feat in features:
        props = feat.get("properties", {})
        level = props.get("traffic_level")
        if level is not None:
            levels.append(level)
    if levels:
        avg_level = sum(levels) / len(levels)
        return avg_level
    return 0


# -----------------------------
# 5. Compute Traffic Score for Each Road Segment
# -----------------------------
zoom_level = 12  # You can adjust this zoom level as needed.
traffic_scores = []
for idx, row in gdf_edges.iterrows():
    print(idx)
    try:
        score = compute_tile_traffic_score_for_segment(row.geometry, zoom_level, tomtom_api_key, flow_type="relative")
    except Exception as e:
        print(f"Error for segment {idx}: {e}")
        score = 0
    traffic_scores.append(score)

print(traffic_scores)

gdf_edges["traffic_score"] = traffic_scores
print("First few road segments with traffic scores:")
print(gdf_edges[["traffic_score"]].head())

# -----------------------------
# 6. Visualize the Road Network Colored by Traffic Score
# -----------------------------
fig, ax = plt.subplots(figsize=(12, 12))
# Plot road segments colored by traffic score using a colormap (e.g., Reds)
gdf_edges.plot(ax=ax, column="traffic_score", cmap="Reds", linewidth=1, legend=True)
plt.title(
    "5th Arrondissement: Road Network Colored by Traffic Score\n(traffic_score = avg relative traffic_level from TomTom)")
plt.xlabel("Easting (m)")
plt.ylabel("Northing (m)")
plt.show()
