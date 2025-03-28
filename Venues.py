import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import requests, json
from shapely.geometry import Point

# Load configuration from config.json
with open("config.json") as f:
    config = json.load(f)

# -----------------------------
# 1. Define the Area of Interest
# -----------------------------
place_name = "5th arrondissement, Paris, France"

# -----------------------------
# 2. Download Road Network
# -----------------------------
G = ox.graph_from_place(place_name, network_type="drive")
print(f"Graph has {len(G.nodes)} nodes and {len(G.edges)} edges")

gdf_edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
print("First few road segments:")
print(gdf_edges.head())

# -----------------------------
# 3. Reproject Road Network to a Metric CRS
# -----------------------------
# For Paris, UTM zone 31N (EPSG:32631) is commonly used.
gdf_edges = gdf_edges.to_crs(epsg=32631)

# -----------------------------
# 4. Buffer Road Segments
# -----------------------------
# Create a 30-meter buffer around each road segment.
buffer_distance = 30  # in meters
gdf_edges["buffer"] = gdf_edges.geometry.buffer(buffer_distance)

# -----------------------------
# 5. Retrieve POIs (Venues) Using OSMnx
# -----------------------------
tags = {'amenity': True, 'shop': True, 'tourism': True}
gdf_pois = ox.features_from_place(place_name, tags)
print("First few POIs from OSM:")
print(gdf_pois.head())

# -----------------------------
# 6. Reproject POIs to the Metric CRS
# -----------------------------
gdf_pois = gdf_pois.to_crs(epsg=32631)

# -----------------------------
# 7. Manually Count Venues per Road Segment Buffer
# -----------------------------
# For each road segment's buffer, count the number of POIs that fall within it.
gdf_edges["venue_count"] = gdf_edges["buffer"].apply(lambda buff: gdf_pois[gdf_pois.within(buff)].shape[0])
print("Road segments with venue counts (manual calculation):")
print(gdf_edges[["venue_count"]].head())

# -----------------------------
# 8. Visualize the Entire 5th Arrondissement
# -----------------------------
fig, ax = plt.subplots(figsize=(12, 12))
# Plot road network in black (thin lines)
gdf_edges.plot(ax=ax, color="black", linewidth=1, alpha=0.5)
# Plot the 200-meter buffers in light blue with blue edges
gdf_edges["buffer"].plot(ax=ax, color="lightblue", edgecolor="blue", alpha=0.3)
# Plot POIs in red
gdf_pois.plot(ax=ax, color="green", markersize=5, alpha=0.8)
plt.title("5th Arrondissement: Road Network, Buffers (200 m) & OSM Venues")
plt.xlabel("Easting (m)")
plt.ylabel("Northing (m)")
plt.show()
