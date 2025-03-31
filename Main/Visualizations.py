import matplotlib.pyplot as plt
from pyproj import Transformer


def visualize_route(G, best_route, start_lon, start_lat, finish_lon, finish_lat):
    """
    Visualize the road network and overlay the best route.

    Parameters:
      - G: NetworkX graph (in metric CRS, e.g., EPSG:32631) with nodes having 'x' and 'y' attributes.
      - best_route: List of node IDs representing the best route.
      - start_lon, start_lat: Start coordinates in EPSG:4326.
      - finish_lon, finish_lat: Finish coordinates in EPSG:4326.

    The function transforms the start/finish coordinates to the graph's CRS,
    plots the entire network in light gray, overlays the best route in red,
    and marks the start and finish points in green and blue respectively.
    """
    # Transform user coordinates from EPSG:4326 to EPSG:32631.
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
