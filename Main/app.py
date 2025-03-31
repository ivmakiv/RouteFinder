from flask import Flask, render_template, request
from pyproj import Transformer
import json

# Import your project modules.
from ObtainRD import generate_road_dataset
from ConvertToGraph import dataset_to_graph
from RouteFinder import find_route

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    route_json = None
    status_message = ""
    total_length = None
    start_user = None
    finish_user = None

    if request.method == 'POST':
        try:
            # Read form inputs (in EPSG:4326)
            start_lat = float(request.form['start_lat'])
            start_lon = float(request.form['start_lon'])
            finish_lat = float(request.form['finish_lat'])
            finish_lon = float(request.form['finish_lon'])
            max_length = float(request.form['max_length'])

            # 1. Generate the road dataset.
            road_dataset = generate_road_dataset(start_lat, start_lon, finish_lat, finish_lon, radius=1000)

            # 2. Convert the dataset to a graph.
            G = dataset_to_graph(road_dataset)

            # 3. Find the best route.
            best_route, total_length, status_message = find_route(
                G, road_dataset, start_lat, start_lon, finish_lat, finish_lon, max_length
            )

            if best_route is None:
                status_message = "No route found within the given constraints."
            else:
                # Transform the best route's nodes (from EPSG:32631 to EPSG:4326).
                transformer_back = Transformer.from_crs("EPSG:32631", "EPSG:4326", always_xy=True)
                route_latlon = []
                for node in best_route:
                    x = G.nodes[node]["x"]
                    y = G.nodes[node]["y"]
                    lon, lat = transformer_back.transform(x, y)
                    # Leaflet expects coordinates as [lat, lon]
                    route_latlon.append([lat, lon])
                route_json = route_latlon

                # Use the provided start/finish (which are in EPSG:4326).
                start_user = [start_lat, start_lon]
                finish_user = [finish_lat, finish_lon]

        except Exception as e:
            status_message = f"Error: {e}"

    return render_template(
        'index.html',
        route_json=route_json,
        status_message=status_message,
        total_length=total_length,
        start_user=start_user,
        finish_user=finish_user
    )


if __name__ == '__main__':
    app.run(debug=True)
