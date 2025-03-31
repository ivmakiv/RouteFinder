# Route Recommendation System for Bicycle Advertising

This project implements a route recommendation system tailored for advertising campaigns using bicycles. It computes optimal routes based on several metrics such as:

- **Number of different venues per street**
- **Traffic score**
- **Weather conditions**

The system is designed to help advertisers plan campaigns by suggesting routes that maximize exposure. You can run the system either locally via the command line or as a web application.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Local Usage (CLI)](#local-usage-cli)
  - [Web Usage (Flask)](#web-usage-flask)
- [Additional Notes](#additional-notes)

---

## Project Structure

```plaintext
.
└──Main
  ├── MainCode.py         # CLI version of the route recommendation system
  ├── app.py              # Flask app for the web interface
  ├── ConvertToGraph.py   # Converts road dataset into a graph structure
  ├── ObtainRD.py         # Generates or retrieves the road dataset
  ├── RouteFinder.py      # Contains route finding logic
  ├── Visualizations.py   # Visualizes the route (used for testing/CLI)
  └── templates
     └── index.html      # HTML template for the web interface
└── TestConnections     # Additional scripts for API connections
    ├── data_acquisition.py
    ├── OpenWeather.py
    ├── TomTomTile.py
    └── Venues.py

```

-MainCode.py: Entry point for local/CLI usage.

-app.py: Entry point for running the system as a web application.

-ConvertToGraph.py, ObtainRD.py, RouteFinder.py, Visualizations.py: Core modules for data processing and route computation.

-templates/index.html: HTML file used by the Flask web interface.

-TestConnections/: Utility scripts for testing API connections.
Installation

---

## Installation


Clone the repository:

```plaintext
git clone https://github.com/your-username/your-repo-name.git
```

Navigate into the project directory:

```plaintext
cd your-repo-name
```

(Optional) Create and activate a virtual environment:

```plaintext
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

Install the required Python packages:

```plaintext
pip install -r requirements.txt
```

## Configuration

Before running the application, create a config.json file in the root directory (the same level as app.py and MainCode.py) with the following content:


```plaintext
{
  "foursquare": {
    "api_key": "your_foursquare_api_key"
  },
  "tomtom": {
    "api_key": "your_tomtom_api_key"
  },
  "openweather": {
    "api_key": "your_openweather_api_key"
  }
}
```
Replace the placeholder API keys with your actual keys.

-Foursquare: For retrieving venue information.

-TomTom: For obtaining traffic or map data.

-OpenWeather: For fetching weather conditions.


## Usage

### Local Usage (CLI)

To run the route recommendation system in your IDE or via the command line:

Ensure config.json is created.

Run:

```plaintext
python MainCode.py
```

Follow the prompts to enter:

-Start latitude and longitude

-Finish latitude and longitude

-Maximum route length

The system will compute and display the optimal route based on your inputs.

### Web Usage (Flask)

To try the system with a web interface:

Ensure config.json is created.

Run the Flask app:

```plaintext
python app.py
```

Open your browser and navigate to http://127.0.0.1:5000/.

Fill in the form fields with the start/finish coordinates and maximum route length.

Click "Find Route" to view the recommended route on an interactive map. The map will also display the total route length if the route is successfully computed.


## Additional Notes

-API Keys: Ensure that your API keys are valid. Incorrect keys will cause the API-based data acquisition (Foursquare, TomTom, OpenWeather) to fail.

-Extensibility: The system is designed to be extensible. You can add more metrics or modify the route scoring algorithm as needed.

-Contributions: Contributions and improvements are welcome! Feel free to open issues or submit pull requests.
