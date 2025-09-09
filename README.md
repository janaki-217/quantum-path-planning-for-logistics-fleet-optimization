# Vehicle Route Visualization (Long Route vs Shortest Route)

This project demonstrates how to visualize and compare *different routing strategies* for delivery customers using Python.
It integrates *OpenStreetMap (OSRM) API, **Folium maps, and simple heuristics to showcase **long route (original order)* vs *short route (nearest neighbor)*.

---

## ✨ Features

* Load customer data from Excel (Customer Data.xlsx)
* Compute pairwise distances using *Geopy (geodesic distance)*
* Build a *long route* (original input order)
* Compute a *short route* using *Nearest Neighbor heuristic*
* Generate routes via *OSRM* (road network routing)
* Interactive map visualization with *Folium*:

  * Left map → Long route in *red*
  * Right map → Shortest route in *green*
* Side-by-side synchronized maps for direct comparison

---

## 📂 Dataset

The dataset is provided in an Excel file Customer Data.xlsx with the following columns:

* CUSTOMER ID – Unique ID of the customer
* LATITUDE – Latitude coordinate
* LONGITUDE – Longitude coordinate

---

## ⚙ Prerequisites

Make sure you have Python *3.8+* installed.

### Required Packages

Install dependencies with:

```bash
pip install pandas numpy geopy folium requests openpyxl

```

---

## 🚀 Usage

### 1. Load Customer Data

```python

import pandas as pd

df_demo = pd.read_excel("Customer Data.xlsx")
```


### 2. Compute Routes

* *Long Route:* Original input order of customers
* *Short Route:* Nearest neighbor heuristic

The nearest neighbor heuristic starts from the first customer and repeatedly visits the closest unvisited customer until all are visited.

### 3. Visualize on Map

We use *Folium’s DualMap* to display two maps side by side:

* *Left Map (Red):* Long route (original order)
* *Right Map (Green):* Shortest route (nearest neighbor)

Example code:

```python
from folium.plugins import DualMap

# Initialize dual map
m = DualMap(location=[df_demo['LATITUDE'].mean(), df_demo['LONGITUDE'].mean()],
            zoom_start=12)

# Add long route (red) to left map
folium.GeoJson(long_route_geojson,
    style_function=lambda x: {'color': 'red', 'weight': 5}).add_to(m.m1)

# Add short route (green) to right map
folium.GeoJson(short_route_geojson,
    style_function=lambda x: {'color': 'green', 'weight': 5}).add_to(m.m2)

# Display
m
```

---

## 📊 Example Output

| Long Route (Original Order)        | Short Route (Nearest Neighbor)       |
| ---------------------------------- | ------------------------------------ |
| ![long-route](![picture1](https://github.com/user-attachments/assets/b57446d8-209b-4e02-bc5d-6c731115ad51)
") | ![short-route](![picture2](https://github.com/user-attachments/assets/1bd93d55-6ea2-405a-97f8-7f636f8be518)
![Uploading picture2.jpg…]()
) |


---

## 🔍 Result Comparison

* *Long Route (Red):*

  * Preserves original order of input
  * May be inefficient in terms of distance
* *Short Route (Green):*

  * Uses Nearest Neighbor heuristic
  * Reduces travel distance, but not guaranteed to be globally optimal

---

## 🛠 Extensions

* Integrate *Google OR-Tools* to solve *Capacitated Vehicle Routing Problem (CVRP)*
* Add *vehicle capacity & demand constraints*
* Include *dynamic routing* with traffic data
* Compare with *metaheuristics (Genetic Algorithm, Tabu Search)*

---

## 📖 References

* [Capacitated Vehicle Routing Problem (CVRP) – GitHub Example](https://github.com/ngchunlong279645/Capacitated-Vehicle-Routing-Problem-CVRP-)
* [OSRM Routing API](http://project-osrm.org/)
* [Folium Documentation](https://python-visualization.github.io/folium/)

