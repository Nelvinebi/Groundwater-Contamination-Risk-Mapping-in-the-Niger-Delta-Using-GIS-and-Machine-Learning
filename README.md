Groundwater Contamination Risk Mapping in the Niger Delta Using GIS and Machine Learning
📌 Project Overview

This project develops a spatially explicit groundwater contamination risk map for the Niger Delta using GIS and machine learning. Synthetic hydro-environmental datasets are integrated to model contamination susceptibility and generate raster and vector risk outputs.

🎯 Objectives

Model groundwater contamination risk using ML classifiers

Integrate hydrogeological and anthropogenic factors in GIS

Produce GeoTIFF and shapefile risk maps for decision support

🗂️ Project Structure
├── data/
│   └── groundwater_contamination_dataset.xlsx
├── scripts/
│   └── groundwater_contamination_risk_ml.py
├── outputs/
│   ├── groundwater_contamination_risk_niger_delta.tif
│   └── groundwater_contamination_risk_zones.shp
├── README.md

🧪 Dataset Description

Synthetic but realistic variables include:

Depth to groundwater

Nitrate concentration

Electrical conductivity

Land use intensity

Distance to pollution sources

Soil permeability

Target variable: Groundwater contamination risk (Low, Moderate, High)

🧠 Methodology Summary

Data preprocessing and normalization

Supervised ML classification (Random Forest)

Rasterization and spatial prediction

Risk zoning and GIS visualization

🗺️ GIS Outputs

GeoTIFF: Continuous groundwater contamination risk surface

Shapefile: Classified contamination risk zones

🛠️ Tools & Libraries

Python, NumPy, Pandas

Scikit-learn

Rasterio, GeoPandas, Shapely

QGIS / ArcGIS for visualization

📍 Study Area

Niger Delta region, Nigeria (WGS84 – EPSG:4326)

👤 Author

AGBOZU EBINGIYE NELVIN

LinkedIn: *https://www.linkedin.com/in/agbozu-ebi/
