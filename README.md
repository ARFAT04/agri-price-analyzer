# Agricultural Market Price Analyzer

Dashboard to analyse and forecast mandi prices of onion and potato in
Uttar Pradesh, Maharashtra and Madhya Pradesh, with weather correlation.

## Data
- Prices: Agmarknet mandi prices (Kaggle: "Indian Agricultural Mandi Prices 2023-2025"), Jun 2023 - Jun 2025
- Weather: Open-Meteo historical API

## Setup
    pip install -r requirements.txt
    python src/preprocess.py
    python src/weather.py

## Status
Work in progress.