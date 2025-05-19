# NYC Taxi Fare Prediction

This project is a **machine learning regression pipeline** that aims to predict taxi fares in **New York City** using historical ride data. The project is managed using **[Poetry](https://python-poetry.org/)** for dependency and environment management, and **[MLflow](https://mlflow.org/)** for tracking model experiments.

## 🧠 Project Overview

The goal is to build a robust regression model that can accurately estimate the fare amount for a taxi ride based on features like:

- Pickup and dropoff locations
- Time and day of ride
- Passenger count
- And other engineered features

I used open datasets such as the https://www.kaggle.com/competitions/new-york-city-taxi-fare-prediction/data.

## 📦 Project Structure
nyc_tfp/
├── data/ # Raw and processed data
├── notebooks/ # Jupyter notebooks for EDA and prototyping
├── src/ # Source code
│ ├── data/ # Data loading and preprocessing
│ ├── features/ # Feature engineering
│ ├── models/ # Model training and evaluation
│ └── utils/ # Helper functions
├── tests/ # Unit and integration tests
├── pyproject.toml # Poetry configuration file
├── mlruns/ # MLflow tracking data
└── README.md # Project description\


## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/napsugark/nyc-taxi-fare-prediction.git
cd nyc-taxi-fare-prediction
```

### 2. Install Poetry (if not installed)

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### 3. Install Dependencies

```bash
poetry install
```


### 4. Activate the Environment

```bash
poetry shell
```

## 🚀 Run the Pipeline

### Training the model

```bash
python src/models/train_model.py
```

### Launch MLFlow UI
```bash
mlflow ui
```
Then open http://localhost:5000 in your browser to explore experiments, metrics, and model versions.

## ✅ Testing

Run all tests with:

```bash
poetry run pytest
```
