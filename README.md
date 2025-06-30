# NYC Taxi Fare Prediction

This project provides an end-to-end machine learning pipeline to predict taxi fares in New York City using historical ride data. The workflow is:

* Version-controlled with **DVC** for data and model reproducibility
* Managed with **Poetry** for dependency management
* Tracked with **MLflow** for experiment logging
* Deployed as an API on **Render**
* Integrated with **DagsHub** for remote experiment tracking and artifact storage, enabling scalable and collaborative MLOps


## Project Overview

The goal is to build a robust regression model that can accurately estimate the fare amount for a taxi ride based on features like:

- Pickup and dropoff locations
- Time and day of ride
- Passenger count
- And other engineered features

I used open datasets such as the https://www.kaggle.com/competitions/new-york-city-taxi-fare-prediction/data.

## Project Structure
```
nyc_tfp/
├── data/                     # Raw, processed, and split data
├── models/                   # Trained model and preprocessor
├── evaluation/               # Evaluation results and metrics
├── src/nyc_tfp/              # Pipeline source code
│   ├── load_data.py
│   ├── preprocess.py
│   ├── train_model.py
│   └── evaluate_model.py
├── notebooks/                # Exploratory notebooks
├── tests/                    # Tests
├── mlruns/                   # MLflow tracking
├── params.yaml               # Pipeline parameters
├── dvc.yaml                  # DVC pipeline definition
├── pyproject.toml            # Poetry configuration
└── README.md                 # Project documentation
```



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

## Run the Pipeline


This project uses DVC to manage the machine learning pipeline in modular stages. To iterate on experiments and reproduce results, follow these steps:

1. **Run DVC experiments** incrementally until you achieve satisfactory results:

   ```bash
   dvc exp run
   ```

   This executes the next pipeline stage and logs the experiment.

2. **Apply the best experiment** to the workspace when ready:

   ```bash
   dvc exp apply <experiment_id>
   ```

   This updates your workspace with the selected experiment’s outputs.

3. **Reproduce the full pipeline** to ensure all stages are consistent and outputs are up-to-date:

   ```bash
   dvc repro
   ```

4. **Push data and model artifacts** to remote storage and commit changes to Git using the helper script provided:

   ```bash
   python run_pipeline.py -m "commit message describing changes"
   ```

   This script will:

   * Reproduce the full pipeline with `dvc repro`
   * Push DVC-tracked files to the remote storage (`dvc push`)
   * Stage changes and commit them to Git with the specified commit message


## ✅ Testing

Run all tests with:

```bash
poetry run pytest
```
