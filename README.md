# Building Energy Prediction

This project aims to predict building energy usage using various machine learning techniques. The project includes data collection, data preprocessing, exploratory data analysis, model training, and deployment.

## Project Structure

```plaintext
Building-Energy-Prediction-main/
├── .gitignore
├── requirements.txt
├── README.md
├── settings/
│   ├── __init__.py
│   └── params.py
├── datasets/
│   ├── 2015-building-energy-benchmarking.csv
│   ├── 2016-building-energy-benchmarking.csv
│   ├── cleaned_data.csv
│   └── raw_data.csv
├── tests/
│   ├── __init__.py
│   ├── evaluate_models_tests.py
│   ├── eval_metrics_tests.py
│   ├── test_dataset.py
│   ├── test_define_pipeline.py
│   ├── test_evaluate_models.py
│   ├── test_tracking_tests.py
│   ├── test_tuning.py
│   └── tuning_tests.py
├── notebooks/
│   ├── 01-data-collection.ipynb
│   ├── 02-exploratory-data-analysis.ipynb
│   ├── 03-data-preparation.ipynb
│   ├── 04-modelisation.ipynb
│   └── __init__.py
├── .github/
│   └── workflows/
│       ├── azure-pipelines.yml
│       ├── deploy-model.yml
│       └── register_models_pipeline.yml
└── src/
    ├── __init__.py
    ├── azure_ml.py
    ├── collect_data.py
    ├── deploy.py
    ├── evaluate.py
    ├── main.py
    ├── metrics.py
    ├── pipeline.py
    ├── plot.py
    ├── tracking.py
    └── utils.py
```

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.10.10 or higher
- Azure CLI
- Virtual environment (optional but recommended)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/Building-Energy-Prediction.git
   cd Building-Energy-Prediction
   ```

2. Create a virtual environment and activate it:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Set up your Azure credentials. You can use the Azure CLI to log in:
   ```bash
   az login
   ```

### Usage

#### Notebooks

You can find Jupyter notebooks in the `notebooks/` directory that demonstrate data collection, preparation, and exploratory data analysis:

- `01-data-collection.ipynb`
- `02-exploratory-data-analysis.ipynb`
- `03-data-preparation.ipynb`
- `04-modelisation.ipynb`

Each of these notebooks contains some parts of the machine learning process. If you want to run them, no need to do it one by one: you can run the script src/main.py with `python src/main.py`. Outputs will be generated for each of the notebooks in the `notebooks/outputs/` directory

#### Modules

In the src folder, you'll find different modules containing methods that we use in the notebooks. Two of those modules are particularly special:

- `main.py` which contains the main script that executes all the notebooks
- `deploy.py` contains the code that will pick the best model obtained after training and deploy it to Azure Machine Learning for Online Inference

#### Running Tests

Run the tests to ensure everything is working correctly:

```bash
pytest tests/
```
