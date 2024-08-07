import papermill as pm
from src.utils import create_dir_if_not_exists

notebooks = [
  '01-data-collection.ipynb',
  '02-exploratory-data-analysis.ipynb',
  '03-data-preparation.ipynb',
  '04-modelisation.ipynb'
]

# Check if output dir exists and create it if not
dir_to_create = "notebooks/outputs"
create_dir_if_not_exists(dir_to_create)

for notebook in notebooks:
    print(f"Executing {notebook}")
    notebook_path = f"notebooks/{notebook}"
    output_notebook = f"{'notebooks/outputs'}/{notebook.replace('.ipynb', '_output.ipynb')}"
    pm.execute_notebook(notebook_path, output_notebook)