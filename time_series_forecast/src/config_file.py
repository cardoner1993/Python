from pathlib import Path

root_path = Path(__file__).parent.parent.absolute()
data_path = root_path / 'data'
data_results_path = data_path / 'results'
visualizations_path = root_path / 'reports'


Path.mkdir(root_path, exist_ok=True, parents=True)
Path.mkdir(data_path, exist_ok=True, parents=True)
Path.mkdir(data_results_path, exist_ok=True, parents=True)
Path.mkdir(visualizations_path, exist_ok=True, parents=True)