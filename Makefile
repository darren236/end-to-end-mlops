PYTHON ?= python3

.PHONY: install train test lint serve ui notebooks notebooks-check monitor docker-build clean

install:
	$(PYTHON) -m pip install -e ".[dev]"

train:
	$(PYTHON) -m mlops_demo.train

test:
	pytest --cov=mlops_demo --cov-report=term-missing

lint:
	ruff check .

serve:
	uvicorn mlops_demo.api:app --host 0.0.0.0 --port 8000 --reload

ui:
	streamlit run src/mlops_demo/ui.py --server.address 0.0.0.0 --server.port 8501

notebooks:
	$(PYTHON) -m jupyterlab notebooks

notebooks-check:
	mkdir -p /tmp/mlops-notebook-runs
	$(PYTHON) -m nbconvert --to notebook --execute --ExecutePreprocessor.timeout=180 --output-dir /tmp/mlops-notebook-runs notebooks/*.ipynb

monitor:
	$(PYTHON) -m mlops_demo.monitor

docker-build:
	docker build -t mlops-iris-demo:latest .

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .coverage .pytest_cache .ruff_cache artifacts coverage.xml htmlcov logs mlruns
