clean:
	rm -rf .ipynb_checkpoints */.ipynb_checkpoints src/analytics/.ipynb_checkpoints
	rm -rf src/analytics.egg-info
	rm -rf .mypy_cache .pytest_cache
	rm -rf build

install:
	pip install .

develop:
	pip install -e ".[dev]"
	pre-commit install

format:
	black src/analytics setup.py
	isort src/analytics/*.py setup.py

check:
	black --check src/analytics setup.py
	flake8 src/analytics setup.py

test:
	pytest --nbval notebooks
