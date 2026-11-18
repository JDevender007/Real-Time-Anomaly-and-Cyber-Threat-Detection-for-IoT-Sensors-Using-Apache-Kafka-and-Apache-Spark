.PHONY: install test lint typecheck train-physical train-cyber start-api start-dashboard start-simulator start-streaming evaluate clean help

help:
	@echo "IoT Cyber-Physical Security Platform"
	@echo "======================================"
	@echo "  make install          Install dependencies"
	@echo "  make test             Run tests"
	@echo "  make lint             Run linter"
	@echo "  make typecheck        Run type checker"
	@echo "  make train-physical   Train physical anomaly models"
	@echo "  make train-cyber      Train cyber threat models"
	@echo "  make start-api        Start FastAPI server"
	@echo "  make start-dashboard  Start Streamlit dashboard"
	@echo "  make start-simulator  Start IoT simulator"
	@echo "  make start-streaming  Start streaming pipeline"
	@echo "  make evaluate         Run evaluation experiments"
	@echo "  make clean            Clean generated files"

install:
	pip install -r requirements.txt
	pip install -e ".[dev]"

test:
	python -m pytest tests/ -v --tb=short

test-unit:
	python -m pytest tests/unit/ -v --tb=short

test-integration:
	python -m pytest tests/integration/ -v --tb=short

lint:
	ruff check src/ tests/ scripts/

typecheck:
	mypy src/

train-physical:
	python scripts/train_physical_models.py

train-cyber:
	python scripts/train_cyber_models.py

start-api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

start-dashboard:
	streamlit run dashboard/app.py --server.port 8501

start-simulator:
	python scripts/start_simulator.py

start-streaming:
	python scripts/start_streaming.py

evaluate:
	python scripts/run_evaluation.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf logs/*.log
