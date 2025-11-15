install:
	pip install -r requirements.txt

download:
	python data/download.py

train-florence:
	python -m src.training.train_florence2

train-contrastive:
	python -m src.training.train_contrastive

build-index:
	python scripts/build_index.py

benchmark:
	python scripts/run_benchmark.py

app:
	python app/gradio_app.py

test:
	pytest tests/ -v
