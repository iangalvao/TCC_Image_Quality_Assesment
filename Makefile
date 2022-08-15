
build:
	docker build -t teste-poetry .

run:
	docker run --rm -it -p 8888:8888 -v $$(pwd)/notebooks/:/home/jovyan/notebooks/ teste-poetry
