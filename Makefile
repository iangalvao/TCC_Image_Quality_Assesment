
build:
	docker build -t teste-poetry .

run:
	docker run --rm -it -p 8888:8888 -v $$(pwd)/src/:/home/jovyan/src/ teste-poetry
