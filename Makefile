
build:
	docker build -t iqa_mac0499 .

run:
	docker run --rm -it -p 8888:8888 -v $$(pwd)/notebooks/:/home/jovyan/notebooks/ -v $$(pwd)/extracao/:/home/jovyan/extracao/ iqa_mac0499
