
build:
	docker build -t iqa_mac0499 .

run:
	docker run --rm -p 8888:8888 -v $$(pwd)/notebooks/:/home/jovyan/notebooks/ iqa_mac0499 jupyter notebook

notebook:
	docker run --rm -it -p 8888:8888 -v $$(pwd)/sample_data/1024x768/:/home/jovyan/notebooks/ -v $$(pwd)/sample_data/1024x768/:/home/jovyan/sample_data/1024x768/ iqa_mac0499 
	