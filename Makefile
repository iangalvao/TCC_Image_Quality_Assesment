
build:
	docker build -t iqa_mac0499 .

run:
	docker run --rm -p 8888:8888 --env AWS_ACCESS_KEY_ID=minio --env AWS_SECRET_ACCESS_KEY=minio123 --env MLFLOW_S3_ENDPOINT_URL=http://172.27.0.1:9000 -v $$(pwd)/notebooks/:/home/jovyan/notebooks/  -v $$(pwd)/sample_data/1024x768/:/home/jovyan/sample_data/1024x768/ iqa_mac0499 jupyter notebook

notebook:
	docker run --rm -it -p 8888:8888 -v $$(pwd)/sample_data/1024x768/:/home/jovyan/notebooks/ -v $$(pwd)/sample_data/1024x768/:/home/jovyan/sample_data/1024x768/ iqa_mac0499 
	