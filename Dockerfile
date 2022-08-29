FROM jupyter/scipy-notebook

RUN pip3 install --upgrade pip wheel setuptools

RUN pip3 install poetry
RUN poetry --version
COPY pyproject.toml poetry.lock ./

RUN poetry export --without-hashes --dev -o requirements.txt

RUN pip3 install -r requirements.txt

USER 1000

ENV MLFLOW_TRACKING_URI=http://localhost:5000
ENV MLFLOW_S3_ENDPOINT_URL=http://localhost:9000

COPY ./aws/credentials ~/.aws/credentials

COPY ./src ./src
COPY ./extracao ./extracao
