#FROM jupyter/scipy-notebook
FROM jupyter/base-notebook:python-3.10.5
RUN pip3 install --upgrade pip wheel setuptools

RUN pip3 install poetry==1.1.14
RUN poetry --version
COPY pyproject.toml poetry.lock ./

RUN poetry export --without-hashes --dev -o requirements.txt

RUN pip3 install -r requirements.txt

USER 1000


COPY ./.aws/credentials ./.aws/credentials

COPY ./src ./src
COPY ./extracao ./extracao
