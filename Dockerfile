FROM jupyter/scipy-notebook

RUN pip3 install --upgrade pip wheel setuptools

RUN pip3 install poetry
RUN poetry --version
COPY pyproject.toml poetry.lock ./

RUN poetry export --without-hashes --dev -o requirements.txt

RUN pip3 install -r requirements.txt

USER 1000




COPY ./src ./src
COPY ./extracao ./extracao
