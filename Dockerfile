FROM jupyter/scipy-notebook

RUN pip3 install --upgrade pip wheel setuptools

RUN pip3 install poetry
RUN poetry --version
COPY pyproject.toml poetry.lock ./

RUN poetry export --without-hashes --dev -o requirements.txt

RUN pip3 install -r requirements.txt

USER 1000

COPY ./notebooks ./notebooks
COPY ./sample_data/1024x768/ ./sample_data/1024x768/
ENTRYPOINT ["jupyter","notebook"]