# Avaliação Estética de Imagens com Aprendizado de Máquina
Projeto de extração de features e treinamento de Projeto de Formatura Supervisionado (MAC0499)

## Descrição

Este software realiza a extração de características estéticas de imagens retirada dos banco KonIQ-10k e treina 6 famílias de modelos com os resultados.
## Instalação

### Dependências

- docker
- poetry 1.1.14
- sistema operacional ubuntu
- python 10.5 (importante usar essa versão)
### Installing

- Instalação do Docker: https://docs.docker.com/engine/install/ubuntu/

- Instale o poetry 


### Subindo o Container e Inicializando o Jupyter

rode (O COMANDO ABAIXO IRA BAIXAR AS IMAGENS DO DOCKER E PACOTES!)


```
make build
```
e então:
```
make run
```

Parte dos notebooks utiliza o MLFLOW para armazenamento. Siga as instruções de https://github.com/Fernando-Freire/MLFlow_docker_compose_template.

Após o uso rode:

make clean

## Dados

Este repositório utiliza apenas uma amostra das imagens rotuladas do banco KonIQ-10k. O banco completo pode ser baixado em http://database.mmsp-kn.de/koniq-10k-database.html. Esse projeto utilizou como base as imagens em alta resolução disponíveis no link.

## Monografia

A monografia e o site do projeto podem ser acessados em https://iangalvao.github.io/.