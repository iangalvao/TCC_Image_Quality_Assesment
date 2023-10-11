# Avaliação Estética de Imagens com Aprendizado de Máquina


## Descrição
Este repositório foi desenvolvido para o Projeto de Formatura Supervisionado do curso de Bacharelado em Ciência da Computação no IME- USP.
O software realiza a avaliação de qualidade de imagens com aprendizado de máquina supervisionado. A partir de imagens rotuladas com avaliações de qualidade feita por crowdsourcing, foram extraídas características estéticas das imagens, resultando em um vetor com 6 características para cada imagem. Esses vetores foram utilizados para treinar 5 modelos de regressão com o objetivo de predizer as avaliações de qualidade dos usuários do crowdsourcing. Os resultados de todos os modelos foram analisados e comparados.  

Como método alternativo, foi realizado um processo de extração de características automatizado com o modelo de rede neural pré-treinado VGG16, e em seguida treinado um modelo para comparação com os demais. O desempenho desse último modelo superou os outros 5.

## Instalação

### Dependências

- docker
- sistema operacional ubuntu

### Subindo o Container e Inicializando o Jupyter

rode:
ATENÇÃO: o comando abaixo irá baixar a imagem do docker e os pacotes necessários.

```
$ make build
```
e então:
```
$ make run
```

Parte dos notebooks utiliza o MLFLOW para armazenamento. Siga as instruções de https://github.com/Fernando-Freire/MLFlow_docker_compose_template para inicializar o serviço.


## Dados

Este repositório utiliza apenas uma amostra das imagens rotuladas do banco KonIQ-10k. O banco completo pode ser baixado em http://database.mmsp-kn.de/koniq-10k-database.html. Esse projeto utilizou como base as imagens em alta resolução disponíveis no link.

## Monografia

A monografia e o site do projeto podem ser acessados em https://iangalvao.github.io/.