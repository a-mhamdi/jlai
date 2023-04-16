## Fuzzy Logic, Machine Learning and Deep Learning with Julia

This repository contains slides, labs and code examples for using `Julia` to implement some **artificial intelligence** related algorithms. Codes run on top of a `Docker` image, ensuring a consistent and reproducible environment.

[![JLAI CI](https://github.com/a-mhamdi/jlai/actions/workflows/docker-image.yml/badge.svg)](https://github.com/a-mhamdi/jlai/actions/workflows/docker-image.yml)
[![Docker Version](https://img.shields.io/docker/v/abmhamdi/jlai?sort=semver)](https://hub.docker.com/r/abmhamdi/jlai)
[![Docker Pulls](https://img.shields.io/docker/pulls/abmhamdi/jlai)](https://hub.docker.com/r/abmhamdi/jlai)
[![Docker Stars](https://img.shields.io/docker/stars/abmhamdi/jlai)](https://hub.docker.com/r/abmhamdi/jlai)

To run the code, you will need to first pull the `Docker` image by running the following command:

```zsh
docker pull abmhamdi/jlai
```

This may take a while, as it will download and install all necessary dependencies.

### How to control the containers:

* ```docker-compose up``` starts the container
* ```docker-compose down``` stops and destroys the container

Services can be run by typing the command `docker compose up`. This will start the `Jupyter Lab` on [http://localhost:2468](http://localhost:2468) and you should be able to use `Julia` from within the notebook by starting a new `Julia` notebook. You can parallelly start `Pluto` on [http://localhost:1234](http://localhost:1234).

### Included Algorithms
The repository includes implementation of the following algorithms:
>1. Linear Regression, Logistic Regression, k-NN, SVM, K-MEANS
>1. Fuzzy Inference System (FIS), Fuzzy Logic Controller
>1. ANN, CNN, GAN, VAE, NLP
>1. Transfer Learning
>1. Reinforcement Learning

### Prerequisites
You will need to have Docker installed on your machine. You can download it from the [Docker website](https://hub.docker.com).

### License
This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
