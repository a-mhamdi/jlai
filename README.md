# Fuzzy Logic, Machine Learning and Deep Learning with Julia

This repository contains slides, labs and code examples for using `Julia` to implement some **artificial intelligence** related algorithms. Codes run on top of a `Docker` image, ensuring a consistent and reproducible environment.


[![CI/CD](https://github.com/a-mhamdi/jlai/actions/workflows/jlai-p1.yml/badge.svg)](https://github.com/a-mhamdi/jlai/actions/workflows/jlai-p1.yml)
[![CI/CD](https://github.com/a-mhamdi/jlai/actions/workflows/jlai-p2.yml/badge.svg)](https://github.com/a-mhamdi/jlai/actions/workflows/jlai-p2.yml)
[![CI/CD](https://github.com/a-mhamdi/jlai/actions/workflows/jlai-p3.yml/badge.svg)](https://github.com/a-mhamdi/jlai/actions/workflows/jlai-p3.yml)

[![Docker Pulls](https://img.shields.io/docker/pulls/abmhamdi/jlai-p1)](https://hub.docker.com/r/abmhamdi/jlai-p1)
[![Docker Pulls](https://img.shields.io/docker/pulls/abmhamdi/jlai-p2)](https://hub.docker.com/r/abmhamdi/jlai-p2)
[![Docker Pulls](https://img.shields.io/docker/pulls/abmhamdi/jlai-p3)](https://hub.docker.com/r/abmhamdi/jlai-p3)


## Included Algorithms

The repository includes implementation of the following algorithms:
>1. Fuzzy Logic, Fuzzy Inference Systems (FIS): Mamdani, Sugeno and Tsukamoto
>1. Linear Regression, Logistic Regression, Naive Bayes, Decision Tree, k-NN, SVM, K-MEANS, and PCA
>1. ANN, CNN, Transfer Learning, GAN, VAE, NLP
>1. Reinforcement Learning

> [!NOTE]
> To run the code, you will need to first pull the `Docker` image by running the following command:
>
> ```zsh
> docker pull abmhamdi/jlai-p1
> ```
> 
> This may take a while, as it will download source code, `julia` image, and all necessary system dependencies.

## How to control the containers:

Services can be started by typing the command:

 ```zsh
docker compose up -d # starts the containers in detached mode
```
```zsh
docker compose down # stops and removes them
```

This will launch the `Jupyter Lab` on [http://localhost:2468](http://localhost:2468), and you should be able to use `Julia` from within the notebook by starting a new `Julia` notebook. You can parallelly use `Pluto` on [http://localhost:1234](http://localhost:1234).

> [!IMPORTANT]
> 
> You will need to have Docker installed on your machine. You can download it from the [Docker website](https://hub.docker.com).

## License
This project is licensed under the MIT License - see the [LICENSE](https://raw.githubusercontent.com/a-mhamdi/jlai/refs/heads/main/LICENSE) file for details.
