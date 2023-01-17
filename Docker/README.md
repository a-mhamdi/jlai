# JLAI

A convenient way to run machine learning codes on multiple platforms via Docker.

[![JLAI CI](https://github.com/a-mhamdi/jlai/actions/workflows/docker-image.yml/badge.svg)](https://github.com/a-mhamdi/jlai/actions/workflows/docker-image.yml)
[![Docker Version](https://img.shields.io/docker/v/abmhamdi/jlai?sort=semver)](https://hub.docker.com/r/abmhamdi/jlai)
[![Docker Pulls](https://img.shields.io/docker/pulls/abmhamdi/jlai)](https://hub.docker.com/r/abmhamdi/jlai)
[![Docker Stars](https://img.shields.io/docker/stars/abmhamdi/jlai)](https://hub.docker.com/r/abmhamdi/jlai)

This repository is a stack of ready-to-run Docker images containing `Julia` and interactive computing tools: `Jupyter Lab` and `Pluto`. The repository contains two main files:
1. `Dockerfile` file  I use to create an image which contains the build context to run codes of machine learning using `Julia`.
1. `docker-compose.yml` file which runs two services, `julia` and `jupyter`, using the latest version of the `Jupyter` image. As for `Julia`, it is built using the context in the `Dockerfile`. The `jupyter` service maps port _4321_ on the host to port _8888_ on the container, and it mounts the current directory on the host to `/app` in the container. The `julia` service also mounts the current directory on the host to `/app` in the container, and runs the `julia` command.

You can then start the services by running the command `docker-compose up` in the same directory as the compose file. This will start the Jupyter notebook on [http://localhost:4321](http://localhost:431) and you should be able to use `Julia` from within the notebook by starting a new `Julia` notebook.

**GitHub Actions** build and push this image to [dockerhub](https://hub.docker.com/). Every update is available at [abmhamdi/jlai](https://hub.docker.com/repository/docker/abmhamdi/jlai)

