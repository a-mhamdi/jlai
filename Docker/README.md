# DOCKER

This repository is a stack of ready-to-run Docker images containing `Julia` and interactive computing tools: `Jupyter Lab` and `Pluto`. The repository contains two main files:
* `Dockerfile` file I use to create an image which contains the build context to run codes of machine learning using `Julia`. `IJulia`, `Jupyter Lab`, and `Pluto` are also being installed.
* `docker-compose.yml` file which runs two services, `jupyter` and `pluto` to execute `Julia` codes. As for `Julia`, it is built using the context in the `Dockerfile`. The `jupyter` and `pluto` services map ports _2468_ and _1234_ on the host to ports _2468_ and _1234_ on the containers.

**GitHub Actions** build and push this image to [dockerhub](https://hub.docker.com/). Every update is available at [abmhamdi/jlai](https://hub.docker.com/repository/docker/abmhamdi/jlai)
