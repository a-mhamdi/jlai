# DOCKER

This repository is a stack of ready-to-run Docker images containing `Julia` and interactive computing tools: `Jupyter Lab` and `Pluto`. You'll find two main kinds of files in this repository:
* `Dockerfile-\d` file I use to create an image which contains the build context to run codes of artificial intelligence using `Julia`.
* `docker-compose.yml` file which runs two services, `jupyter` and `pluto` to execute `Julia` codes. As for `Julia`, it is built using the context in the `Dockerfile`. The `jupyter` and `pluto` services map ports _2468_ and _1234_ on the host to ports _2468_ and _1234_ on the containers.

**GitHub Actions** build and push this image to [dockerhub](https://hub.docker.com/). Every update is available at [abmhamdi/jlai-p1](https://hub.docker.com/repository/docker/abmhamdi/jlai-p1)
