# DOCKER

This repository is a stack of ready-to-run Docker images containing `Julia` and interactive computing tools: `Jupyter Lab` and `Pluto`. The repository contains two main files:
* `Dockerfile` file I use to create an image which contains the build context to run codes of machine learning using `Julia`. `Jupyter Lab`, `Pluto` and `IJulia` are also being installed.
* `docker-compose.yml` file which runs two services, `jupyter` and `pluto` to execute `Julia` codes. As for `Julia`, it is built using the context in the `Dockerfile`. The `jupyter` and `pluto` services map ports _2468_ and _1234_ on the host to ports _2468_ and _1234_ on the containers.

Services can be run by typing the command `docker compose up`. This will start the `Jupyter Lab` on [http://localhost:2468](http://localhost:2468) and you should be able to use `Julia` from within the notebook by starting a new `Julia` notebook. You can parallelly start `Pluto` on [http://localhost:1234](http://localhost:1234).

**GitHub Actions** build and push this image to [dockerhub](https://hub.docker.com/). Every update is available at [abmhamdi/jlai](https://hub.docker.com/repository/docker/abmhamdi/jlai)
