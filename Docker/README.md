# DOCKER

This repository is a stack of ready-to-run Docker images containing `Julia` and interactive computing tools: `Jupyter Lab` and `Pluto`. The repository contains two main files: _NEEDS FURTHER MAINTENANCE_
* `Dockerfile` file  I use to create an image which contains the build context to run codes of machine learning using `Julia`. `Pluto` and `IJulia` are also being installed.
* `docker-compose.yml` file which runs two services, `julia` and `jupyter`, using the latest version of the `minimal-notebook` image. As for `Julia`, it is built using the context in the `Dockerfile`. The `jupyter` service maps port _2468_ on the host to port _8888_ on the container, and it mounts the current directory on the host to `/app` in the container. The `julia` service also mounts the current directory on the host to `/app` in the container, and runs the `julia` command.

You can then start the services by running the command `docker-compose up` in the same directory as the compose file. This will start the Jupyter notebook on [http://localhost:2468](http://localhost:2468) and you should be able to use `Julia` from within the notebook by starting a new `Julia` notebook. You can parallelly start `Pluto on [http://localhost:1234](http://localhost:1234).

**GitHub Actions** build and push this image to [dockerhub](https://hub.docker.com/). Every update is available at [abmhamdi/jlai](https://hub.docker.com/repository/docker/abmhamdi/jlai)
