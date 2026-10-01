# Docker

This directory contains ready-to-run Docker images with [Julia](https://julialang.org/) and two interactive computing tools: [Jupyter Lab](https://jupyter.org/) and [Pluto](https://plutojl.org/). Together they provide a consistent, reproducible environment for the AI code samples in this repository.

## Contents

- **`Dockerfile-<n>`**: build context for the images used to run the artificial intelligence code in `Julia`. `<n>` is the variant number.
- **`compose.yml`**: defines two services that run `Julia` code, both built from the Dockerfile:
  - `jupyter`: Jupyter Lab, available at <http://localhost:2468>
  - `pluto`: Pluto, available at <http://localhost:1234>

  Each service maps its port on the host to the same port in the container (2468 and 1234).

## Usage

```bash
docker compose up -d   # start both services in the background
docker compose down    # stop and remove the containers
```

## Continuous integration

**GitHub Actions** builds the image and pushes it to [Docker Hub](https://hub.docker.com/). Every update is published at [abmhamdi/jlai-p1](https://hub.docker.com/r/abmhamdi/jlai-p1).
