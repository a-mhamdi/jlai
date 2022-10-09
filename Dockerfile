FROM --platform=${BUILDPLATFORM} julia:1.6

ENV USER ml
ENV USER_HOME_DIR /home/${USER}
ENV JULIA_DEPOT_PATH ${USER_HOME_DIR}/.julia
ENV WORKING_DIR ${USER_HOME_DIR}/repo

RUN useradd -m -d ${USER_HOME_DIR} ${USER} \
    && mkdir -p ${WORKING_DIR}\
    && mkdir -p ${USER_HOME_DIR}/.julia/environments/v1.6/

COPY ./toml/* ${USER_HOME_DIR}/.julia/environments/v1.6/

RUN julia -e "import Pkg; Pkg.activate(); Pkg.instantiate(); Pkg.precompile();" \
	&& chown -R ${USER} ${USER_HOME_DIR}

USER ${USER}

WORKDIR ${WORKING_DIR}

# Default command: Julia Pluto
CMD julia -e "import Pluto; Pluto.run(host=\"0.0.0.0\", port=1234, launch_browser=false, require_secret_for_open_links=false, require_secret_for_access=false)"

# docker exec -u root -it jlml-raia /bin/bash
