FROM julia:1.8.2

ENV USER isetbz
ENV USER_HOME_DIR /home/${USER}
# ENV JULIA_DEPOT_PATH ${USER_HOME_DIR}/.julia
ENV WORKING_DIR ${USER_HOME_DIR}/repo

RUN useradd -m -d ${USER_HOME_DIR} ${USER} \
    && mkdir -p ${WORKING_DIR}

COPY ./toml/* ${WORKING_DIR}/

WORKDIR ${WORKING_DIR}

# Add Julia Packages
RUN julia -e 'import Pkg; Pkg.activate(); Pkg.instantiate();' && \
    julia -e 'using Pkg; pkg"add IJulia"; pkg"precompile"' && \
    chown -R ${USER} ${USER_HOME_DIR}

USER ${USER}


# Default command: 

## Julia Pluto
# CMD julia -e "import Pluto; Pluto.run(host=\"0.0.0.0\", port=1234, launch_browser=false, require_secret_for_open_links=false, require_secret_for_access=false)"
# CMD julia -e "using Pkg; Pkg.add(\"IJulia\"); using IJulia; notebook()"
## Jupyter Notebook
CMD julia 
# jupyter/base-notebook --host="0.0.0.0" --port=4321 --launch_browser=false --require_secret_for_open_links=false --require_secret_for_access=false
# Jupyter:
# Setting JUPYTER_TOKEN environment variable sets the token
# EXPOSE 8888/tcp
# CMD ["jupyter", "lab", "--ip", "0.0.0.0", "--port", "8888", "--launch_browser", "false", "require_secret_for_open_links", "false", "require_secret_for_access", "false"]

# docker exec -u root -it aijl /bin/bash
