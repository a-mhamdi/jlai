FROM julia:1.8.2
LABEL author="A. Mhamdi <abmhamdi@hotmail.com>"

# Expose ports
EXPOSE 1234 2468
# Set ENV variables
ENV USERNAME mhamdi 
ENV USER_HOME_DIR /home/${USERNAME}
ENV JULIA_DEPOT_PATH ${USER_HOME_DIR}/.julia
ENV JL ${JULIA_DEPOT_PATH}/conda/3/bin/jupyter\ lab
ENV WORKING_DIR ${USER_HOME_DIR}/work
# Add user and set ownership
RUN useradd -ms /bin/bash  -p $(echo ${USERNAME} | openssl passwd -1 -stdin) ${USERNAME}
# Change user & working directory
USER ${USERNAME}
WORKDIR ${WORKING_DIR}

RUN mkdir -p ${USER_HOME_DIR}/.jupyter/lab/user-settings/@jupyterlab/apputils-extension & \
    echo '{"theme": "JupyterLab Dark"}' > themes.jupyterlab-settings

# Copy datasets
COPY ./Codes/Julia/Datasets ./
# Copy TOML files (deps, ver, pkg, UUID)
COPY ./Codes/Julia/Project.toml ./
COPY ./Codes/Julia/Manifest.toml ./

# Activate environment and add IJulia kernel, Pluto and JupyterLab IDE
# RUN julia -e 'import Pkg; Pkg.activate("."); Pkg.instantiate(); Pkg.precompile()'
RUN julia -e 'import Pkg; Pkg.add(["Conda", "IJulia", "Pluto"]); Pkg.precompile()'
RUN julia -e 'import Conda; Conda.add("JupyterLab")'

# Default command
CMD julia
