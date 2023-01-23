FROM julia:1.8.2
LABEL author="A. Mhamdi <a_mhamdi@outlook.com>"

# Expose ports
EXPOSE 1234 2468
# SetENV variables
ENV USERNAME mhamdi 
ENV USER_HOME_DIR /home/${USERNAME}
ENV JULIA_DEPOT_PATH ${USER_HOME_DIR}/.julia
ENV WORKING_DIR ${USER_HOME_DIR}/work
# Add user and set ownership
RUN useradd -ms /bin/bash  -p $(echo "${USERNAME}" | openssl passwd -1 -stdin) "${USERNAME}"
# Change user & working directory
USER ${USERNAME}
WORKDIR ${WORKING_DIR}
# Copy datasets
COPY ./Codes/Julia/Datasets ./
# Copy TOML files (deps, ver, pkg, UUID)
COPY ./Codes/Julia/Project.toml ./
COPY ./Codes/Julia/Manifest.toml ./

# Activate environment and add IJulia kernel
RUN julia -e "import Pkg; Pkg.activate(\".\"); Pkg.instantiate(); Pkg.precompile()"
RUN julia -e 'import Pkg; Pkg.add(["IJulia", "Pluto"]); Pkg.precompile()'

ENTRYPOINT julia -e "import Pluto; Pluto.run(host=\"0.0.0.0\", port=1234, launch_browser=false, require_secret_for_open_links=false, require_secret_for_access=false)"
# Default command
CMD julia
