# Add Julia kernel to Jupyter Notebook and/or JupyterLab IDEs.
using Pkg
Pkg.add("IJulia")

# In case the kernel is not installed, another method is to use the following code.
using IJulia
installkernel("Julia")

