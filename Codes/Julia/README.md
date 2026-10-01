# Julia Code Examples

This repository contains code examples written in **Julia**, covering a range of artificial intelligence algorithms.

> [!NOTE]
> **Running the code**
>
> - **Recommended:** use the provided [Docker image](https://www.github.com/a-mhamdi/jlai/tree/main/Docker) for a consistent, reproducible environment.
> - **New to Julia?** Launch an interactive environment directly in your browser, with nothing to install:
>   [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/a-mhamdi/jlai/main?labpath=Codes%2FJulia)

## Why Julia?

Julia was designed to remove the "two-language problem": prototyping in a high-level language (Python, MATLAB), then rewriting the slow parts in C or C++. Its main strengths for AI and scientific computing are:

- **Speed:** Julia is compiled just-in-time (JIT) via LLVM, so well-written code often runs at speeds comparable to C and Fortran without leaving the language.
- **Readable, math-friendly syntax:** array operations, broadcasting (`.`), and Unicode symbols (`α`, `∇`, `∑`) let code look close to the equations in a paper.
- **Multiple dispatch:** functions specialize on the types of all their arguments, which makes code highly composable and extensible. Packages work together without being designed for each other.
- **One language end to end:** you can prototype and run production-grade numerical code in the same language, and read the source of the libraries you use, since most are written in Julia itself.
- **Native differentiable programming:** [Flux.jl](https://fluxml.ai/) and automatic differentiation tools work on ordinary Julia code, which makes it straightforward to implement algorithms from scratch.
- **Built-in parallelism:** multithreading, distributed computing, and GPU support _(e.g. CUDA.jl)_ are part of the ecosystem.
- **Interoperability:** you can call Python (PythonCall.jl), C, and R libraries when needed.

> [!WARNING]
> Julia isn't the best choice for everything:
>
> - **First-run latency:** the first call to a function includes compilation time ("time to first plot").
> - **Smaller ecosystem:** Python has far more libraries, tutorials, and community resources, especially for deep learning.
> - **Fewer production and deployment tools** compared with Python.
