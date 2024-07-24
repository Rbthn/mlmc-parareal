# MLMC_Parareal

This code is intended to demonstrate a reduction of the time bottleneck
when performing a Multilevel Monte Carlo simulation in a parallel computing environment
using the Parareal algorithm.
It uses the [Julia Language](https://julialang.org/) and (among others), the package [MultilevelEstimators.jl](https://github.com/PieterjanRobbe/MultilevelEstimators.jl/)

### Getting started
- To clone this repository including its submodules, use

   `git clone --recurse-submodules <URL>`
- If you already cloned the repository, you can populate the submodules via

   `git submodule update --init --recursive`

### Examples
- Models are defined in [src/models/](src/models/)
- To reproduce numerical test cases, see [scripts/](scripts/)