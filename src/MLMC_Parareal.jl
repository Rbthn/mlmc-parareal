module MLMC_Parareal

# re-export relevant things from MultilevelEstimators
using Reexport
@reexport using MultilevelEstimators

include("problem.jl")
export MLMC_Problem, solve, end_value, L2_squared

include("experiment.jl")
export MLMC_Experiment, run

# models
include("models/dahlquist.jl")
export Dahlquist_Problem, instantiate_problem

include("models/FE.jl")
export FE_Problem, instantiate_problem

include("models/pendulum.jl")
export Pendulum_Problem, instantiate_problem

include("models/steinmetz.jl")
export Steinmetz_Problem, instantiate_problem, compute_timestep

# utilities
include("utilities.jl")

end
