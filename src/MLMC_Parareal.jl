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

include("models/pendulum.jl")
export Pendulum_Problem, instantiate_problem

include("models/heat_fem.jl")
export Heat_Problem, instantiate_problem

# utilities
include("utilities.jl")
export namedtuple_to_dict

end
