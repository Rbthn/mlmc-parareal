using DrWatson
@quickactivate "MLMC_Parareal"

using DifferentialEquations


### This type defines an MLMC Problem.
### Primary use: Objects of this type are passed to MLMC_Experiment,
### where solve(MLMC_Problem, level, ζ) is called.
###
### This file contains all the parts that can (hopefully) be reused.
### For the usage, see examples in src/models/.
### To create your own model:
###     - derive a type from MLMC_Problem
###     - if the default behavior of solve() and compute_timestep()
###         as defined below work for you, you only have to implement
###         instantiate_problem(::your_derived_type, ζ)
###     - alternatively, implement solve(::your_derived_type, level, ζ)
###         to have full control over how the solution is produced
###     - this file contains some QoI functions that work on the solution
###         of an ODEProblem. If you need another QoI function, implement it
###         for the return type of solve(::your_derived_type, level, ζ)
abstract type MLMC_Problem{T<:AbstractFloat,U<:AbstractFloat} end


### Default behavior:
###     - Compute timestep based on static information in the problem and current refinement level
###     - Instantiate an ODEProblem based on static information and current random sample
###     - solve resulting ODEProblem with explicit Euler and computed timestep
function solve(problem::MLMC_Problem, level, ζ; use_parareal=false, integrator=ImplicitEuler(), kwargs...)
    l, L = level
    if !use_parareal || l != L
        # Don't use Parareal
        dt = compute_timestep(problem, l)
        p::ODEProblem = instantiate_problem(problem, ζ)
        return DifferentialEquations.solve(
            p,                  # problem
            integrator,         # integrator
            dt=dt,              # timestep
            adaptive=false;     # disable adaptive timestepping to force dt
            kwargs...
        )
    else
        dt_fine = compute_timestep(problem, l)
        dt_coarse = compute_timestep(problem, l - 1)

        F = (t_1, t_2, u) -> propagator(problem, ζ, t_1, t_2, u, dt=dt_fine)
        G = (t_1, t_2, u) -> propagator(problem, ζ, t_1, t_2, u, dt=dt_coarse)

        u = solve_parareal(F, G, problem.t_0, problem.t_end, problem.u_0)
        return u
    end
end

### Default behavior:
###     - Start with initial timestep Δt_0 given in problem at level 0
###     - halve in each higher level
function compute_timestep(problem::MLMC_Problem, level)
    return problem.Δt_0 / 2^level
end


### QoI functions for ODESolutions

# End value
function end_value(solution::ODESolution)
    return solution[end]
end

# (squared) L2 norm
function L2_squared(solution::ODESolution, pointwise_norm2=(x) -> sum(x .^ 2))
    """
        Compute the square of the L2 norm of the solution.
        For multidimensional input, specify the pointwise (squared) norm function. Default: Euclidian norm
    """
    pointwise_sq = pointwise_norm2.(solution[:]) # squared norm in each timestep
    return integrate(solution.t, pointwise_sq)
end