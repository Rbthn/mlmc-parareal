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

"""
    solve(problem, level, ζ[, integrator][, use_parareal][, parareal_intervals][, parareal_tolerance][, kwargs...])

Solve a given `MLMC_Problem` with realization `ζ` and discretization level `level`. Return solution and (total timesteps, sequential timesteps)
"""
function solve(problem::MLMC_Problem, level, ζ; integrator=ImplicitEuler(),
    use_parareal=false,
    parareal_args::Union{Parareal_Args,Nothing}=nothing,
    kwargs...)

    l, L = level
    p::ODEProblem = instantiate_problem(problem, ζ)
    if !use_parareal || l != L
        # Don't use Parareal
        dt = compute_timestep(problem, l)
        sol = DifferentialEquations.solve(
            p,                  # problem
            integrator,         # timestepping algorithm
            dt=dt,              # timestep
            adaptive=false;     # disable adaptive timestepping to force dt
            kwargs...           # additional keyword-args for solver
        )
        timesteps = sol.stats.naccept
        return sol, [timesteps, timesteps]
    else
        dt_fine = compute_timestep(problem, l)
        dt_coarse = compute_timestep(problem, 0)

        # create num_intervals fine integrators to run in parallel
        fine_integrators = [
            init(
                p,                  # problem
                integrator,         # timestepping algorithm
                dt=dt_fine,         # timestep
                adaptive=false;     # disable adaptive timestepping to force dt
                kwargs...           # additional keyword-args for solver
            ) for _ in range(1, parareal_args.num_intervals)]

        int_coarse = init(
            p,                  # problem
            integrator,         # timestepping algorithm
            dt=dt_coarse,       # timestep
            adaptive=false;     # disable adaptive timestepping to force dt
            kwargs...           # additional keyword-args for solver
        )

        sol = solve_parareal(fine_integrators, int_coarse,
            problem.t_0, problem.t_end,
            problem.u_0,
            parareal_args
        )
        return sol, sol.stats.timesteps
    end
end

"""
    compute_timestep(problem, level)

Given an initial timestep `problem.Δt_0` at level 0,
compute the appropriate timestep for `level`.
"""
function compute_timestep(problem::MLMC_Problem, level)
    return problem.Δt_0 / 2^level
end


### QoI functions for ODESolutions

"""
    end_value(solution)

Simple QoI function for testing purposes.
Return the value at the final timestep in `solution`.
"""
function end_value(solution)
    return solution.u[end]
end

"""
    L2_squared(solution[, pointwise_norm2])

Compute the square of the L2 norm of the solution.
For multidimensional input, specify the pointwise (squared) norm function. Default: 2-norm.
"""
function L2_squared(solution, pointwise_norm2=(x) -> sum(abs2.(x)))
    pointwise_sq = pointwise_norm2.(solution.u)
    return integrate(solution.t, pointwise_sq)
end