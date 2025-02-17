using DifferentialEquations
using Parareal

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

    Solve a given `MLMC_Problem` with realization `ζ` and discretization level `level`. Return solution and (total timesteps, sequential timesteps).

    If `use_parareal` is `true`, the parareal algorithm is used to speed up
    the solution for samples on the finest level.
    Options for parareal can be passed via `parareal_args`. See `Parareal`
    documentation for a list of arguments.

    Additional solver arguments (used on all levels) can be passed as `kwargs`.
"""
function solve(problem::MLMC_Problem, alg, level, ζ, worker_ids=[];
    use_parareal=false,
    parareal_args=(;
        coarse_args=(;),
        fine_args=(;)),
    kwargs...
)
    # current, maximum
    l, L = level

    prob = instantiate_problem(problem, ζ)
    if !use_parareal || l != L
        # Don't use Parareal
        dt = compute_timestep(problem, l)
        sol = DifferentialEquations.solve(
            prob,               # problem
            alg;                # timestepping algorithm
            dt=dt,              # timestep
            kwargs...           # additional keyword-args for solver
        )
        timesteps = sol.stats.nsolve
        return sol, [timesteps, timesteps]
    else
        dt_fine = compute_timestep(problem, l)
        dt_coarse = compute_timestep(problem, 0)    # using very coarse timestep

        # set dt for coarse and fine arguments and remove from parareal args
        c_args = (; dt=dt_coarse, parareal_args.coarse_args...)
        f_args = (; dt=dt_fine, parareal_args.fine_args...)

        parareal_args = drop(parareal_args, :coarse_args)
        parareal_args = drop(parareal_args, :fine_args)

        sol, _ = Parareal.solve(
            prob,               # problem
            alg;                # timestepping algorithm
            worker_ids=worker_ids,
            coarse_args=c_args,
            fine_args=f_args,
            kwargs...,
            parareal_args...,   # prefer parareal_args in case of key conflict with kwargs
        )

        return sol, [sol.stats.nsolve, sol.stats.nsolve]
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
