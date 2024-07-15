using DrWatson
@quickactivate "MLMC_Parareal"

using DifferentialEquations
using LinearAlgebra
using StaticArrays
using Parameters
#include(srcdir("problem.jl"))

@with_kw struct Parareal_Args
    num_intervals::Int
    max_iterations::Int = num_intervals
    tolerance::Real = 1e-3
    jump_norm::Function = (x) -> maximum(norm.(x, 2)) # computes error from jump in u
end


"""
    solve_parareal(fine_integrator, coarse_integrator, t_0, t_end, u_0, num_intervals, jump_tol[, jump_norm])

Perform the Parareal algorithm using `fine_integrator` and `coarse_integrator`
for the fine and coarse iteration, respectively.

# Inputs:
- `fine_integrator`:      DEIntegrator with high resolution and high cost.
- `coarse_integrator`:    DEIntegrator with lower resolution and lower cost.
- `t_0`:                  Start time.
- `t_end`:                End time.
- `u_0`:                  Initial value at t=t_0
- `num_intervals`:        Number of intervals (=threads) to use for Parareal.
- `jump_tol`:             Tolerance on the discontinuity of the solution
                            between intervals.
- `jump_norm`:            Norm used to measure the discontinuities between
                            intervals. Default: max. over all synchronization
                            points, 2-norm at each synchronization point.

# Outputs:
named tuple with fields:
- `u`:                    Solution values.
- `t`:                    Timesteps.
- `info`:                 Addtional solver information.
"""
function solve_parareal(
    fine_integrator,
    coarse_integrator,
    t_0, t_end, u_0,
    args::Parareal_Args)

    ###
    ### Initialization
    ###
    @unpack num_intervals, max_iterations, tolerance, jump_norm = args

    # stats
    timesteps_total = 0 # total timesteps, proxy for power
    timesteps_seq = 0   # sequential timesteps, proxy for WCT with inf. #cores
    retcode = nothing
    k = 0               # iteration counter
    errors = fill(Inf, num_intervals) # Parareal errors after each iteration

    # set parareal sync points
    t = range(t_0, t_end, length=num_intervals + 1)

    # values of solution at sync points
    sync_values = SizedVector{num_intervals + 1,typeof(u_0)}(undef)
    sync_values[1] = u_0

    # jumps at sync points t_1 ... t_(end-1)
    sync_jumps = SizedVector{num_intervals,typeof(u_0)}(undef)

    # values of coarse solution at sync points from last iteration.
    # at index j, store G(t_j, t_j+1, u_j^(k-1))
    coarse_old = SizedVector{num_intervals,typeof(u_0)}(undef)

    # create num_intervals fine integrators to run in parallel
    fine_integrators = [deepcopy(fine_integrator) for _ in range(1, num_intervals)]

    # determine initial coarse solution
    for j in range(1, num_intervals)
        coarse_result = propagate!(
            coarse_integrator, t[j], t[j+1], sync_values[j])

        sync_values[j+1] = coarse_result
        coarse_old[j] = coarse_result

        # count timesteps
        steps = length(coarse_integrator.sol.t) - 1
        timesteps_total += steps
        timesteps_seq += steps
    end

    ###
    ### Main loop
    ###

    for outer k in range(1, num_intervals)
        if k > max_iterations
            retcode = :MaxIter
            k -= 1  # don't count current iteration
            break
        end

        # parallel
        Threads.@threads for j in range(k, num_intervals)
            end_value = propagate!(fine_integrators[j], t[j], t[j+1], sync_values[j])
            # ϵ = u^k_j - F(t_j, t_j+1, u^k_j-1)
            # sync value from last interation, fine solution from this iteration
            sync_jumps[j] = sync_values[j+1] - end_value
        end
        # count timesteps outside of parallel loop to avoid race condition
        steps = length(fine_integrators[1].sol.t) - 1
        timesteps_total += (num_intervals - k + 1) * steps
        timesteps_seq += steps

        # Due to Parareal exactness, the jumps at t_1...t_k-1 are zero.
        # There is no need to compute these jumps in the loop above, we can just set them to zero.
        for idx in range(1, k - 1)
            sync_jumps[idx] = zero(sync_jumps[idx])
        end

        # convergence, if jumps are below tolerance according to given norm
        errors[k] = jump_norm(sync_jumps)
        if errors[k] < tolerance
            retcode = :Success
            break
        end

        # sequential
        for j in range(k, num_intervals)
            coarse_result = propagate!(
                coarse_integrator, t[j], t[j+1], sync_values[j])

            sync_values[j+1] = coarse_result + fine_integrators[j].u - coarse_old[j]
            coarse_old[j] = coarse_result

            # count timesteps
            steps = length(coarse_integrator.sol.t) - 1
            timesteps_total += steps
            timesteps_seq += steps
        end
    end

    # We cannot return an ODESolution, since DifferentialEquations does not
    # allow constructing an ODESolution from given data.
    # To allow for uniform handling of the return type, we return a named tuple,
    # with the fields u, t, and additional information.
    # At the discontinuities, use the left solution.
    # TODO is this correct?
    solution_sizes = map(integrator -> length(integrator.sol.t) - 1,
        fine_integrators)
    total_size = sum(solution_sizes) + 1

    all_t = SizedVector{total_size,typeof(t_0)}(undef)
    all_t[1] = t_0
    all_u = SizedVector{total_size,typeof(u_0)}(undef)
    all_u[1] = u_0
    idx = 1

    for i in range(1, num_intervals)
        size = solution_sizes[i]
        int = fine_integrators[i]
        all_t[idx+1:idx+size] = int.sol.t[2:end]
        all_u[idx+1:idx+size] = int.sol.u[2:end]

        idx += size
    end

    return (u=all_u, t=all_t,
        stats=(
            timesteps=(timesteps_total, timesteps_seq),
            iterations=k,
            error=errors
        ),
        retcode=retcode
    )
end

"""
    propagate!(integrator, t_1, t_2, u_1)

Re-Initialize `integrator` to t=`t_1`, u=`u_1`
and step to t=`t_2`.

# Inputs:
- `integrator`:           DEIntegrator to use. Is modified by a call to this
                            function!
- `t_1`:                  Start time.
- `t_2`:                  End time.
- `u_1`:                  Start value.

# Outputs:
    Solution is contained in `integrator`. For convenience, this function
    also returns the computed value u at t=`t_2`
"""
function propagate!(integrator, t_1, t_2, u_1)
    reinit!(integrator, u_1, t0=t_1)
    step!(integrator, t_2 - t_1, true)
    return integrator.sol.u[end]
end