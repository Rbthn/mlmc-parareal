using DrWatson
@quickactivate "MLMC_Parareal"

using DifferentialEquations
using LinearAlgebra
#include(srcdir("problem.jl"))

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
TODO
"""
function solve_parareal(
    fine_integrator,
    coarse_integrator,
    t_0, t_end,
    u_0,
    num_intervals,
    jump_tol,
    jump_norm=(x) -> maximum(norm.(x, 2))
)
    ###
    ### Initialization
    ###

    # set parareal sync points
    t = range(t_0, t_end, length=num_intervals + 1)

    # values of solution at sync points
    sync_values = Vector{typeof(u_0)}(undef, num_intervals + 1)
    sync_values[1] = u_0

    # jumps at sync points t_1 ... t_(end-1)
    sync_jumps = Vector{typeof(u_0)}(undef, num_intervals)

    # values of coarse solution at sync points from last iteration.
    # at index j, store G(t_j, t_j+1, u_j^(k-1))
    coarse_old = Vector{typeof(u_0)}(undef, num_intervals)

    # create num_intervals fine integrators to run in parallel
    fine_integrators = [deepcopy(fine_integrator) for _ in range(1, num_intervals)]

    # solver message
    message = ""

    # determine initial coarse solution
    for j in range(1, num_intervals)
        coarse_result = propagate!(
            coarse_integrator, t[j], t[j+1], sync_values[j])

        sync_values[j+1] = coarse_result
        coarse_old[j] = coarse_result
    end

    ###
    ### Main loop
    ###

    for k in range(1, num_intervals)
        # parallel
        Threads.@threads for j in range(k, num_intervals)
            end_value = propagate!(fine_integrators[j], t[j], t[j+1], sync_values[j])
            # ϵ = u^k_j - F(t_j, t_j+1, u^k_j-1)
            # sync value from last interation, fine solution from this iteration
            sync_jumps[j] = sync_values[j+1] - end_value
        end
        # Due to Parareal exactness, the jumps at t_1...t_k-1 are zero.
        # There is no need to compute these jumps in the loop above, we can just set them to zero.
        map(idx -> fill!(sync_jumps[idx], 0), 1:k-1)

        # convergence, if jumps are below tolerance according to given norm
        sync_jumps_norm = jump_norm(sync_jumps)
        if sync_jumps_norm < jump_tol
            message = "Tolerance reached after iteration k=$k"
            break
        end

        # sequential
        for j in range(k, num_intervals)
            coarse_result = propagate!(
                coarse_integrator, t[j], t[j+1], sync_values[j])

            sync_values[j+1] = coarse_result + fine_integrators[j].u - coarse_old[j]
            coarse_old[j] = coarse_result
        end

    end

    # TODO stitch together ODESolution.
    # What to do at sync_points? Mean of left and right value? Keep both?
    all_t = reduce(vcat, [int.sol.t for int in fine_integrators])
    all_u = reduce(vcat, [int.sol.u for int in fine_integrators])
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
    return copy(integrator.u)
end