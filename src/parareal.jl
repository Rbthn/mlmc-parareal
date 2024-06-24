using DrWatson
@quickactivate "MLMC_Parareal"

using DifferentialEquations
#include(srcdir("problem.jl"))

function solve_parareal(
    F::Function,    # fine propagator
    G::Function,    # coarse propagator
    t_0, t_end,
    u_0,
    num_intervals=8
)
    # initialize
    t = range(t_0, t_end, length=num_intervals + 1) # parareal timesteps
    # solution vector over timesteps
    u = Array{eltype(u_0)}(undef, (size(u_0)..., num_intervals + 1))
    # initial condition
    u[:, 1] = u_0

    # hold results of coarse propagator from previous iteration
    coarse_old = Array{eltype(u_0)}(undef, (size(u_0)..., num_intervals))
    # hold results of fine propagator from current iteration
    fine_results = Array{eltype(u_0)}(undef, (size(u_0)..., num_intervals))

    # initialize u with sequential coarse solutions
    for j in range(1, num_intervals)
        coarse_result = G(t[j], t[j+1], u[:, j])
        u[:, j+1] = coarse_result
        coarse_old[:, j] = coarse_result
    end

    # further iterations
    for k in range(1, num_intervals)
        # parallel
        num_parallel = num_intervals - k + 1
        for j in range(k, num_intervals)
            # TODO compute in parallel on num_parallel cores
            fine_results[:, j] = F(t[j], t[j+1], u[:, j])
        end
        # sequential
        for j in range(k, num_intervals)
            coarse_result = G(t[j], t[j+1], u[:, j])
            u[:, j+1] = coarse_result + fine_results[:, j] - coarse_old[:, j]
            coarse_old[:, j] = coarse_result
        end
        # TODO check convergence
    end

    return u
end

function propagator(problem::MLMC_Problem, ζ, t_1, t_2, u; dt)
    p = instantiate_problem(problem, ζ)
    int = init(p, dt=dt, save_everystep=false)
    set_u!(int, u)
    set_t!(int, t_1)
    step!(int, t_2 - t_1, true) # step until t+dt and ensure that timestep is included
    return int.u[:, end]
end