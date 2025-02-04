# %% Packages
@everywhere begin
    using DifferentialEquations
    using Parareal
    using SparseArrays
    using LinearAlgebra

    using BenchmarkTools
end



# %% Parameters
@everywhere begin
    const project_root = dirname(Base.active_project())
    test_file_location = joinpath(project_root, "files/im_3kW")

    # currently, there is no clever way to share these parameters between
    # a Julia program and the GetDP .pro file. Hardcoding these values for now.
    freq = 50
    period = 1 / freq
    nsteps = 100
    dt = period / nsteps
    t_end = period * 8
end



# %% load system matrices
@everywhere begin
    include(joinpath(test_file_location, "K.jl"))    # load K
    include(joinpath(test_file_location, "M.jl"))    # load M
    include(joinpath(test_file_location, "rhs_coef.jl"))  # load rhs_coef

    ndof = size(K)[2]

    function rhs_ansatz(t)
        [sin(2 * pi * freq * t), sin(2 * pi * freq * t - 2 * pi / 3)]
    end
    function rhs_fct(t)
        if t == 0
            return zeros(ndof)
        end
        rhs_coef * rhs_ansatz(t)
    end
end



# %% prepare Problem

alg = ImplicitEuler()

ode_args = (;
    dt=dt,
    adaptive=false,
    saveat=1e-3
)

p = FE_Problem(
    zeros(ndof),
    0.0,
    t_end,
    dt;
    alg=alg,
    M=M,
    K=K,
    r=rhs_fct,
    ode_args...
)



# %% MLMC
L = 2

deviations = 0.05 * [norm(K), norm(M)]
dists = Uniform.(-deviations, deviations)

qoi_fn = (sol) -> norm(sol.u[end])


# %% Parareal
parareal_args = (;
    parareal_intervals=8,
    maxit=3,
    reltol=1e-2,
    coarse_args=(; dt=10 * dt),
    fine_args=(; dt=dt),
    shared_memory=false,
)
