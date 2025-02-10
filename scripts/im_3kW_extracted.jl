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
    nsteps = 10
    dt = period / nsteps
    t_end = period * 8
end



# %% load system matrices
@everywhere begin
    include(joinpath(test_file_location, "K.jl"))    # load K
    include(joinpath(test_file_location, "K_nu.jl")) # load K_nu
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
    dK=K_nu,
    r=rhs_fct,
    ode_args...
)



# %% MLMC
L = 2
mlmc_tol = 1e-1
warmup_samples = 10

deviations = 0.05 * [1]
dists = Uniform.(-deviations, deviations)

@everywhere function qoi_fn(sol)
    integrate(sol.t, [abs(e[8]) for e in sol.u], SimpsonEven())
end

# %% Parareal
parareal_args = (;
    parareal_intervals=8,
    maxit=3,
    reltol=1e-2,
    coarse_args=(; dt=dt),
    fine_args=(;),
    shared_memory=false,
)



# %% Benchmark

nruns = 10      # number of runs over which to average
ncores = 100    # numer of parallel evaluations assumed for benchmark

# fine cost ref, fine cost para, total cost ref, total cost para
timing = zeros(nruns, 4)

# determine cost of single eval per level
costs = fill(Inf, L + 1)
for l = 0:L
    costs[l+1] = @belapsed begin
        n_params = length($dists)
        params = transform.($dists, rand(n_params))
        prob = instantiate_problem($p, params)
        sol = DifferentialEquations.solve(
            prob, $p.alg;
            dt=compute_timestep($p, $l)
        )
        qoi = qoi_fn(sol)
    end seconds = 300
end

# determine cost on finest level (with parareal)
cost_para = @belapsed begin
    n_params = length($dists)
    params = transform.($dists, rand(n_params))
    prob = instantiate_problem($p, params)
    sol, _ = Parareal.solve(
        prob, $p.alg;
        dt=compute_timestep($p, $L),
        parareal_args...
    )
    qoi = qoi_fn(sol)
end seconds = 300


for i = 1:nruns
    # run MultilevelEstimators once to determine number of samples per level
    e_ref = MLMC_Experiment(p, qoi_fn, dists,
        L, mlmc_tol;
        use_parareal=false,
        cost_model=(l -> costs[l[1]+1])
    )
    res_ref = run(
        e_ref;
        continuate=false,
        do_mse_splitting=true,
        min_splitting=0.01,
        warmup_samples=warmup_samples
    )

    nb_of_samples = res_ref["history"][:nb_of_samples]

    # MultilevelEstimators computes samples from levels sequentially.
    # As such, total time is the sum of time each level takes (plus convergence estimate overhead)
    # given the number of cores, the time it takes to compute the samples for a
    # given level equals the number of sequential runs due to insufficient cores
    # needed on that level times the average runtime on that level
    level_times = fill(Inf, L + 1)

    # integer divide, round up
    div_up = (x, y) -> ceil(Int, x / y)

    seq_runs = div_up.(nb_of_samples, ncores)

    total_cost_ref = sum(seq_runs .* costs)
    total_cost_para = sum(seq_runs[1:end-1] .* costs[1:end-1]) + div_up.(nb_of_samples[end] * parareal_args.parareal_intervals, ncores) * cost_para

    timing[i, :] = [costs[end], cost_para, total_cost_ref, total_cost_para]
end

# mean reduction in single eval
mean_reduction_single = mean(1 .- timing[:, 2] ./ timing[:, 1])
mean_reduction_overall = mean(1 .- timing[:, 4] ./ timing[:, 3])
