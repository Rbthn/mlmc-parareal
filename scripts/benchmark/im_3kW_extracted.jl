# %% dependencies
@everywhere begin
    using DifferentialEquations
    using Parareal
    using SparseArrays
    using LinearAlgebra
    using BenchmarkTools
    using NumericalIntegration
end



# %% Parameters
@everywhere begin
    const project_root = dirname(Base.active_project())
    test_file_location = joinpath(project_root, "files/im_3kW")

    # currently, there is no clever way to share these parameters between
    # a Julia program and the GetDP .pro file. Hardcoding these values for now.
    const freq = 50             # frequency of the excitation
    const period = 1 / freq     # period length
    const nsteps = 100          # number of timesteps per period
    const nperiods = 8          # number of periods
    const dt = period / nsteps  # time step
    const t_0 = 0.0             # start time
    const t_end = period * nperiods # stop time
    const sigma_nom = 26.7e6    # nominal conductivity
    const num_bars = 32         # number of rotor bars
end



# %% load system matrices
@everywhere begin
    include(joinpath(test_file_location, "K.jl"))       # load K
    include(joinpath(test_file_location, "M_const.jl")) # load M_const
    include(joinpath(test_file_location, "M_diffs.jl")) # load M_diffs
    include(joinpath(test_file_location, "rhs_coef.jl"))# load rhs_coef

    const ndof = size(K)[2]
    # indices in sol.u that belong to vector potential
    const a_dofs = collect(39:ndof)

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
    saveat=dt,
)

@everywhere M_fct = (p) -> M_const + sum(M_diffs .* p)

p = FE_Problem(
    zeros(ndof),
    t_0,
    t_end,
    dt;
    alg=alg,
    M=M_fct,
    K=K,
    r=rhs_fct,
    ode_args...
)



# %% Parareal
parareal_args = (;
    parareal_intervals=8,
    maxit=3,
    reltol=1e-2,
    coarse_args=(; dt=dt),
    fine_args=(;),
    shared_memory=false,
)



# %% MLMC
L = 2                               # use refinement levels 0, ..., L
mlmc_tol = 1e-1                     # desired tolerance on RMSE
warmup_samples = 10                 # number of samples initially evaluated
run_args = (;
    continuate=false,
    do_mse_splitting=true,
    min_splitting=0.01,
    warmup_samples=warmup_samples
)

deviations = 0.05 * sigma_nom * ones(num_bars)
dists = Uniform.(sigma_nom .- deviations, sigma_nom .+ deviations)

@everywhere function qoi_fn(sol)
    # \dot a^T M_sigma \dot a
    M_sigma = M_fct(sol.prob.p) - M_const

    a_dot = [zeros(ndof) for _ in 1:length(sol.t)]
    for k in 1:length(sol.t)-1
        a_dot[k+1] = (sol.u[k+1] - sol.u[k]) / (sol.t[k+1] - sol.t[k])
    end
    loss = [a_dot[k][a_dofs]' * M_sigma[a_dofs, a_dofs] * a_dot[k][a_dofs] for k in 1:length(sol.t)]

    integrate(sol.t, loss, SimpsonEven()) / (sol.t[end] - sol.t[1])
end


# %% Benchmark
nruns = 10                      # number of runs over which to average
ncores = 100                    # numer of parallel evaluations assumed
cost_benchmark_time = 300       # cost benchmark length

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
    end seconds = cost_benchmark_time
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
end seconds = cost_benchmark_time


for i = 1:nruns
    # run MultilevelEstimators once to determine number of samples per level
    e_ref = MLMC_Experiment(p, qoi_fn, dists,
        L, mlmc_tol;
        use_parareal=false,
        cost_model=(l -> costs[l[1]+1])
    )
    res_ref = run(
        e_ref;
        run_args...
    )

    nb_of_samples = res_ref["history"][:nb_of_samples]

    # MultilevelEstimators computes samples from levels sequentially.
    # As such, total time is the sum of time each level takes (plus convergence estimate overhead)
    # given the number of cores, the time it takes to compute the samples for a
    # given level equals the number of sequential runs due to insufficient cores
    # needed on that level times the average runtime on that level
    level_times = fill(Inf, L + 1)

    seq_runs = div_up.(nb_of_samples, ncores)

    total_cost_ref = sum(seq_runs .* costs)
    total_cost_para = sum(seq_runs[1:end-1] .* costs[1:end-1]) + div_up.(nb_of_samples[end] * parareal_args.parareal_intervals, ncores) * cost_para

    timing[i, :] = [costs[end], cost_para, total_cost_ref, total_cost_para]
end

# mean reduction in single eval
mean_reduction_single = mean(1 .- timing[:, 2] ./ timing[:, 1])
mean_reduction_overall = mean(1 .- timing[:, 4] ./ timing[:, 3])



# %% save settings, results
settings = (;
    parareal_args,
    ncores, nruns, cost_benchmark_time,
    L, mlmc_tol, deviations, warmup_samples, run_args
)
results = (;
    costs, cost_para,
    timing, mean_reduction_single, mean_reduction_overall
)

name = savename(p.name, settings, "jld2")
tagsave(datadir("benchmarks", name), struct2dict((; settings, results)))
