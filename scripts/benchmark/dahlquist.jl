# %% dependencies
@everywhere begin
    using DifferentialEquations
    using Parareal
    using BenchmarkTools
end


# %% prepare problem
const u_0 = 1.0
const t_0 = 0.0
const t_end = 1.0
const λ = -1.0
const Δt_0 = 0.1 / 2^5

p = Dahlquist_Problem(u_0, t_0, t_end, λ, Δt_0)
alg = ImplicitEuler()



# %% Parareal
const N = 10
const parareal_args = (;
    parareal_intervals=N,
    reltol=1e-4,
    abstol=1e-2,
    coarse_args=(;),
    fine_args=(;),
    shared_memory=false
)



# %% MLMC
const deviation = 0.5
const dists = Uniform(-deviation, deviation)
const L = 5
const mlmc_tol = 1e-4
const warmup_samples = 10
const qoi_fn = L2_squared
run_args = (;
    continuate=false,
    do_mse_splitting=true,
    min_splitting=0.01,
    warmup_samples=warmup_samples
)


# %% Benchmark
nruns = 10                      # number of runs over which to average savings
ncores = 100                    # number of parallel evaluations assumed
cost_benchmark_time = 10        # cost benchmark length

# determine cost of single eval per level (reference)
costs = fill(Inf, L + 1)
for l = 0:L
    costs[l+1] = @belapsed begin
        n_params = length($dists)
        params = transform.($dists, rand(n_params))
        prob = instantiate_problem($p, params)
        sol = DifferentialEquations.solve(
            prob, $alg;
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
        prob, $alg;
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
settings = (; ncores, nruns, cost_benchmark_time, L, mlmc_tol, warmup_samples, parareal_args, run_args)
results = (; costs, cost_para, timing, mean_reduction_single, mean_reduction_overall)

name = savename(p.name, settings, "jld2")
tagsave(datadir("benchmarks", name), struct2dict((; settings, results)))
