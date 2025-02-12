## Perform MLMC for the simple Dahlquist test equation: y' = λy

# %% prepare problem
const u_0 = 1.0
const t_0 = 0.0
const t_end = 8.0
const λ = -1.0
const Δt_0 = 0.1 / 2^5

p = Dahlquist_Problem(u_0, t_0, t_end, λ, Δt_0)

const solver_args = (;
    adaptive=false,
    maxiters=typemax(Int),
    saveat=1e-3
)



# %% Parareal
const parareal_args = (;
    parareal_intervals=8,
    maxit=4,
    reltol=1e-4,
    abstol=1e-2,
    coarse_args=(;),
    fine_args=(;),
    shared_memory=false
)



# %% MLMC
const deviation = 0.5
const dist = Uniform(-deviation, deviation)
const L = 5
const mlmc_tol = 1e-2
const warmup_samples = 10
const qoi_fn = L2_squared
const benchmark_time = 100
run_args = (;
    continuate=false,
    do_mse_splitting=true,
    min_splitting=0.01,
    warmup_samples=warmup_samples,
)

# %% MLMC without Parareal
e_ref = MLMC_Experiment(p, qoi_fn, dist,
    L, mlmc_tol;
    use_parareal=false,
)
@time res_ref = run(
    e_ref;
    run_args...
)
bench_ref = @benchmark run(
    e_ref;
    run_args...
) seconds = benchmark_time



# %% MLMC with Parareal
e_para = MLMC_Experiment(p, qoi_fn, dist,
    L, mlmc_tol;
    seed=e_ref.seed,
    use_parareal=true,
    parareal_args=parareal_args
)
@time res_para = run(
    e_para;
    run_args...
)
bench_para = @benchmark run(
    e_para;
    run_args...
) seconds = benchmark_time



# %% save settings, results
settings = (; L, mlmc_tol, warmup_samples, parareal_args, seed=e_ref.seed, benchmark_time, runs_args)
results = (; costs, res_ref, res_para, bench_ref, bench_para)

name = savename(p.name, settings, "jld2")
tagsave(datadir("simulations", name), struct2dict((; settings, results)))
