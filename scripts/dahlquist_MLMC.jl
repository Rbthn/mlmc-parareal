## Perform MLMC for the simple Dahlquist test equation: y' = λy

# problem
const u_0 = 1.0
const t_0 = 0.0
const t_end = 8.0
const λ = -1.0
const Δt_0 = 0.1 / 2^5
const p = Dahlquist_Problem(u_0, t_0, t_end, λ, Δt_0)

# parareal
const parareal_args = (;
    parareal_intervals=8,
    reltol=1e-4,
    abstol=1e-2,
    coarse_args=(;),
    fine_args=(;)
)

# additional solver args
const solver_args = (;
    adaptive=false,
    maxiters=typemax(Int),
)

# MLMC
const deviation = 0.5
const dist = Uniform(-deviation, deviation)
const L = 10
const ϵ = 1e-3
const qoi_fn = L2_squared

e = MLMC_Experiment(
    p, qoi_fn, dist,
    L, ϵ,
    use_parareal=true,
    parareal_args=parareal_args,
)
result = run(e, continuate=false; solver_args...)

#save with git commit hash (and patch if repo is dirty)
problem_name = "Dahlquist"
name = savename(problem_name, result["settings"], "jld2")
tagsave(datadir("simulations", name), result, storepatch=true)

print("done")
