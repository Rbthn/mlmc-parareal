using DrWatson
@quickactivate :MLMC_Parareal

# problem
u_0 = 1.0
t_0 = 0.0
t_end = 8.0
λ = -1.0
Δt_0 = 0.1
p = Dahlquist_Problem(u_0, t_0, t_end, λ, Δt_0)

# parareal
parareal_args = Parareal_Args(num_intervals=8, tolerance=1e-4)

# MLMC
deviation = 0.5
L = 4
ϵ = 1e-3
qoi_fn = L2_squared

e = MLMC_Experiment(p, qoi_fn, Uniform(-deviation, deviation),
    L, ϵ, use_parareal=true, parareal_args=parareal_args)
result = run(e)

#save with git commit hash (and patch if repo is dirty)
problem_name = "Dahlquist"
name = savename(problem_name, result["settings"], "jld2")
tagsave(datadir("simulations", name), result, storepatch=true)

print("done")