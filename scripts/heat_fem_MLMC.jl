using DrWatson
@quickactivate :MLMC_Parareal

# problem
n = 11
x = range(0.0, 1.0, n)
u_left = 0.5
u_right = 1.0
u_0 = [u_left; repeat([0.0], n - 2); u_right]

cv = [1.0 for _ in range(1, n - 1)]
k = [1.0 for _ in range(1, n - 1)]
Q = [0.0 for _ in range(1, n - 1)]

t_0 = 0.0
t_end = 1.0
Δt_0 = 0.1

p = Heat_Problem(u_0, u_left, u_right, t_0, t_end, Δt_0, x, cv, k, Q)

# parareal
parareal_args = (;
    parareal_intervals=4,
    reltol=1e-4,
    abstol=1e-2,
    coarse_args=(;),
    fine_args=(;)
)

# MLMC
deviation = 0.5
dist = Uniform(1 - deviation, 1 + deviation)
L = 4
ϵ = 1e-3
qoi_fn = L2_squared

e = MLMC_Experiment(p, qoi_fn, dist,
    L, ϵ, use_parareal=true, parareal_args=parareal_args)
result = run(e)

#save with git commit hash (and patch if repo is dirty)
problem_name = "HeatFEM"
name = savename(problem_name, result["settings"], "jld2")
tagsave(datadir("simulations", name), result, storepatch=true)

print("done")
