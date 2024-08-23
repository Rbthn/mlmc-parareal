using DrWatson
@quickactivate :MLMC_Parareal

using BenchmarkTools
using Plots, LaTeXStrings
gr()

u_0 = 1.0
t_0 = 0.0
t_end = 1.0
λ = -1.0
Δt_0 = 0.1
p = Dahlquist_Problem(u_0, t_0, t_end, λ, Δt_0)
L = 5

# MLMC
deviation = 0.5
mlmc_tol = 1e-4
qoi_fn = L2_squared

reference = MLMC_Experiment(p, qoi_fn, Uniform(-deviation, deviation),
    L, mlmc_tol, use_parareal=false)

# parareal
N = 10
jump_tol = 1e-4
# TODO somehow decide on coarse propagator here?
parareal_args = Parareal_Args(
    num_intervals=N, tolerance=jump_tol, max_iterations=N)

parareal = MLMC_Experiment(p, qoi_fn, Uniform(-deviation, deviation),
    L, mlmc_tol, use_parareal=true, parareal_args=parareal_args)


# compare
result_ref = run(reference)
result_para = run(parareal)

power_ratio = result_para["timesteps"][1] / result_ref["timesteps"][1]
bottleneck_ratio = result_para["timesteps"][2] / result_ref["timesteps"][2]
power_increase_percent = (power_ratio - 1) * 100
bottleneck_speedup_percent = bottleneck_ratio * 100

println("Theoretical bottleneck time reduced to $(trunc(bottleneck_speedup_percent, digits=2))% with a power increase of $(trunc(power_increase_percent, digits=2))%")

print("done")