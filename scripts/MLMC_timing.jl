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
seed = reference.seed

# parareal
N = 10
jump_tol = 1e-4
# TODO somehow decide on coarse propagator here?
parareal_args = Parareal_Args(
    num_intervals=N, tolerance=jump_tol, max_iterations=N)

parareal = MLMC_Experiment(p, qoi_fn, Uniform(-deviation, deviation),
    L, mlmc_tol, use_parareal=true, parareal_args=parareal_args, seed=seed)


# run both experiments
result_ref = run(reference, verbose=false, continuate=false)
result_para = run(parareal, verbose=false, continuate=false)

# compare results
mean_ref = result_ref["history"][:mean]
mean_para = result_para["history"][:mean]
rel_err = (mean_ref - mean_para) / mean_ref

# compare timing
power_para = result_para["timesteps"][1]
power_ref = result_ref["timesteps"][1]
time_para = result_para["timesteps"][2]
time_ref = result_ref["timesteps"][2]

power_ratio = power_para / power_ref
bottleneck_ratio = time_para / time_ref
power_increase_percent = (power_ratio - 1) * 100
bottleneck_speedup_percent = bottleneck_ratio * 100

# print info
println(
    "Theoretical bottleneck time reduced to $(trunc(bottleneck_speedup_percent, digits=2))%
    with a power increase of $(trunc(power_increase_percent, digits=2))%.
    Computed mean has a rel. err. of $(rel_err)."
)

# plot results
plt = plot()
ax_energy = twinx(plt)
xoffset = 0.3
leftx = 1
rightx = 3

# set colors
colors = palette(:viridis, 3)
color_ref = colors[2]
color_para = colors[1]

# plot times
bar!(plt, [leftx], [time_ref], label="Reference", xticks=([leftx - xoffset / 2, rightx - xoffset / 2], ["Time", "Energy"]), color=color_ref, legend=:topleft, ylabel="sequential linear system solves")
bar!(plt, [leftx - xoffset], [time_para], label="Parareal", color=color_para)

# plot energy
bar!(ax_energy, [rightx], [power_ref], label=nothing, color=color_ref, ylabel="total linear system solves")
bar!(ax_energy, [rightx - xoffset], [power_para], label=nothing, color=color_para)

# save plot and settings
name = savename(
    "MLMC_timing",
    (problem=p.name,))
settings = (; u_0, t_0, t_end, λ, Δt_0, L, deviation, mlmc_tol, qoi_fn, N, jump_tol, seed)
results = (; result_ref, result_para, mean_ref, mean_para, power_para, power_ref, time_para, time_ref)

wsave(plotsdir(name * ".pdf"), plt)
wsave(plotsdir(name * ".jld2"), Dict(
    "settings" => namedtuple_to_dict(settings),
    "results" => namedtuple_to_dict(results)))

print("done")