using DrWatson
@quickactivate :MLMC_Parareal

using BenchmarkTools
using Plots, LaTeXStrings
gr()

"""
Computes estimates for computational effort of the Parareal method.
# Inputs:
- `work_coarse`: computational effort to solve for one interval using the coarse propagator
- `work_fine`: computational effort to solve for one interval using the fine propagator
- `num_intervals`: number of intervals used for Parareal
- `num_iterations`: number of iterations after which Parareal terminates

# Outputs:
- `work`: estimated total work on all cores, proxy for energy use
- `time`: estimated work that has to be performed sequentially, proxy for computation time
"""
function work_estimate(work_coarse, work_fine, num_intervals, num_iterations)
    aggregate_intervals(k) = k * (num_intervals + 1 - (k + 1) / 2)
    work = aggregate_intervals(num_iterations) * work_fine +
           (N + aggregate_intervals(num_iterations - 1)) * work_coarse
    time = num_iterations * work_fine +
           (N + aggregate_intervals(num_iterations - 1)) * work_coarse
    return work, time
end


u_0 = 1.0
t_0 = 0.0
t_end = 1.0
λ = -1.0
Δt_0 = 0.1
p = Dahlquist_Problem(u_0, t_0, t_end, λ, Δt_0)

N = 10
L = 15
benchmark_time = 30

sol_1, steps_1 = solve(p, [L, L], 0, use_parareal=false)
bench_1 = @benchmark solve($p, [$L, $L], 0, use_parareal=false) seconds = benchmark_time
reference_ts = steps_1[1]
reference_bench = minimum(bench_1.times)

ts_parareal = zeros(2, N)
actual_parareal = zeros(1, N)
ts_theoretical = zeros(2, N)

for k in range(1, N)

    # use zero tolerance to force iteration until i=k
    parareal_args = Parareal_Args(
        num_intervals=10, tolerance=0, max_iterations=k)
    sol_2, steps_2 = solve(p, [L, L], 0, use_parareal=true, parareal_args=parareal_args)
    bench_2 = @benchmark solve($p, [$L, $L], 0, use_parareal=true, parareal_args=$parareal_args) seconds = benchmark_time

    ### compare number of timesteps (total, sequential)

    # seq. and total timesteps are equal for purely sequential solution
    ts_parareal[:, k] .= steps_2
    actual_parareal[k] = minimum(bench_2.times)

    work_coarse = (t_end - t_0) / Δt_0 / N # number of timesteps per interval for coarse solver
    ts_theoretical[:, k] .= work_estimate(work_coarse, 2^L * work_coarse, N, k)
end

# compute ratio of estimated timesteps vs. counted timesteps. Expected to be slightly below 1 due to imperfect estimate
estimate_quality = ts_theoretical ./ ts_parareal
# compute factor by which actual runtime is above expectation. If the implementation is as efficient as the reference, this should be 1 for all timesteps
runtime_multiple = (reference_ts / reference_bench) * actual_parareal[:] ./ ts_parareal[2, :]

plt = plot(xlabel=L"iteration $k$",
    xticks=1:N,
    ylabel="number of timesteps",
    ylim=(0, 2e6),
    minorgrid=:on,
    legend=:topleft
    #title="total and sequential number of timesteps over Parareal iterations"
)
colors = palette(:viridis, 3)

# plot timesteps
hline!(plt, [reference_ts], label="reference solution", linestyle=:dash, color=:black)
scatter!(plt, ts_parareal[1, :], label="total timesteps", color=colors[1])
scatter!(plt, ts_parareal[2, :], label="sequential timesteps", color=colors[2])

# scale second yaxis
ax_actual = twinx(plt)
scale = 1e-6 # ns to ms
limits = ylims(plt)
ratio = (reference_ts - limits[1]) / (limits[2] - limits[1])
actual_min = 0
actual_max = 1 / ratio * (scale * reference_bench + (ratio - 1) * actual_min)
ylims!(ax_actual, actual_min, actual_max)

# plot actual runtimes
ylabel!(ax_actual, "runtime [ms]")
hline!(ax_actual, [scale * reference_bench], label=nothing, linestyle=:dash, color=:black)
scatter!(ax_actual, scale * actual_parareal[:], label=nothing, color=colors[3])
scatter!(plt, fill(NaN, 1:N), label="measured runtime", color=colors[3])

# save plot and settings
name = savename("parareal_timing",
    (problem=p.name,))
settings = (; u_0, t_0, t_end, λ, Δt_0, N, L, benchmark_time)
results = (; actual_parareal, reference_bench, reference_ts, ts_parareal, ts_theoretical, estimate_quality, runtime_multiple)

wsave(plotsdir(name * ".pdf"), plt)
wsave(plotsdir(name * ".jld2"), Dict(
    "settings" => namedtuple_to_dict(settings),
    "results" => namedtuple_to_dict(results)))


print("done")