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

N = 10
L = 15
benchmark_time = 30

### Plot number of timesteps over iteration number k
plt = plot(xlabel=L"iteration $k$",
    xticks=1:N,
    ylabel="number of timesteps",
    yaxis=:log,
    ylim=(1e4, 3e6),
    minorgrid=:on,
    legend=:bottomright
    #title="total and sequential number of timesteps over Parareal iterations"
)
colors = palette(:viridis, 3)

println("Reference solution:")
sol_1 = @btime solve($p, [$L, $L], 0, use_parareal=false) seconds = benchmark_time
ts_parareal = zeros(2, N)
for k in range(1, N)

    # use zero tolerance to force iteration until i=k
    parareal_args = Parareal_Args(
        num_intervals=10, tolerance=0, max_iterations=k)
    println("====================================")
    println("Parareal, k=$k:")
    sol_2 = @btime solve($p, [$L, $L], 0, use_parareal=true, parareal_args=$parareal_args) seconds = benchmark_time

    ### compare number of timesteps (total, sequential)

    # seq. and total timesteps are equal for purely sequential solution
    ts_parareal[:, k] .= sol_2.stats.timesteps
end

hline!(plt, [sol_1.stats.naccept], label="reference solution", linestyle=:dash, color=:black)
scatter!(plt, ts_parareal[1, :], label="total", color=colors[1])
scatter!(plt, ts_parareal[2, :], label="sequential", color=colors[2])

wsave(plotsdir(savename(
        "parareal_timing",
        (problem=p.name, level=L, K=N),
        "pdf")), plt)

print("done")