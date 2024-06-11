using DrWatson
@quickactivate "MLMC_Parareal"

include(srcdir("experiment.jl"))
include(srcdir("models.jl"))

u_0 = 1.0
t_0 = 0.0
t_end = 1.0
λ = -1.0
Δt_0 = 0.1
p = dahlquist_problem(u_0, t_0, t_end, λ, Δt_0)

deviation = 0.5
L = 3
ϵ = 1e-3

e = MLMC_Experiment(p, MultilevelEstimators.Uniform(-deviation, deviation), L, ϵ)
run(e)