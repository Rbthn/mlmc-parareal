using DrWatson
@quickactivate "MLMC_Parareal"

include(srcdir("experiment.jl"))
include(srcdir("problem.jl"))
include(srcdir("models/dahlquist.jl"))

u_0 = 1.0
t_0 = 0.0
t_end = 1.0
λ = -1.0
Δt_0 = 0.1
use_parareal = false
p = Dahlquist_Problem(u_0, t_0, t_end, λ, Δt_0, use_parareal=use_parareal)

deviation = 0.5
L = 4
ϵ = 1e-3
qoi_fn = L2_squared

e = MLMC_Experiment(p, qoi_fn, Uniform(-deviation, deviation), L, ϵ)
result = run(e)