using DrWatson
@quickactivate :MLMC_Parareal


u_0 = 1.0
t_0 = 0.0
t_end = 1.0
λ = -1.0
Δt_0 = 0.1
use_parareal = false
p = Dahlquist_Problem(u_0, t_0, t_end, λ, Δt_0)

deviation = 0.5
L = 4
ϵ = 1e-3
qoi_fn = L2_squared

e = MLMC_Experiment(p, qoi_fn, Uniform(-deviation, deviation), L, ϵ; use_parareal)
result = run(e)