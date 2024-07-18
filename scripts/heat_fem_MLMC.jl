using DrWatson
@quickactivate :MLMC_Parareal

n = 11
x = range(0.0, 1.0, n)
u_0 = [0.0 for _ in range(1, n - 2)]
u_left = 0.5
u_right = 1.0

cv = [1.0 for _ in range(1, n - 1)]
k = [1.0 for _ in range(1, n - 1)]
Q = [0.0 for _ in range(1, n - 1)]

t_0 = 0.0
t_end = 1.0
Δt_0 = 0.1

use_parareal = false
p = Heat_Problem(u_0, u_left, u_right, t_0, t_end, Δt_0, x, cv, k, Q)

deviation = 0.5
L = 4
ϵ = 1e-3
qoi_fn = L2_squared

e = MLMC_Experiment(p, qoi_fn, Uniform(-deviation, deviation), L, ϵ; use_parareal)
result = run(e)

print("done")