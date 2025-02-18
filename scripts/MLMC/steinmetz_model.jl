# TIND : 4-pole induction machine
#
# Robrecht De Weerdt, Eindige Elementen Modellering van Kooianker Inductiemotoren, PhD thesis, KU Leuven, 1995.
# IRM-motor, p. 108.
#
# Prof. Dr.-Ing. Herbert De Gersem
# Institute for Accelerator Science and Electromagnetic Fields (TEMF)
# Technische Universität Darmstadt
# www.temf.de

@everywhere begin
    using DifferentialEquations
    using Parareal
    using BenchmarkTools
    using NumericalIntegration
end


# %% Parameters
@everywhere begin
    const f = 50.0                        # applied frequency [Hz]
    const w = 2 * pi * f                  # applied angular frequency [rad/s]
    # A.1. Rated operation
    const Prat = 402e3                    # rated power (3 phases) [W]
    const Urat = 1716 / sqrt(3)           # rated voltage (equivalent phase, rms value) [V]
    const frat = 46.6                     # rated frequency [Hz]
    const Irat = 154                      # rated current [A]
    const cosphirat = 0.91                # rated power factor []
    # A.2. Data
    const pp = 2                          # pole-pair number [#]
    const wmrat = 2 * pi * frat / pp      # rated speed [rad/s]
    const Mzrat = Prat / wmrat            # rated torque [Nm]
    # A.3. Steinmetz model
    const nominal_params = (;
        pp=pp,                      # pole-pair number [#]
        J=8.4603 + 12.5014,         # moment of inertia of the rotor and drive train [kgm^2]
        R1=1.111140e-01,            # stator resistance [Ohm]
        Lh=6.407774e-02,            # main inductance [H]
        RFe=1.736354e+06,           # iron-loss resistance [Ohm]
        R2p=7.158602e-02,           # resistance of the squirrel case [Ohm]
        L1=1.649983e-03,            # leakage inductance of the stator windings[H]
        L2p=1.063014e-03,           # leakage inductance of the squirrel cage [H]
    )

    ## Excitation (line start-up)
    # applied voltage [V]
    ftu = (t) -> Urat * [cos(w * t); cos(w * t - 2 * pi / 3); cos(w * t + 2 * pi / 3)]
    # Line start-up with friction load [Nm]
    ftload = (t, wm) -> Mzrat * (wm / wmrat)
    const u_0 = zeros(10)
end



# %% prepare problem
t_0 = 0.0
t_end = 1.0
Δt_0 = 1e-4
ode_args = (;
    adaptive=false,
    maxiters=typemax(Int),
    saveat=1e-3
)

p = Steinmetz_Problem(
    u_0,
    t_0,
    t_end,
    Δt_0;
    nominal_params...,
    w,
    ftu,
    ftload,
    ode_args...
)



# %% Parareal
parareal_args = (;
    parareal_intervals=8,
    reltol=5e-2,
    abstol=10,
    maxit=4,
    coarse_args=(; dt=Δt_0),
    fine_args=(;),
    shared_memory=false
)



# %% MLMC
L = 2               # use refinement levels 0, ..., L
mlmc_tol = 1e-2
warmup_samples = 20
benchmark_time = 100
run_args = (;
    continuate=false,
    do_mse_splitting=true,
    min_splitting=0.01,
    warmup_samples=warmup_samples,
)

# noise
deviations = 0.05 * [
    nominal_params.R1
    nominal_params.Lh
    nominal_params.RFe
    nominal_params.R2p
    nominal_params.L1
    nominal_params.L2p
]
dists = Uniform.(-deviations, deviations)

# qoi function
@everywhere function total_energy(sol::ODESolution)
    i_1 = [e[5] for e in sol.u]
    i_2 = [e[6] for e in sol.u]
    i_3 = [e[7] for e in sol.u]

    v = ftu.(sol.t)
    v_1 = [e[1] for e in v]
    v_2 = [e[2] for e in v]
    v_3 = [e[3] for e in v]

    p_1 = i_1 .* v_1
    p_2 = i_2 .* v_2
    p_3 = i_3 .* v_3

    e_1 = integrate(sol.t, p_1, SimpsonEven())
    e_2 = integrate(sol.t, p_2, SimpsonEven())
    e_3 = integrate(sol.t, p_3, SimpsonEven())

    e = e_1 + e_2 + e_3

    return e / 1e6
end



# %% cost model
cost_benchmark_time = 10
costs = fill(Inf, L + 1)
for l = 0:L
    costs[l+1] = @belapsed begin
        n_params = length($dists)
        params = transform.($dists, rand(n_params))
        prob = instantiate_problem($p, params)
        sol = DifferentialEquations.solve(
            prob, $p.alg;
            dt=compute_timestep($p, $l)
        )
        qoi = total_energy(sol)
    end seconds = cost_benchmark_time
end



# %% MLMC without Parareal
e_ref = MLMC_Experiment(p, total_energy, dists,
    L, mlmc_tol;
    use_parareal=false,
    cost_model=(l -> costs[l[1]+1]),
)
@time res_ref = run(
    e_ref;
    run_args...,
)

bench_ref = @benchmark run(
    e_ref;
    run_args...,
    verbose=false,
) seconds = benchmark_time



# %% MLMC with Parareal
e_para = MLMC_Experiment(p, total_energy, dists,
    L, mlmc_tol;
    seed=e_ref.seed,
    use_parareal=true,
    parareal_args=parareal_args,
    cost_model=(l -> costs[l[1]+1]),
)
@time res_para = run(
    e_para;
    run_args...,
)

bench_para = @benchmark run(
    e_para;
    run_args...,
    verbose=false,
) seconds = benchmark_time



# %% save settings, results
settings = (;
    parareal_args, benchmark_time,
    L, mlmc_tol, deviations, warmup_samples, seed=e_ref.seed, run_args
)
results = (;
    costs,
    res_ref, bench_ref,
    res_para, bench_para
)

name = savename(p.name, settings, "jld2")
tagsave(datadir("simulations", name), struct2dict((; settings, results)))
