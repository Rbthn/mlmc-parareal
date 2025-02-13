# TIND : 4-pole induction machine
#
# Robrecht De Weerdt, Eindige Elementen Modellering van Kooianker Inductiemotoren, PhD thesis, KU Leuven, 1995.
# IRM-motor, p. 108.
#
# Prof. Dr.-Ing. Herbert De Gersem
# Institute for Accelerator Science and Electromagnetic Fields (TEMF)
# Technische Universität Darmstadt
# www.temf.de

# %% dependencies
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
    const Irat = 154.0                    # rated current [A]
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



# %% prepare Problem
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
L = 2                               # use refinement levels 0, ..., L
mlmc_tol = 1e-2                     # desired tolerance on RMSE
warmup_samples = 3                  # number of samples initially evaluated
run_args = (;
    continuate=false,
    do_mse_splitting=true,
    min_splitting=0.01,
    warmup_samples=warmup_samples
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



# %% Benchmark
nruns = 10                      # number of runs over which to average savings
ncores = 100                    # number of parallel evaluations assumed
cost_benchmark_time = 10        # cost benchmark length

# fine cost ref, fine cost para, total cost ref, total cost para
timing = zeros(nruns, 4)

# determine cost of single eval per level (reference)
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

# determine cost on finest level (with parareal)
cost_para = @belapsed begin
    n_params = length($dists)
    params = transform.($dists, rand(n_params))
    prob = instantiate_problem($p, params)
    sol, _ = Parareal.solve(
        prob, $p.alg;
        dt=compute_timestep($p, $L),
        parareal_args...
    )
    qoi = total_energy(sol)
end seconds = cost_benchmark_time

for i = 1:nruns
    # run MultilevelEstimators once to determine number of samples per level
    e_ref = MLMC_Experiment(p, total_energy, dists,
        L, mlmc_tol;
        use_parareal=false,
        cost_model=(l -> costs[l[1]+1])
    )
    res_ref = run(
        e_ref;
        run_args...
    )

    nb_of_samples = res_ref["history"][:nb_of_samples]

    # MultilevelEstimators computes samples from levels sequentially.
    # As such, total time is the sum of time each level takes (plus convergence estimate overhead)
    # given the number of cores, the time it takes to compute the samples for a
    # given level equals the number of sequential runs due to insufficient cores
    # needed on that level times the average runtime on that level
    level_times = fill(Inf, L + 1)

    # integer divide, round up
    div_up = (x, y) -> ceil(Int, x / y)

    seq_runs = div_up.(nb_of_samples, ncores)

    total_cost_ref = sum(seq_runs .* costs)
    total_cost_para = sum(seq_runs[1:end-1] .* costs[1:end-1]) + div_up.(nb_of_samples[end] * parareal_args.parareal_intervals, ncores) * cost_para

    timing[i, :] = [costs[end], cost_para, total_cost_ref, total_cost_para]
end

# mean reduction in single eval
mean_reduction_single = mean(1 .- timing[:, 2] ./ timing[:, 1])
mean_reduction_overall = mean(1 .- timing[:, 4] ./ timing[:, 3])



# %% save settings, results
settings = (; ncores, nruns, cost_benchmark_time, L, mlmc_tol, warmup_samples, parareal_args, run_args)
results = (; costs, cost_para, timing, mean_reduction_single, mean_reduction_overall)

name = savename(p.name, settings, "jld2")
tagsave(datadir("benchmarks", name), struct2dict((; settings, results)))
