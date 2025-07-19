# %% dependencies
@everywhere begin
    using DifferentialEquations
    using Parareal
    using SparseArrays
    using LinearAlgebra
    using BenchmarkTools
    using NumericalIntegration
end



# %% Parameters
@everywhere begin
    const project_root = dirname(Base.active_project())
    test_file_location = joinpath(project_root, "files/im_3kW")

    # currently, there is no clever way to share these parameters between
    # a Julia program and the GetDP .pro file. Hardcoding these values for now.
    const freq = 50             # frequency of the excitation
    const period = 1 / freq     # period length
    const nsteps = 10           # number of timesteps per period
    const nperiods = 8          # number of periods
    const dt = period / nsteps  # time step
    const t_0 = 0.0             # start time
    const t_end = period * nperiods # stop time
    const sigma_nom = 26.7e6    # nominal conductivity
    const num_bars = 32         # number of rotor bars
end



# %% load system matrices
@everywhere begin
    include(joinpath(test_file_location, "K.jl"))       # load K
    include(joinpath(test_file_location, "M_const.jl")) # load M_const
    include(joinpath(test_file_location, "M_diffs.jl")) # load M_diffs
    include(joinpath(test_file_location, "rhs_coef.jl"))# load rhs_coef

    const ndof = size(K)[2]
    # indices in sol.u that belong to vector potential
    const a_dofs = collect(39:ndof)

    function rhs_ansatz(t)
        [sin(2 * pi * freq * t), sin(2 * pi * freq * t - 2 * pi / 3)]
    end
    function rhs_fct(t)
        if t == 0
            return zeros(ndof)
        end
        rhs_coef * rhs_ansatz(t)
    end
end



# %% prepare Problem
@everywhere begin
    alg = ImplicitEuler()

    ode_args = (;
        adaptive=false,
        saveat=dt / 100,
    )

    M_fct = (p) -> getproperty(Main, :M_const) + sum(getproperty(Main, :M_diffs) .* p)

    p = FE_Problem(
        zeros(ndof),
        t_0,
        t_end,
        dt;
        alg=alg,
        M=M_fct,
        K=:K,
        r=rhs_fct,
        ode_args...
    )

    function qoi_fn(sol)
        # compute QoI: \int \dot a^T M_\sigma \dot a dt
        M_sigma = sum(M_diffs .* sol.prob.p)
        a_dot = [zeros(ndof) for _ in 1:length(sol.t)]
        for k in 1:length(sol.t)-1
            a_dot[k+1] = (sol.u[k+1] - sol.u[k]) / (sol.t[k+1] - sol.t[k])
        end
        loss = [a_dot[k][a_dofs]' * M_sigma[a_dofs, a_dofs] * a_dot[k][a_dofs] for k in 1:length(sol.t)]
        qoi = integrate(sol.t, loss, SimpsonEven()) / (sol.t[end] - sol.t[1]) / 1000

        return qoi
    end
end

# %% Parareal
parareal_args = (;
    parareal_intervals=16,
    reltol=1e-3,
    coarse_args=(; dt=dt),
    fine_args=(;),
    shared_memory=false,
)

# %% MLMC
L = 2                               # use refinement levels 0, ..., L
mlmc_tol = 3e-3                     # desired tolerance on RMSE
warmup_samples = 10                 # number of samples initially evaluated
benchmark_time = 1000
run_args = (;
    continuate=false,
    do_mse_splitting=true,
    min_splitting=0.01,
    warmup_samples=warmup_samples
)

deviations = 0.05 * sigma_nom * ones(num_bars)
dists = Uniform.(sigma_nom .- deviations, sigma_nom .+ deviations)


# %% cost model
cost_benchmark_time = 300       # cost benchmark length
# determine cost of single eval per level
costs = fill(Inf, L + 1)
effort = fill(Inf, L + 1)
for l = 0:L
    costs[l+1] = @belapsed begin
        n_params = length($dists)
        params = transform.($dists, rand(n_params))

        sol = MLMC_Parareal.solve(p, p.alg, ($l, $L), params; use_parareal=false)
        qoi = qoi_fn(sol)
        effort[$l+1] = sol.stats.nsolve
    end seconds = cost_benchmark_time
end

# determine cost on finest level (with parareal)
effort_para_single = [Inf]
cost_para = @belapsed begin
    n_params = length($dists)
    params = transform.($dists, rand(n_params))

    sol = MLMC_Parareal.solve(p, p.alg, ($L, $L), params; use_parareal=true, parareal_args=$parareal_args)
    qoi = qoi_fn(sol)
    effort_para_single[1] = sol.stats.nsolve
end seconds = cost_benchmark_time


# %% MLMC without Parareal
e_ref = MLMC_Experiment(p, qoi_fn, dists,
    L, mlmc_tol;
    use_parareal=false,
    cost_model=(l -> costs[l[1]+1]),
)
time_ref = @elapsed res_ref = run(
    e_ref;
    run_args...,
)
effort_ref = sum(res_ref["history"][:nb_of_samples] .* effort)

# %% MLMC with Parareal
e_para = MLMC_Experiment(p, qoi_fn, dists,
    L, mlmc_tol;
    seed=e_ref.seed,
    use_parareal=true,
    parareal_args=parareal_args,
    cost_model=(l -> costs[l[1]+1]),
)
time_para = @elapsed res_para = run(
    e_para;
    run_args...,
)
effort_para = sum(res_para["history"][:nb_of_samples][1:end-1] .* effort[1:end-1]) + res_para["history"][:nb_of_samples][end] * effort_para_single[1]


# %% save settings, results
settings = (;
    freq, period,
    nsteps, nperiods,
    dt, t_0, t_end,
    sigma_nom, num_bars,
    ndof, a_dofs,
    alg, ode_args,
    parareal_args, benchmark_time, N_PROCS,
    L, mlmc_tol, deviations, warmup_samples, seed=e_ref.seed, run_args
)
results = (;
    costs, cost_para,
    effort, effort_para_single,
    res_ref, time_ref,
    res_para, time_para,
    effort_ref, effort_para,
)

name = savename(p.name, settings, "jld2")
tagsave(datadir("simulations", name), struct2dict((; settings, results)))
