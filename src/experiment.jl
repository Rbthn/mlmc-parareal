using Distributed
using Dates
using Random
using MultilevelEstimators
using StaticArrays

# required to overload run for MLMC_Experiment
import Base: run

struct MLMC_Experiment
    ### Related to discretization of the problem
    problem::MLMC_Problem
    qoi::Function
    ### Related to random sampling
    seed::UInt
    dist::Vector{AbstractDistribution}
    ### Related to MLMC
    L::Int              # Use discretization levels l=0...L
    ϵ::Float64          # RMSE tolerance
    ### Related to Parareal
    use_parareal::Bool
    parareal_args

    ### Additional arguments for estimator
    estimator_args::NamedTuple

    ### Distributions.jl allows us to sample random numbers
    ### according to a distribution, e.g. Uniform() or Normal().
    ### To make results reproducible*, we have to seed the PRNG
    ### with a known value. Since we cannot query
    ### the current seed, the idea is to seed the generator with a
    ### random, but known number.
    ###
    ### CAUTION: Currently, results are not reproducible.
    ### Initially, we generate the same random samples. But since the estimated
    ### cost of those samples is based on runtime,
    ### we cannot expect the same cost estimation. As such, the number of
    ### samples on each level (and therefore the results) will differ between
    ### runs. TODO: Fix by supplying a deterministic cost function.
    ###
    ### *: Due to updates of the underlying algorithms, the sequence of
    ###     random numbers generated from a given seed can change between
    ###     Julia (minor) version updates.
    ###     Also see https://docs.julialang.org/en/v1/stdlib/Random/#Reproducibility
    function MLMC_Experiment(
        problem::MLMC_Problem,
        qoi::Function,
        dist, L=2, ϵ=1e-3;
        seed=rand(UInt),
        use_parareal=false,
        parareal_args=(),
        kwargs...
    )
        # Seed PRNG
        Random.seed!(seed)

        if isa(dist, AbstractDistribution)
            dist = [dist]
        end

        return new(
            problem,
            qoi,
            seed, dist, L, ϵ,
            use_parareal,
            parareal_args,
            NamedTuple(kwargs))
    end
end

"""
    run(experiment; kwargs...)

Run the MLMC experiments with settings supplied at construction time.
Additional kwargs are passed to DifferentialEquations.solve
"""
function run(
    experiment::MLMC_Experiment;
    verbose=true,
    warmup_samples=20,
    continuate=true,
    do_mse_splitting=true,
    do_regression=true,
    min_splitting=0.5,
    max_splitting=0.99,
    worker_ids=workers(),
    kwargs...
)
    ############################################################################
    #########################   COLLECT SYSTEM INFO   ##########################
    ############################################################################
    info = Dict(
        :tstart => now(),
        :tstop => DateTime(0),
        :host => gethostname(),
        :mem_total_GiB => Sys.total_physical_memory() / 2^30,
        :mem_free_GiB => Sys.free_physical_memory() / 2^30,
        :cpu_name => Sys.CPU_NAME,
        :cpu_threads => Sys.CPU_THREADS
    )

    # not thread safe! TODO use atomics when splitting samples across cores
    # all timesteps to evaluate samples associated with level l. This includes solves for l and l-1
    total_timesteps = zeros(SizedVector{experiment.L + 1,Int})
    sequential_timesteps = zeros(SizedVector{experiment.L + 1,Int})


    ############################################################################
    ###########################   WORKER MAPPING   #############################
    ############################################################################

    # calculate ids for worker pools for parallel evaluation in MultilevelEstimators and Parareal

    all_sample_worker_ids = worker_ids
    all_parareal_worker_ids = (i) -> []

    if experiment.use_parareal
        avail_worker_count = length(worker_ids)
        n_intervals = experiment.parareal_args.parareal_intervals

        # workers()[sample_worker_idx] gives the worker IDs on which samples should be evaluated
        sample_worker_idx = range(
            1,
            length=div(avail_worker_count, n_intervals + 1),
            step=n_intervals + 1
        )
        sample_worker_ids = worker_ids[sample_worker_idx]

        if n_intervals == 1
            sample_worker_ids = all_sample_worker_ids
            parareal_worker_ids = all_parareal_worker_ids
        end

        # parareal_worker_idx(i) gives the worker ids onto which
        # the fine propagator should be delegated
        parareal_worker_ids = (i) -> worker_ids[
            [sample_worker_idx[(i-1)%length(sample_worker_idx)+1] + k for k = 1:n_intervals]
        ]
    else
        sample_worker_ids = all_sample_worker_ids
        parareal_worker_ids = all_parareal_worker_ids
    end


    worker_id_fct = (l) -> l[1] < experiment.L ? all_sample_worker_ids : sample_worker_ids
    parareal_id_fct = (l, sample_num) -> l < experiment.L ? all_parareal_worker_ids(sample_num) : parareal_worker_ids(sample_num)


    ############################################################################
    ###############################   RUN MLMC   ###############################
    ############################################################################

    """
        sample_function(level::MultilevelEstimators.Level, ζ)

    Compute QoI for given discretization level and next coarser level.
    This function is passed to MultilevelEstimators.

    # Inputs:
    - `ζ`:      Realization of the random variable.
    - `level`:  Discretization level.

    # Outputs:
    - `(ΔQ, Q)`, where `Q` is the QoI obtained from a solution at `level` and `ΔQ` is the difference to the QoI obtained from the solution at the previous level.
    """
    function sample_fn(level::MultilevelEstimators.Level, ζ, sample_num=-1)
        # MultilevelEstimators uses a multi-index. For MLMC, this index only has one entry.
        l = level[1]

        # solve given sample (defined by ζ) on the current level
        sol_current = solve(
            experiment.problem, experiment.problem.alg,
            (l, experiment.L), ζ, parareal_id_fct(l, sample_num),
            use_parareal=experiment.use_parareal,
            parareal_args=experiment.parareal_args; kwargs...)
        qoi_current = experiment.qoi(sol_current)

        if l == 0
            return qoi_current, qoi_current
        end
        # solve given sample one level lower
        # pass parareal arguments to solve.
        # Usually parareal should only be used for l==L,
        # but we'll let the Problem decide.
        sol_last = solve(
            experiment.problem, experiment.problem.alg,
            (l - 1, experiment.L), ζ, parareal_id_fct(l - 1, sample_num),
            use_parareal=experiment.use_parareal,
            parareal_args=experiment.parareal_args; kwargs...)
        qoi_last = experiment.qoi(sol_last)

        qoi_diff = qoi_current - qoi_last
        return qoi_diff, qoi_current
    end

    MLMC_estimator = MultilevelEstimators.Estimator(
        MultilevelEstimators.ML(),  # Multilevel index set
        MultilevelEstimators.MC(),  # Monte-Carlo sampling
        sample_fn,                  # (level, ζ) -> (ΔQ, Q)
        experiment.dist,
        save=false,
        worker_ids=worker_id_fct,   # (level) -> workers to distribute samples over
        ### force the use of all levels
        max_index_set_param=experiment.L,
        min_index_set_param=experiment.L,
        ### optimizations
        do_regression=do_regression,
        continuate=continuate,
        do_mse_splitting=do_mse_splitting,
        min_splitting=min_splitting,
        max_splitting=max_splitting,
        verbose=verbose,
        ### set number of warmup samples
        nb_of_warm_up_samples=warmup_samples;
        # additional options
        experiment.estimator_args...
    )

    ###
    ### actual run
    ###
    h = MultilevelEstimators.run(MLMC_estimator, experiment.ϵ)
    info[:tstop] = now()


    ############################################################################
    #############################   SAVE RESULTS   #############################
    ############################################################################
    # inputs / fields of experiment
    settings = Dict(fieldnames(MLMC_Experiment) .=> getfield.(Ref(experiment), fieldnames(MLMC_Experiment)))
    runtime_options = Dict(
        :continuate => continuate, :do_regression => do_regression, :do_mse_splitting => do_mse_splitting, :verbose => verbose
    )
    merge!(settings, runtime_options)

    # combine. Using strings here, as DrWatson does not like Symbols as keys.
    d = Dict(
        "settings" => settings,
        "info" => info,
        "history" => h[1],
    )

    return d
end
