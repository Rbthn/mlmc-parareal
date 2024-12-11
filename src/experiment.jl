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
    dist::AbstractDistribution
    ### Related to MLMC
    L::Int              # Use discretization levels l=0...L
    ϵ::Float64          # RMSE tolerance
    ### Related to Parareal
    use_parareal::Bool
    parareal_args

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
    function MLMC_Experiment(problem::MLMC_Problem, qoi::Function, dist=Uniform(0.0, 1.0), L=2, ϵ=1e-3; seed=rand(UInt),
        use_parareal=false, parareal_args=())
        # Seed PRNG
        Random.seed!(seed)
        return new(problem, qoi, seed, dist, L, ϵ, use_parareal, parareal_args)
    end
end

"""
    run(experiment; kwargs...)

Run the MLMC experiments with settings supplied at construction time.
Additional kwargs are passed to DifferentialEquations.solve
"""
function run(experiment::MLMC_Experiment; verbose=true, warmup_samples=20, continuate=true, do_mse_splitting=true, do_regression=true, kwargs...)
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
    function sample_function(level::MultilevelEstimators.Level, ζ)
        # MultilevelEstimators uses a multi-index. For MLMC, this index only has one entry.
        l = level[1]

        # solve given sample (defined by ζ) on the current level
        sol_current_l, timesteps_current =
            solve(experiment.problem, experiment.problem.alg, (l, experiment.L), ζ,
                use_parareal=experiment.use_parareal,
                parareal_args=experiment.parareal_args; kwargs...)
        # compute corresponding QoI
        qoi_current_l = experiment.qoi(sol_current_l)

        if l == 0
            # update timesteps
            total_timesteps[l+1] += timesteps_current[1]
            sequential_timesteps[l+1] = max(sequential_timesteps[l+1], timesteps_current[2])
            return qoi_current_l, qoi_current_l
        end
        # solve given sample one level lower
        sol_last_l, timesteps_last =
            solve(experiment.problem, experiment.problem.alg, (l - 1, experiment.L), ζ,
                use_parareal=experiment.use_parareal,
                parareal_args=experiment.parareal_args; kwargs...)
        # compute QoI
        qoi_last_l = experiment.qoi(sol_last_l)
        qoi_diff = qoi_current_l - qoi_last_l

        # update timesteps
        total_timesteps[l+1] += timesteps_current[1] + timesteps_last[1]
        sequential_timesteps[l+1] = max(sequential_timesteps[l+1], timesteps_current[2] + timesteps_last[2]) # for a given sample, solutions for l and l-1 are currently done sequentially

        return qoi_diff, qoi_current_l
    end

    ###
    ### warm-up
    ###

    MLMC_estimator = MultilevelEstimators.Estimator(
        MultilevelEstimators.ML(),  # Multilevel index set
        MultilevelEstimators.MC(),  # Monte-Carlo sampling
        sample_function,            # (level, ζ) -> (ΔQ, Q)
        experiment.dist,
        save=false,
        ### force the use of all levels
        max_index_set_param=experiment.L,
        min_index_set_param=experiment.L,
        ### disable optimizations
        do_regression=false,
        continuate=false,
        do_mse_splitting=false,
        verbose=false,
        ### set number of warmup samples
        nb_of_warm_up_samples=warmup_samples
    )
    MultilevelEstimators.run(MLMC_estimator, 1e99)

    # use number of sequential steps as proxy for time under ideal conditions
    runtimes = deepcopy(sequential_timesteps)
    function cost_fct(level)
        # use first and only entry of multi-index,
        # correct offset (levels start from 0)
        runtimes[level[1]+1]
    end


    ###
    ### actual run
    ###

    # use runtimes obtained from warmup
    MLMC_estimator.options[:cost_model] = cost_fct

    # set number of samples already evaluated
    for level in range(0, experiment.L)
        MultilevelEstimators.add_to_total_work(MLMC_estimator, Level(level), warmup_samples)
    end

    # enable optimizations
    MLMC_estimator.options[:do_regression] = do_regression
    MLMC_estimator.options[:continuate] = continuate # perhaps not the best idea, since the runs with different tolerances will be sequential
    MLMC_estimator.options[:do_mse_splitting] = do_mse_splitting

    # enable output
    MLMC_estimator.options[:verbose] = verbose

    # actual run
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
        "timesteps" => [sum(total_timesteps), maximum(sequential_timesteps)] # assuming all samples of all levels are evaluated in parallel, which they currently aren't
    )

    return d
end
