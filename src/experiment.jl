using DrWatson
@quickactivate "MLMC_Parareal"

using Random
using MultilevelEstimators
include(srcdir("problem.jl"))

struct MLMC_Experiment
    ### Related to discretization of the problem
    problem::MLMC_Problem
    qoi::Function
    ### Related to random sampling
    seed::UInt           #
    dist::AbstractDistribution  #
    ### Related to MLMC
    L::Int              # Use discretization levels l=0...L
    ϵ::Float64          # RMSE tolerance
    ### Related to Parareal
    use_parareal::Bool
    #
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
    function MLMC_Experiment(problem::MLMC_Problem, qoi::Function, dist=Uniform(0.0, 1.0), L=2, ϵ=1e-3; seed=rand(UInt), use_parareal=false)
        # Seed PRNG
        Random.seed!(seed)
        return new(problem, qoi, seed, dist, L, ϵ, use_parareal)
    end
end

function run(experiment::MLMC_Experiment; kwargs...)
    """
    Run the MLMC experiments with settings supplied at construction time.
    Additional kwargs are passed to DifferentialEquations.solve
    """
    ############################################################################
    ###########################   VALIDATE INPUTS   ############################
    ############################################################################
    # TODO

    ############################################################################
    #########################   COLLECT SYSTEM INFO   ##########################
    ############################################################################
    # TODO
    info = Dict()

    ############################################################################
    ###############################   RUN MLMC   ###############################
    ############################################################################
    function sample_function(level::MultilevelEstimators.Level, ζ)
        """
            This function is passed to MultilevelEstimators.
            Input:
                Realization ζ of the random variable
                Level of discretization level
            Output:
                (ΔQ, Q), where Q is the QoI obtained from a solution at the current level and ΔQ is the difference to the QoI obtained from the solution at the previous level
        """
        # MultilevelEstimators uses a multi-index. For MLMC, this index only has one entry.
        l = level[1]

        # solve given sample (defined by ζ) on the current level
        sol_current_l = solve(experiment.problem, (l, experiment.L), ζ, use_parareal=experiment.use_parareal; kwargs...)
        # compute corresponding QoI
        qoi_current_l = experiment.qoi(sol_current_l)

        if l == 0
            return qoi_current_l, qoi_current_l
        end
        # solve given sample one level lower
        sol_last_l = solve(experiment.problem, (l - 1, experiment.L), ζ, use_parareal=experiment.use_parareal; kwargs...)
        # compute QoI
        qoi_last_l = experiment.qoi(sol_last_l)
        qoi_diff = qoi_current_l - qoi_last_l

        return qoi_diff, qoi_current_l
    end

    # MultilevelEstimators allows the use of different distributions on each level. We only use one distribution for now.
    distributions = [experiment.dist for _ in range(1, experiment.L)]

    MLMC_estimator = MultilevelEstimators.Estimator(
        MultilevelEstimators.ML(), # Multilevel index set
        MultilevelEstimators.MC(), # Monte-Carlo sampling
        sample_function,    # (level, ζ) -> (ΔQ, Q)
        distributions,      # Vector of distributions for the different levels
        save=false,
        ###
        max_index_set_param=experiment.L,
        min_index_set_param=experiment.L,   # force the use of all levels
        # TODO additional settings
        nb_of_warm_up_samples=10,           # Number of initial samples to use for variance estimation
        continuate=false,                   # don't perform additional runs with looser tolerance
        do_regression=false,                # don't guess variance of next level based on convergence rate
        do_mse_splitting=false,             # no advanced splitting of permissible MSE between bias and variance, balance errors 50/50
    )

    # do warm-up run with very loose tolerance to take care of precompilation
    MultilevelEstimators.run(MLMC_estimator, 1)

    # TODO timing
    # do actual run
    h = MultilevelEstimators.run(MLMC_estimator, experiment.ϵ)


    ############################################################################
    #############################   SAVE RESULTS   #############################
    ############################################################################
    # inputs / fields of experiment
    settings = Dict(fieldnames(MLMC_Experiment) .=> getfield.(Ref(experiment), fieldnames(MLMC_Experiment)))

    # combine. Using strings here, as DrWatson does not like Symbols as keys.
    d = Dict("settings" => settings, "info" => info, "history" => h[1])

    #save with git commit hash (and patch if repo is dirty)
    savename = "tmp.jld2"
    return tagsave(datadir("simulations", savename), d, storepatch=true)
end