using Distributed
using MLMC_Parareal

addprocs(10)

@everywhere begin
    using MLMC_Parareal
end

@everywhere begin
    using DrWatson
    using DifferentialEquations
    using MultilevelEstimators
    using BenchmarkTools
    using Plots
    using NumericalIntegration
end
