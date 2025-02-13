using Distributed

# precompile on control node
using MLMC_Parareal

# add workers
addprocs(N_PROCS)

# precompile on all workers
@everywhere begin
    using MLMC_Parareal
end

# load necessary dependencies on all workers
@everywhere begin
    using DrWatson
end
