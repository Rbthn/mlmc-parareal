using DrWatson
@quickactivate "MLMC_Parareal"

### This type contains the problem that is passed to MLMC_experiment.
### It is initialized with some certain parameters and provides the methods
### solve(level, ζ) -> solution and
### qoi(solution) -> QoI

struct MLMC_Problem
    certain_parameters
    solve::Function
    qoi::Function
end
