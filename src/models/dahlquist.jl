using DrWatson
@quickactivate "MLMC_Parareal"

using DifferentialEquations
using NumericalIntegration
include(srcdir("problem.jl"))

struct Dahlquist_Problem{T,U} <: MLMC_Problem{T,U}

    u_0::Vector{U}  # initial value
    t_0::T          # start time
    t_end::T        # stop time
    λ               # factor in dahlquist equation
    Δt_0::T         # timestep at level 0
    use_parareal::Bool

    # define internal constructor to check inputs
    function Dahlquist_Problem(u_0::Vector{U}, t_0::T, t_end::T, λ, Δt_0; use_parareal=false) where {T<:AbstractFloat,U<:AbstractFloat}
        # validate time interval
        @assert t_0 <= t_end

        return new{T,U}(u_0, t_0, t_end, λ, Δt_0, use_parareal)
    end

    # convert scalar unknown to 1D vector to please DifferentialEquations
    function Dahlquist_Problem(u_0::U, t_0::T, t_end::T, λ, Δt_0; use_parareal=false) where {T<:AbstractFloat,U<:AbstractFloat}
        return Dahlquist_Problem([u_0], t_0, t_end, λ, Δt_0, use_parareal=use_parareal)
    end
end

function solve(problem::Dahlquist_Problem, level, ζ)
    l, L = level
    if !problem.use_parareal || l != L
        ### Don't use Parareal
        dt = compute_timestep(problem, l)
        p::ODEProblem = instantiate_problem(problem, ζ)
        return DifferentialEquations.solve(p, Euler(), dt=dt)
    else
        ### TODO use Parareal
    end
end

function instantiate_problem(problem::Dahlquist_Problem, ζ)
    function dahlquist_deriv!(du, u, ζ, t)
        σ = ζ[1]
        du[:] .= problem.λ * (1 + σ) .* u
    end

    # Construct ODEProblem
    return ODEProblem(
        dahlquist_deriv!,               # function that produces derivative
        problem.u_0,                    # initial value
        (problem.t_0, problem.t_end),   # time span
        ζ                               # parameters
    )
end