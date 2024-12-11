using DifferentialEquations
using NumericalIntegration
using StaticArrays

struct Dahlquist_Problem{T,U,S} <: MLMC_Problem{T,U} where {S}

    u_0::SizedVector{S,U}# initial value
    t_0::T          # start time
    t_end::T        # stop time
    λ               # factor in dahlquist equation
    Δt_0::T         # timestep at level 0
    alg             # timestepping algorithm
    name::String

    # define internal constructor to check inputs
    function Dahlquist_Problem(u_0::SizedVector{S,U}, t_0::T, t_end::T, λ, Δt_0; alg=ImplicitEuler()) where {T<:AbstractFloat,U<:AbstractFloat,S}
        # validate time interval
        @assert t_0 <= t_end

        return new{T,U,S}(u_0, t_0, t_end, λ, Δt_0, alg, "Dahlquist")
    end

    # convert scalar unknown to 1D vector to please DifferentialEquations
    function Dahlquist_Problem(u_0::U, t_0::T, t_end::T, λ, Δt_0) where {T<:AbstractFloat,U<:AbstractFloat}
        return Dahlquist_Problem(SizedVector(u_0), t_0, t_end, λ, Δt_0)
    end
end

function instantiate_problem(problem::Dahlquist_Problem, ζ)
    function dahlquist_deriv!(du, u, ζ, t)
        σ = ζ[1]
        du[:] .= problem.λ * (1 + σ) .* u
        return nothing
    end

    # Construct ODEProblem
    return ODEProblem(
        dahlquist_deriv!,               # function that produces derivative
        problem.u_0,                    # initial value
        (problem.t_0, problem.t_end),   # time span
        ζ                               # parameters
    )
end
