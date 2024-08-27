using DifferentialEquations
using NumericalIntegration
using StaticArrays

struct Pendulum_Problem{T,U} <: MLMC_Problem{T,U}

    u_0::SizedVector{2,U}# initial value
    t_0::T          # start time
    t_end::T        # stop time
    g               # gravity
    Δt_0::T         # timestep at level 0
    name::String

    # define internal constructor to check inputs
    function Pendulum_Problem(u_0::SizedVector{2,U}, t_0::T, t_end::T, g, Δt_0) where {T<:AbstractFloat,U<:AbstractFloat}
        # validate time interval
        @assert t_0 <= t_end

        return new{T,U}(u_0, t_0, t_end, g, Δt_0, "Pendulum")
    end

    function Pendulum_Problem(u_0::Vector{U}, t_0::T, t_end::T, g, Δt_0) where {T<:AbstractFloat,U<:AbstractFloat}
        return Pendulum_Problem(SizedVector{2,U}(u_0), t_0, t_end, g, Δt_0)
    end
end

function instantiate_problem(problem::Pendulum_Problem, ζ)
    function pendulum_deriv!(du, u, ζ, t)
        l = ζ[1]    # length of arm
        du[1] = u[2]
        du[2] = -problem.g / l * sin(u[1])
        return nothing
    end

    # Construct ODEProblem
    return ODEProblem(
        pendulum_deriv!,                # function that produces derivative
        problem.u_0,                    # initial value
        (problem.t_0, problem.t_end),   # time span
        ζ                               # parameters
    )
end