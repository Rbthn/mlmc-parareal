using DifferentialEquations
using SparseArrays
using LinearAlgebra

"""
    FE problem with constant mass, stiffness matrices given as Mx' + Kx = r.
"""
struct FE_Problem{T,U} <: MLMC_Problem{T,U}
    M::AbstractMatrix{U}    # mass matrix
    K::AbstractMatrix{U}    # stiffness matrix
    dK::AbstractMatrix{U}   # stiffness used: K + ζ*dK
    ndof::Int               # number of unknowns

    r::Function             # provides right-hand side value as function of time

    solver_args::NamedTuple # additional kwargs to pass to solvers

    u_0::AbstractVector{U}  # initial condition
    t_0::T                  # start time
    t_end::T                # stop time
    Δt_0::T                 # time step at level 0
    alg                     # timestepping algorithm
    name::String

    function FE_Problem(
        u_0::AbstractVector{U},
        t_0::T,
        t_end::T,
        Δt_0::T;
        alg=ImplicitEuler(),
        M::AbstractMatrix{U},
        K::AbstractMatrix{U},
        dK::AbstractMatrix{U},
        r::Function,
        kwargs...
    ) where {T<:AbstractFloat,U<:AbstractFloat}
        # validate time interval
        @assert t_0 <= t_end

        # make sure system dimensions make sense
        dims = size(M)
        @assert dims == size(K)
        @assert dims[2] == length(u_0)

        new{T,U}(
            M,
            K,
            dK,
            dims[2],
            r,
            NamedTuple(kwargs),
            u_0,
            t_0,
            t_end,
            Δt_0,
            alg,
            "FE"
        )
    end
end

function instantiate_problem(problem::FE_Problem, ζ)
    function rhs!(du, u, p, t)
        K = problem.K + p[1] * problem.dK

        mul!(du, -K, u)
        du[:] .+= problem.r(t)
        return nothing
    end

    # Construct ODEProblem
    func = ODEFunction(rhs!, mass_matrix=problem.M, jac_prototype=copy(problem.K))
    return ODEProblem(
        func,
        problem.u_0,
        (problem.t_0, problem.t_end),
        ζ;
        problem.solver_args...
    )
end

function compute_timestep(problem::FE_Problem, level)
    return problem.Δt_0 / 10^level
end
