using DifferentialEquations
using SparseArrays
using LinearAlgebra

"""
    FE problem with constant mass, stiffness matrices given as Mx' + Kx = r.
"""
struct FE_Problem{T,U} <: MLMC_Problem{T,U}
    M::Function             # parameters -> mass matrix
    K::Function             # parameters -> stiffness matrix
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
        M::Union{AbstractMatrix{U},Function},
        K::Union{AbstractMatrix{U},Function},
        r::Function,
        kwargs...
    ) where {T<:AbstractFloat,U<:AbstractFloat}
        # validate time interval
        @assert t_0 <= t_end

        ndof = length(u_0)

        # convert matrices to functions if necessary
        if M isa AbstractMatrix
            # make sure system dimensions make sense
            @assert size(M) == (ndof, ndof)

            M_f = (p...) -> M
        else
            M_f = M
        end
        if K isa AbstractMatrix
            # make sure system dimensions make sense
            @assert size(K) == (ndof, ndof)

            K_f = (p...) -> K
        else
            K_f = K
        end

        new{T,U}(
            M_f,
            K_f,
            ndof,
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
    M = problem.M(ζ)
    K = problem.K(ζ)
    function rhs!(du, u, ζ, t)
        mul!(du, -K, u)
        du[:] .+= problem.r(t)
        return nothing
    end

    # Construct ODEProblem
    func = ODEFunction(rhs!, mass_matrix=M, jac_prototype=copy(K))
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
