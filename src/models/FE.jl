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
        M::Union{AbstractMatrix{U},Function,Symbol},
        K::Union{AbstractMatrix{U},Function,Symbol},
        r::Function,
        kwargs...
    ) where {T<:AbstractFloat,U<:AbstractFloat}
        # validate time interval
        @assert t_0 <= t_end

        ndof = length(u_0)


        if M isa Symbol
            # load global variable if given as Symbol
            M_f = (p...) -> getproperty(Main, M)

        elseif M isa AbstractMatrix
            # convert matrices to functions if necessary
            # make sure system dimensions make sense
            @assert size(M) == (ndof, ndof)

            M_f = (p...) -> M
        elseif M isa Function
            M_f = M
        else
            error("unexpected type for M: $(typeof(M))")
        end

        if K isa Symbol
            # load global variable if given as Symbol
            K_f = (p...) -> getproperty(Main, K)

        elseif K isa AbstractMatrix
            # make sure system dimensions make sense
            @assert size(K) == (ndof, ndof)

            K_f = (p...) -> K
        elseif K isa Function
            K_f = K
        else
            error("unexpected type for K: $(typeof(K))")
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
        mul!(du, K, -u)
        du[:] .+= problem.r(t)
        return nothing
    end

    function jac!(J, u, ζ, t)
        mul!(J, K, -I)
        return nothing
    end

    # Construct ODEProblem
    func = ODEFunction(rhs!, mass_matrix=M, jac=jac!, jac_prototype=copy(K))
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
