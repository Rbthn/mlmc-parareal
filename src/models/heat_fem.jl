using DifferentialEquations
using SparseArrays
using NumericalIntegration

struct Heat_Problem{T,U} <: MLMC_Problem{T,U}

    u_0::Vector{U}  # initial value for each node.
    u_BC::Vector{U} # Boundary conditions
    t_0::T          # start time
    t_end::T        # stop time
    Δt_0::T         # timestep at level 0
    xi::Vector      # space discretization as x-values of nodes. Size n
    cv::Vector      # volumetric heat capacity for each element. Size n-1
    k::Vector       # thermal conductivity for each element. Size n-1
    Q::Vector       # Heat source term. Constant per element and in time for now.
    alg
    name::String

    # define internal constructor to check inputs
    function Heat_Problem(u_0::Vector{U}, u_left, u_right, t_0::T, t_end::T, Δt_0, xi, cv, k, Q; alg=ImplicitEuler()) where {T<:AbstractFloat,U<:AbstractFloat}
        # validate time interval
        @assert t_0 <= t_end

        # validate sizes
        n = length(xi)
        @assert length(cv) == n - 1
        @assert length(k) == n - 1
        @assert length(Q) == n - 1

        return new{T,U}(u_0, [u_left, u_right], t_0, t_end, Δt_0, xi, cv, k, Q, alg, "Heat_FEM")
    end
end

function instantiate_problem(problem::Heat_Problem, ζ)
    n = length(problem.xi)
    # Boundary conditions
    idxBC = [1, n]
    idxDoF = setdiff(1:n, idxBC)

    function assemble_matrices(κ)
        M = spzeros(n, n)
        K = spzeros(n, n)
        r = zeros(n)
        for i in range(1, n - 1)
            Δx_i = problem.xi[i+1] - problem.xi[i]

            mass_stencil = [2 1; 1 2] * Δx_i / 6 * problem.cv[i]
            stiff_stencil = [1 -1; -1 1] * κ[i] / Δx_i # κ depends on realization ζ
            rhs_stencil = [1; 1] * Δx_i / 2 * problem.Q[i]

            M[i:i+1, i:i+1] += mass_stencil
            K[i:i+1, i:i+1] += stiff_stencil
            r[i:i+1] += rhs_stencil
        end

        # Dirichlet: Remove rows (test fct. vanishes),
        # move contributions in other rows to RHS
        r = r[idxDoF] -
            K[idxDoF, idxBC] * problem.u_BC #- M[idxDoF, idxBC] * problem.u_BC_diff
        M[idxBC, idxDoF] .= 0.0 # size of M has to be compatible to full u
        K = K[idxDoF, idxDoF]   # size of K can be reduced

        return M, K, r
    end

    M, K, r = assemble_matrices(repeat([ζ[1]], n - 1))

    function heat_deriv!(du, u, ζ, t)
        # Mu' = r - Ku
        du[idxDoF] = r - K * u[idxDoF]
        du[idxBC] .= 0.0 # for convenience, include idxBC are not eliminated from u
        return nothing
    end

    return ODEProblem(
        ODEFunction(heat_deriv!, mass_matrix=M),    # avoid inverting the mass matrix
        problem.u_0,                    # initial value
        (problem.t_0, problem.t_end),   # time span
        ζ                               # parameters
    )
end
