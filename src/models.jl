using DrWatson
@quickactivate "MLMC_Parareal"

using DifferentialEquations
using NumericalIntegration
include(srcdir("problem.jl"))

# L2 norm
function L2_qoi(sol)
    return integrate(sol.t, sol[1, :] .^ 2)
end

### produce dahlquist problem from parametrization u_0, t_0, t_end, λ, Δt_0
function dahlquist_problem(u_0, t_0, t_end, λ, Δt_0)
    c = u_0, t_0, t_end, λ, Δt_0

    function dahlquist_deriv!(du, u, ζ, t)
        ζ = ζ[1]
        du[1] = λ * (1 + ζ) * u[1]
    end

    function problem(ζ)
        return ODEProblem(
            (du, u, ζ, t) -> dahlquist_deriv!(du, u, ζ, t),
            [u_0],
            (t_0, t_end),
            ζ)
    end

    function solve(level, ζ)
        dt = Δt_0 / 2^level
        return DifferentialEquations.solve(problem(ζ), Euler(), dt=dt)
    end

    function qoi(solution)
        return solution[end]
    end

    return MLMC_Problem(c, solve, qoi)
end