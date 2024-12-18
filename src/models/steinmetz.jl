using DifferentialEquations
using NumericalIntegration
using StaticArrays
using UnPack

struct Steinmetz_Problem{T,U} <: MLMC_Problem{T,U}
    """
        Hold nominal parameters of machine
        Required parameters:
            pp          # pole-pair number [#]
            J           # moment of inertia of the rotor and drive train [kgm^2]
            R1          # stator resistance [Ohm]
            Lh          # main inductance [H]
            RFe         # iron-loss resistance [Ohm]
            R2p         # resistance of the squirrel case [Ohm]
            L1          # leakage inductance of the stator windings[H]
            L2p         # leakage inductance of the squirrel cage [H]
    """
    nominal_params::NamedTuple

    w::Number               # applied angular frequency [rad/s]
    ftu::Function           # applied 3-phase line voltage as function of time [V]
    ftload::Function        # applied load torque as function of time [Nm]

    solver_args::NamedTuple # additional kwargs to pass to solvers

    u_0::SizedVector{10,U}  # initial conditions (speed and currents) [rad/s,A]
    t_0::T                  # start time
    t_end::T                # stop time
    Δt_0::T                 # time step at level 0
    alg                     # timestepping algorithm
    name::String

    # define internal constructor to check inputs
    function Steinmetz_Problem(u_0::Union{Vector{U},SizedVector{10,U}},
        t_0::T, t_end::T, Δt_0;
        alg=ImplicitEuler(),
        pp,
        J,
        R1,
        Lh,
        RFe,
        R2p,
        L1,
        L2p,
        w,
        ftu,
        ftload,
        kwargs...
    ) where {T<:AbstractFloat,U<:AbstractFloat}
        # validate time interval
        @assert t_0 <= t_end

        return new{T,U}(
            (;
                pp=pp,
                J=J,
                R1=R1,
                Lh=Lh,
                RFe=RFe,
                R2p=R2p,
                L1=L1,
                L2p=L2p
            ),
            w,
            ftu,
            ftload,
            NamedTuple(kwargs),
            u_0,
            t_0,
            t_end,
            Δt_0,
            alg,
            "Steinmetz"
        )
    end

end

function instantiate_problem(problem::Steinmetz_Problem, ζ)
    """
        function im_steinmetz!(du, u, p, t)
            return the time derivatives of y (see input parameters)
        (Herbert De Gersem, www.temf.de)

        input parameters:
            u               state variables
                u[1]        angular speed [rad/s]
                u[2:4]      three-phase currents through the main branch [A]
                u[5:7]      three-phase currents through the primary branch [A]
                u[8:10]     three-phase currents through the secondary branch (scaled) [A]


            p               parameters of the Steinmetz model for induction machines
                .pp         pole-pair number [1]
                .J          moment of inertia of the rotor and drive train [kgm^2]
                .R1         stator resistance [Ω]
                .Lh         main inductance [H]
                .RFe        iron-loss resistance [Ω]
                .R2p        resistance of the squirrel cage [Ω]
                .L1         leakage inductance of the stator windings [H]
                .L2p        leakage inductance of the squirrel cage [H]

            t               time [s]

        other parameters:
        w                   applied angular frequency [rad/s]
        ftu                 three-phase voltage as a function of time [V]
        ftload              load torque as a function of time and angular speed [Nm]
    """
    function im_steinmetz!(du, u, p, t)
        # parameters
        @unpack pp, J, R1, Lh, RFe, R2p, L1, L2p = problem.nominal_params
        w = problem.w
        ftu = problem.ftu
        ftload = problem.ftload

        # assume uncertainty in circuit parameters
        params = length(p)
        offset = (idx) -> idx <= params ? p[idx] : 0.0
        R1 += offset(1)
        Lh += offset(2)
        RFe += offset(3)
        R2p += offset(4)
        L1 += offset(5)
        L2p += offset(6)

        # helper
        function ftMz(i2p, wm)
            if abs((w - pp * wm) / w) < 1e-3
                return zeros(size(i2p, 1), 1)
            else
                return 3 * R2p * pp ./ (w - pp * wm) .* sum(i2p .^ 2, 2)
            end
        end

        # A. Speed and slip
        wm = u[1]
        s = 1 - pp * wm / w
        tol = 1e-6
        if abs(s) < tol
            if s > 0
                s = tol
            else
                s = -tol
            end
        end

        # B. Current
        ih = u[2:4]
        i1 = u[5:7]
        i2p = u[8:10]
        iFe = i1 - i2p - ih

        #du = [dwmdt; dihdt; di1dt; di2pdt], use in-place assignment
        du[1] = (3 * R2p * pp / w / s * sum(i2p .^ 2) - ftload(t, wm)) / J
        du[2:4] = RFe * iFe / Lh                        # dihdt
        du[5:7] = (ftu(t) - RFe * iFe - R1 * i1) / L1   # di1dt
        du[8:10] = (RFe * iFe - R2p / s * i2p) / L2p    # di2pdt

        return nothing
    end

    # Construct ODEProblem
    return ODEProblem(
        im_steinmetz!,                  # function that produces derivative
        problem.u_0,                    # initial value
        (problem.t_0, problem.t_end),   # time span
        ζ;                              # parameters
        problem.solver_args...
    )
end

"""
    Return time taken to reach 99.9% of end value
"""
function time_to_steady(sol::ODESolution)
    speed = [e[1] for e in sol.u]
    end_value = speed[end]
    thrsh = 0.99 * end_value

    idx = findfirst(x -> speed[x] >= thrsh, eachindex(speed))
    return sol.t[idx]
end

function compute_timestep(problem::Steinmetz_Problem, level)
    return problem.Δt_0 / 10^level
end
