using DrWatson
@quickactivate :MLMC_Parareal

# TIND : 4-pole induction machine
#
# Robrecht De Weerdt, Eindige Elementen Modellering van Kooianker Inductiemotoren, PhD thesis, KU Leuven, 1995.
# IRM-motor, p. 108.
#
# Prof. Dr.-Ing. Herbert De Gersem
# Institute for Accelerator Science and Electromagnetic Fields (TEMF)
# Technische Universität Darmstadt
# www.temf.de

### PROBLEM
## A. Data
f = 50                          # applied frequency [Hz]
w = 2 * pi * f                  # applied angular frequency [rad/s]
# A.1. Rated operation
Prat = 402e3                    # rated power (3 phases) [W]
Urat = 1716 / sqrt(3)           # rated voltage (equivalent phase, rms value) [V]
frat = 46.6                     # rated frequency [Hz]
Irat = 154                      # rated current [A]
cosphirat = 0.91                # rated power factor []
# A.2. Data
pp = 2                          # pole-pair number [#]
wmrat = 2 * pi * frat / pp      # rated speed [rad/s]
Mzrat = Prat / wmrat            # rated torque [Nm]
# A.3. Steinmetz model
nominal_params = (;
    pp=pp,                      # pole-pair number [#]
    J=8.4603 + 12.5014,         # moment of inertia of the rotor and drive train [kgm^2]
    R1=1.111140e-01,            # stator resistance [Ohm]
    Lh=6.407774e-02,            # main inductance [H]
    RFe=1.736354e+06,           # iron-loss resistance [Ohm]
    R2p=7.158602e-02,           # resistance of the squirrel case [Ohm]
    L1=1.649983e-03,            # leakage inductance of the stator windings[H]
    L2p=1.063014e-03,           # leakage inductance of the squirrel cage [H]
)

## Excitation (line start-up)
# applied voltage [V]
ftu = (t) -> Urat * [cos(w * t); cos(w * t - 2 * pi / 3); cos(w * t + 2 * pi / 3)];
# Line start-up with friction load [Nm]
ftload = (t, wm) -> Mzrat * (wm / wmrat);
u_0 = zeros(10)

## Timestepping
t_0 = 0.0
t_end = 1.0
Δt_0 = 1e-3

### PARAREAL
parareal_intervals = 10
parareal_reltol = 5e-2
parareal_abstol = 10
parareal_maxit = 4

args = (; adaptive=false, maxiters=typemax(Int), saveat=1e-3)
coarse_args = (; dt=1e-3)
fine_args = (; dt=1e-5)

p = Steinmetz_Problem(u_0, t_0, t_end, Δt_0; nominal_params..., w, ftu, ftload, args...)

# parareal
parareal_args = (;
    parareal_intervals=8,
    tol=1e-4,
    coarse_args=(;),
    fine_args=(;)
)





# MLMC
L = 2               # use refinement levels 0, 1, 2

# noise
deviation = 0.5
dist = Normal(0, deviation)

# QoI
qoi_fn = MLMC_Parareal.time_to_steady

# UQ tolerance
ϵ = 1e-3

e = MLMC_Experiment(p, qoi_fn, dist,
    L, ϵ, use_parareal=true, parareal_args=parareal_args)
result = run(e, continuate=false)

#save with git commit hash (and patch if repo is dirty)
problem_name = p.name
name = savename(problem_name, result["settings"], "jld2")
tagsave(datadir("simulations", name), result, storepatch=true)

print("done")
