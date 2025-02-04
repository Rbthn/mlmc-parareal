This directory contains the system matrices and right-hand-side vectors for the im_3kW induction machine extracted from GetDP.
To obtain the system matrices, one has to edit the `Formulation` section in a GetDP `.pro` file,
multiplying all `Dof`-terms by zero and leaving only `DtDof` terms (for M) and vice versa (for K).
The matrices are then extraced by inserting a `Print` directive (e.g. `Generate[A]; Print[A];`) in the `Resolution` section of the `.pro` file.

The following options were used to generate these matrices:

```julia
dt = period / 100       # 100 timesteps per period
t_0 = 0.0               # start time
t_end = 8 * period      # end time
Flag_AnalysisType=1,    # 1: time-domain
Flag_ImposedSpeed=2,    # 2: imposed speed
myrpm=0,                # 0: locked rotor
Flag_NL=0,              # 0: linear iron, 1: nonlinear iron
Flag_SrcType_Stator=1,  # 1: excite with current instead of voltage
```

As GetDP outputs matrices in Matlab format, the files here are read into julia using the function `JuliaGetDP._read_matrix`.
For easier handling afterwards, the function `JuliaGetDP._print_matrix` prints the julia code required to construct a matrix to a file.
The matrix can then be loaded into scope via `include`.
