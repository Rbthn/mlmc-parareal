# run each model to make sure they run without errors

@testset "models" begin
    # %% Dahlquist equation
    @testset "dahlquist" begin
        # %% Problem
        u_0 = 1.0
        t_0 = 0.0
        t_end = 1.0
        λ = -1.0
        Δt_0 = 0.1 / 2^5

        p = Dahlquist_Problem(u_0, t_0, t_end, λ, Δt_0)
        alg = ImplicitEuler()


        # %% MLMC
        deviation = [0.5]
        dists = Uniform.(-deviation, deviation)
        L = 3
        mlmc_tol = 1e-2
        warmup_samples = 20
        qoi_fn = L2_squared
        run_args = (;
            continuate=false,
            do_mse_splitting=true,
            min_splitting=0.01,
            warmup_samples=warmup_samples
        )

        # %% Test if runs without Parareal
        e_ref = MLMC_Experiment(p, qoi_fn, dists,
            L, mlmc_tol;
            use_parareal=false,
        )
        @test res_ref = run(
            e_ref;
            run_args...
        )
    end


    # %% Steinmetz model
    @testset "steinmetz" begin

    end


    # %% FE model
    @testset "FE" begin

    end
end
