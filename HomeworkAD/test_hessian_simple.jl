LabAD = joinpath(dirname(@__DIR__), "LabAD")

include(joinpath(LabAD, "test", "test.jl"))
include(joinpath(LabAD, "solution", "forward.jl"))
include(joinpath(@__DIR__, "reverse_vectorized.jl"))

using LinearAlgebra

# Simple test case
function simple_test()
    println("=" ^ 60)
    println("Testing simple Hessian computation...")
    println("=" ^ 60)
    
    # Create a simple test case
    num_data = 3
    num_features = 2
    num_hidden = 2
    
    X = rand(num_data, num_features)
    y = rand(num_data)
    w = random_weights(X, y, num_hidden)
    
    # Normalize data
    X .= X .* 2 .- 1
    w.components .= [c .* 2 .- 1 for c in w.components]
    y .= y .* 2 .- 1
    
    println("X: ", size(X))
    println("y: ", size(y))
    println("w components: ", [size(c) for c in w.components])
    
    # Define loss
    L = loss(mse, identity_activation, X, y)
    
    # Compute Hessian with both methods
    println("\nComputing Forward.hessian...")
    ∇f = Forward.hessian(L, deepcopy(w))
    println("Forward.hessian computed successfully")
    ∇f_flat = flatten_gradient(∇f)
    println("Forward.hessian size: ", size(∇f_flat))
    
    println("\nComputing VectReverse.hessian...")
    try
        ∇r = VectReverse.hessian(L, deepcopy(w))
        println("VectReverse.hessian computed successfully")
        
        # Flatten and compare
        ∇r_flat = flatten_gradient(∇r)
        println("VectReverse.hessian size: ", size(∇r_flat))
        
        diff_norm = norm(∇f_flat - ∇r_flat)
        ref_norm = norm(∇f_flat)
        rel_error = diff_norm / ref_norm
        
        println("\nResults:")
        println("  Reference norm: ", ref_norm)
        println("  Difference norm: ", diff_norm)
        println("  Relative error: ", rel_error)
        println("  Tolerance: ", 1e-8)
        
        if isapprox(diff_norm, 0.0; atol = 1e-8 * ref_norm)
            println("\n✓ Test PASSED!")
        else
            println("\n✗ Test FAILED!")
            println("\nFirst 10 elements comparison:")
            for i in 1:min(10, length(∇f_flat))
                println("  [$i] Forward: $(∇f_flat[i]), VectReverse: $(∇r_flat[i]), diff: $(∇f_flat[i] - ∇r_flat[i])")
            end
        end
    catch e
        println("ERROR: ", e)
        println("\nStacktrace:")
        for (exc, bt) in Base.catch_stack()
            showerror(stdout, exc, bt)
            println()
        end
    end
    println("=" ^ 60)
end

simple_test()
