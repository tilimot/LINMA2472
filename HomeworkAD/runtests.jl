LabAD = joinpath(dirname(@__DIR__), "LabAD")

include(joinpath(LabAD, "test", "test.jl"))

# Reference implementation we test against
include(joinpath(LabAD, "solution", "forward.jl"))

## First order
include(joinpath(@__DIR__, "reverse_vectorized.jl"))
# run_gradient_tests(Forward.gradient, VectReverse.gradient)
W0 = randn(3, 4)
b0 = randn(4)
x_flat = Flatten([W0, b0])          # pour VectReverse
x_vec  = vcat(vec(W0), b0)          # pour Forward (vecteur 16)

# 2. Fonction pour Forward.hessian : f_forward(::Vector)
function f_forward(z::Vector{Float64})
    W = reshape(z[1:12], 3, 4)
    b = z[13:16]
    y = W * b
    return 0.5 * sum(y.^2)
end

# 3. Même fonction pour VectReverse.hessian : f_reverse(::Flatten)
function f_reverse(x::Flatten)
    W = x.components[1]
    b = x.components[2]
    y = W * b
    # VectNode feuille, scalaire
    return VectReverse.VectNode(
        0.5 * sum(y.^2),                                  # value
        0.0,                                              # derivative init
        Vector{Tuple{VectReverse.VectNode, Function}}()   # parents vides
    )
end

# 4. Hessiennes
H_ref = Forward.hessian(f_forward, x_vec)
H_rev = VectReverse.hessian(f_reverse, x_flat)

println("H_ref size = ", size(H_ref))
println("H_rev size = ", size(H_rev))
println("norm diff = ", norm(H_ref - H_rev))

## Second order
# We only test `hessian` and not `hvp` but if `hessian` is implemented
# by reusing `hvp`, this is testing both at the same time.
#run_gradient_tests(Forward.hessian, VectReverse.hessian, hessian = true)
