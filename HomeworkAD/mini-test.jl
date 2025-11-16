
include( "forward.jl")
include("reverse_vectorized.jl")
include("models.jl")

# MAINTENANT LA HESSIENNE:
# f(x) = sum(x.^2)
# x = [1.0, 1.0]

# # forward-on-forward
# VectReverse.gradient(f, x)

#VectReverse.hessian(f,x)


# Méthode: Forward-on-Reverse
# grad_f = z -> gradient_simple(f, z)
# H = jacobian_forward(grad_f, x)


using Test

# 1. Construire un Flatten identique aux tests
x = Flatten([ [1.0, 1.0] ])

# 2. Fonction f compatible VectNode/Flatten
f(x_nodes::Flatten) = sum(x_nodes.components[1] .^ 2)

# 3. Calcul des gradients
#ref_grad = Forward.gradient(f, deepcopy(x))
rev_grad = VectReverse.gradient(f, deepcopy(x))

println("reverse gradient non flatten: ", rev_grad)


# # 4. Flattening (extrait de vos tests)
# function flatten_gradient(g)
#     if isa(g, Float64)
#         return [g]
#     elseif hasproperty(g, :components)
#         return reduce(vcat, vec.(g.components))
#     elseif isa(g, AbstractArray) && !isa(g, Vector)
#         return reduce(vcat, flatten_gradient.(g))
#     else
#         return vec(g)
#     end
# end

# rg = flatten_gradient(rev_grad)

# #println("Reference Gradient: ", fg)
# println("Reverse Gradient:   ", rg)

# 5. Vérification
#@test isapprox(norm(fg - rg), 0.0; atol = 1e-8 * norm(fg))
