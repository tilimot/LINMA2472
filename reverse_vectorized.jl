include("flatten.jl")

module VectReverse
const relu = Main.relu
const Flatten = Main.Flatten
mutable struct VectNode
	value::Any
	derivative::Any
	parents::Vector{Tuple{VectNode, Function}}
end


# Constructeurs pratiques
# pour scalaires
VectNode(x::Number) = VectNode(x, zero(x), Vector{Tuple{VectNode,Function}}())

# pour vecteurs / matrices
VectNode(x::AbstractArray) = VectNode(x, zeros(size(x)), Vector{Tuple{VectNode,Function}}())

# For `tanh.(X)`
function Base.broadcasted(::typeof(tanh), x::VectNode)
	VectNode(
		tanh.(x.value),
		zero(x.value),
		[(x, Δ -> (1 .- tanh.(x.value).^2) .* Δ)]
	)
end

# For ReLU
function Base.broadcasted(::typeof(relu), x::VectNode)
    y = max.(x.value, 0.0)
    dydx = @. ifelse(x.value >= 0.0, 1.0, 0.0)

    return VectNode(
        y,
        zero(y),
        [(x, Δ -> dydx .* Δ)]  # on multiplie simplement par la dérivée locale
    )
end

relu_activation(x) = relu.(x)

# For `X .* Y`
function Base.broadcasted(op::Function, x::VectNode, y::VectNode)
    VectNode(
		x.value .* y.value,
		zero(x.value),
		[(x, Δ -> Δ .* y.value), (y, Δ -> x.value .* Δ)]
	)
end

# For `X .* Y` where `Y` is a constant
function Base.broadcasted(op::Function, x::VectNode, y::Union{AbstractArray,Number})
    VectNode(
		x.value .* y,
		zero(x.value),
		[(x, Δ -> Δ .* y)]
	)
end

# For `X .* Y` where `X` is a constant
function Base.broadcasted(op::Function, x::Union{AbstractArray,Number}, y::VectNode)
    VectNode(
		x .* y.value,
		zero(y.value),
		[(y, Δ -> x .* Δ)]
	)
end

# For A*x where A is a matrix and x a VectNode
function Base.:*(A::AbstractMatrix, x::VectNode)
	return VectNode(
		A * x.value,
		zero(A * x.value),
		[(x, Δ -> A' * Δ)]
	)
end

# For A*x where x is a matrix and A a VectNode
function Base.:*(A::VectNode, x::AbstractMatrix)
	return VectNode(
		A.value * x,
		zero(A.value * x),
		[(A, Δ -> Δ * x')]
	)
end

# For x * y where both are VectNode (element-wise or matrix depending on values)
function Base.:*(x::VectNode, y::VectNode)
	return VectNode(
		x.value * y.value,
		zero(x.value * y.value),
		[(x, Δ -> Δ * y.value'), (y, Δ -> x.value' * Δ)]
	)
end

function Base.:*(x::VectNode, y::Union{AbstractArray,Number})
    VectNode(
        x.value * y,
        zero(x.value * y),
        [(x, Δ -> Δ * y')]
    )
end

function Base.:*(x::Union{AbstractArray,Number}, y::VectNode)
    VectNode(
        x * y.value,
        zero(x * y.value),
        [(y, Δ -> x' * Δ)]
    )
end

# For A * X where A is a Flatten of VectNodes and X is a matrix
function Base.:*(A::Flatten{<:VectNode}, X::AbstractMatrix)
    # On multiplie chaque composant de A.value (donc chaque matrice contenue dans les VectNode) par X, et on renvoie un Flatten de ces résultats.
    return Flatten(map(a -> a * X, A.components))
end

# Idem pour X * A (utile selon la structure du modèle)
function Base.:*(X::AbstractMatrix, A::Flatten{<:VectNode})
    return Flatten(map(a -> X * a, A.components))
end

function Base.:*(A::AbstractMatrix, B::Flatten)
    return Flatten(map(b -> A * b, B.components))
end

function Base.:*(A::Flatten, B::AbstractMatrix)
    return Flatten(map(a -> a * B, A.components))
end

# For 'X./Y'
function Base.broadcasted(::typeof(/), x::VectNode, y::VectNode)
	VectNode(
		x.value ./ y.value,
		zero(x.value),
		[(x, Δ -> Δ ./ y.value), (y, Δ -> -Δ .* x.value ./ (y.value .^ 2))]
	)
end

# For `X ./ Y` where `Y` is a constant
function Base.broadcasted(::typeof(/), x::VectNode, y::Union{AbstractArray,Number})
	VectNode(
		x.value ./ y,
		zero(x.value),
		[(x, Δ -> Δ ./ y)]
	)
end

# For `X ./ Y` where `X` is a constant
function Base.broadcasted(::typeof(/), x::Union{AbstractArray,Number}, y::VectNode)
	VectNode(
		x ./ y.value,
		zero(y.value),
		[(y, Δ -> -Δ .* x ./ (y.value .^ 2))]
	)
end

# For `x .^ 2`
function Base.broadcasted(::typeof(Base.literal_pow), ::typeof(^), x::VectNode, ::Val{y}) where {y}
	Base.broadcasted(^, x, y)
end

# For `X .^ y'
function Base.broadcasted(::typeof(^), x::VectNode, y::VectNode)
	VectNode(
		x.value .^ y.value,
		zero(x.value),
		[(x, Δ -> Δ .* (y.value .* (x.value .^ (y.value .- 1)))), 
		 (y, Δ -> Δ .* (log.(x.value) .* (x.value .^ y.value)))]
	)
end

# For `X .^ y` where `y` is a constant
function Base.broadcasted(::typeof(^), x::VectNode, y::Union{AbstractArray,Number})
	VectNode(
		x.value .^ y,
		zero(x.value),
		[(x, Δ -> Δ .* (y .* (x.value .^ (y .- 1))))]
	)
end

# For `X .^ y` where `X` is a constant
function Base.broadcasted(::typeof(^), x::Union{AbstractArray,Number}, y::VectNode)
	VectNode(
		x .^ y.value,
		zero(y.value),
		[(y, Δ -> Δ .* (log.(x) .* (x .^ y.value)))]
	)
end
# For X. + Y
function Base.broadcasted(::typeof(+), x::VectNode, y::VectNode)
    VectNode(
        x.value .+ y.value,
        zero(x.value),
        [(x, Δ -> Δ), (y, Δ -> Δ)]
    )
end

# For `X .+ Y` where `Y` is a constant
function Base.broadcasted(::typeof(+), x::VectNode, y::Union{AbstractArray,Number})
    VectNode(
        x.value .+ y,
        zero(x.value),
        [(x, Δ -> Δ)]
    )
end
# For `X .+ Y` where `X` is a constant
function Base.broadcasted(::typeof(+), x::Union{AbstractArray,Number}, y::VectNode)
	VectNode(
		x .+ y.value,
		zero(y.value),
		[(y, Δ -> Δ)]
	)
end

# For `X .- Y`
function Base.broadcasted(::typeof(-), x::VectNode, y::VectNode)
	VectNode(
		x.value .- y.value,
		zero(x.value),
		[(x, Δ -> Δ), (y, Δ -> -Δ)]
	)
end

# For `X .- Y` where `Y` is a constant
function Base.broadcasted(::typeof(-), x::VectNode, y::Union{AbstractArray,Number})
	VectNode(
		x.value .- y,
		zero(x.value),
		[(x, Δ -> Δ)]
	)
end

# For `X .- Y` where `X` is a constant
function Base.broadcasted(::typeof(-), x::Union{AbstractArray,Number}, y::VectNode)
	VectNode(
		x .- y.value,
		zero(y.value),
		[(y, Δ -> -Δ)]
	)
end

# Non-broadcasted subtraction: support VectNode - array/number, array/number - VectNode, and VectNode - VectNode
function Base.:-(x::VectNode, y::Union{AbstractArray,Number})
	VectNode(
		x.value .- y,
		zero(x.value),
		[(x, Δ -> Δ)]
	)
end

function Base.:-(x::Union{AbstractArray,Number}, y::VectNode)
	VectNode(
		x .- y.value,
		zero(y.value),
		[(y, Δ -> -Δ)]
	)
end

function Base.:-(x::VectNode, y::VectNode)
	VectNode(
		x.value .- y.value,
		zero(x.value),
		[(x, Δ -> Δ), (y, Δ -> -Δ)]
	)
end

# Identity
function Base.broadcasted(::typeof(identity), x::VectNode)
    VectNode(
        identity.(x.value),
        zero(x.value),
        [(x, Δ -> Δ)]
    )
end

########################################################################################"

function topo_sort!(visited, topo, f::VectNode)
	if !(f in visited)
		push!(visited, f)
		for (parent, _) in f.parents
			topo_sort!(visited, topo, parent)
		end
		push!(topo, f)
	end
end

function _backward!(f::VectNode)
	for (parent, backprop_fn) in f.parents
		parent.derivative = parent.derivative .+ backprop_fn(f.derivative)
	end
end

function backward!(f::VectNode)
	visited = Set{VectNode}()
	topo = VectNode[]
	topo_sort!(visited, topo, f)
	f.derivative = one.(f.value)	# Initialisation du gradient de la sortie à 1
	for node in reverse(topo)
		_backward!(node)
	end
end

import Base: sum
function Base.sum(v::VectNode)
	s = sum(v.value)
	# derivative of sum each element is 1, so backprop returns Δ for each element
	return VectNode(s, zero(s), [(v, Δ -> fill(Δ, size(v.value)))])
end

function Base.:/(v::VectNode, n::Number)
	res = v.value / n
	return VectNode(res, zero(res), [(v, Δ -> Δ / n)])
end

# Exponential (element-wise) broadcast
function Base.broadcasted(::typeof(exp), x::VectNode)
	y = exp.(x.value)
	VectNode(y, zero(y), [(x, Δ -> Δ .* y)])
end

# Log (element-wise) broadcast
function Base.broadcasted(::typeof(log), x::VectNode)
	y = log.(x.value)
	VectNode(y, zero(y), [(x, Δ -> Δ ./ x.value)])
end

# maximum(x, dims=2) -> returns a column vector of row-wise maxima
function maximum(x::VectNode; dims=2)
	mx = maximum(x.value, dims=2)
	# backprop: gradient is 1 for positions equal to the max (ties split arbitrarily)
	return VectNode(mx, zero(mx), [(x, Δ -> map((row, d) -> (row .== maximum(row)) .* d, eachrow(x.value), eachrow(Δ)) |> vcat)])
end

# sum over dims=2 returning column vector
function sum(x::VectNode; dims=2)
	s = sum(x.value, dims=2)
	return VectNode(s, zero(s), [(x, Δ -> repeat(Δ, 1, size(x.value, 2)))])
end

# elementwise division for VectNode ./ VectNode
function Base.broadcasted(::typeof(/), x::VectNode, y::VectNode)
	VectNode(x.value ./ y.value, zero(x.value), [(x, Δ -> Δ ./ y.value), (y, Δ -> -Δ .* x.value ./ (y.value .^ 2))])
end

# Dedicated softmax for VectNode (row-wise softmax). Uses numerically stable subtraction
# and attaches the correct Jacobian-based backprop for each row.
function softmax(x::VectNode)
	# x.value is (n, k)
	mx = maximum(x.value, dims=2)
	exps = exp.(x.value .- mx)
	sums = sum(exps, dims=2)
	s = exps ./ sums
	return VectNode(s, zero(s), [(x, Δ -> s .* (Δ .- sum(Δ .* s, dims=2)))])
end


Base.iterate(n::VectNode) = Base.iterate(n.value)
Base.iterate(n::VectNode, s) = Base.iterate(n.value, s)
Base.getindex(n::VectNode, i...) = getindex(n.value, i...)
Base.eachindex(n::VectNode) = eachindex(n.value)
Base.length(n::VectNode) = length(n.value)
Base.size(n::VectNode) = size(n.value)


function gradient!(f, g::Flatten, x::Flatten)
	# Convertit chaque composant en VectNode
	x_nodes = Flatten(VectNode.(x.components))
	# Calcul de la fonction
	expr = f(x_nodes)
	# Backprop
	backward!(expr)
	# Récupère les dérivées
	for i in eachindex(x.components)
		g.components[i] .= x_nodes.components[i].derivative
	end
	return g
end

gradient(f, x) = gradient!(f, zero(x), x)

end
