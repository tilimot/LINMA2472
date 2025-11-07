include("flatten.jl")

module VectReverse
#const relu = Main.relu
const Flatten = Main.Flatten
mutable struct VectNode
	value::Any
	derivative::Any
	parents::Vector{Tuple{VectNode, Function}}
end


relu(x) = max(x, 0)
# For scalars
VectNode(x::Number) = VectNode(x, zero(x), Vector{Tuple{VectNode,Function}}())

# For vectors / matrix
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
    dydx = @. ifelse(x.value >= 0.0, 1.0, 0.0) # The >= ensures gradient is 0 at 0 -> fixed the error

    return VectNode(
        y,
        zero(y),
        [(x, Δ -> dydx .* Δ)]  
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
    # Each component of A.value is multiplied by X, and a Flatten of these results is returned
    return Flatten(map(a -> a * X, A.components))
end

# Same pour X * A 
function Base.:*(X::AbstractMatrix, A::Flatten{<:VectNode})
    return Flatten(map(a -> X * a, A.components))
end

# For A * B where A is a matrix and B is a Flatten
function Base.:*(A::AbstractMatrix, B::Flatten)
    return Flatten(map(b -> A * b, B.components))
end

# For A * B where B is a matrix and A is a Flatten
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

# Function added to prevent errors on INGInious (but code worked without it on VS code)
function Base.:-(x::VectNode)
    # -x.value forward, derivative is -1 times incoming Δ
    return VectNode(-x.value, zero(x.value), [(x, Δ -> -Δ)])
end

# Identity
function Base.broadcasted(::typeof(identity), x::VectNode)
    VectNode(
        identity.(x.value),
        zero(x.value),
        [(x, Δ -> Δ)]
    )
end

import Base: sum
function Base.sum(v::VectNode)
	s = sum(v.value)
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

# maximum(x; dims=2)
function Base.maximum(x::VectNode; dims=2) # "Base.maximum" defines a new method for the maximum function instead of defining a new maximum function
	mx = Base.maximum(x.value, dims=dims)
	mask = x.value .== mx
	return VectNode(mx, zero(mx), [(x, Δ -> mask .* Δ)])
end

# sum(x; dims=2)
function Base.sum(x::VectNode; dims=2)
	s = Base.sum(x.value, dims=dims)
	return VectNode(s, zero(s), [(x, Δ -> repeat(Δ, 1, size(x.value, 2)))])
end

# elementwise division for VectNode ./ VectNode
function Base.broadcasted(::typeof(/), x::VectNode, y::VectNode)
	VectNode(x.value ./ y.value, zero(x.value), [(x, Δ -> Δ ./ y.value), (y, Δ -> -Δ .* x.value ./ (y.value .^ 2))])
end

# softmax for VectNode (row-wise softmax)
import ..softmax # defines new methods for the existing softmax function instead of defining a new VectReverse.softmax functions
function softmax(x::VectNode)
	mx = Base.maximum(x.value, dims=2)
	exps = exp.(x.value .- mx)
	sums = Base.sum(exps, dims=2)
	s = exps ./ sums
	return VectNode(s, zero(s), [(x, Δ -> s .* (Δ .- sum(Δ .* s, dims=2)))])
end


Base.iterate(n::VectNode) = Base.iterate(n.value)
Base.iterate(n::VectNode, s) = Base.iterate(n.value, s)
Base.getindex(n::VectNode, i...) = getindex(n.value, i...)
Base.eachindex(n::VectNode) = eachindex(n.value)
Base.length(n::VectNode) = length(n.value)
Base.size(n::VectNode) = size(n.value)


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
	f.derivative = one.(f.value)	# Initialization of the output gradient to 1
	for node in reverse(topo)
		_backward!(node)
	end
end

function gradient!(f, g::Flatten, x::Flatten)
	# Converts each component to a VectNode
	x_nodes = Flatten(VectNode.(x.components))
	# function calculation
	expr = f(x_nodes)
	# Backprop
	backward!(expr)
	for i in eachindex(x.components)
		g.components[i] .= x_nodes.components[i].derivative
	end
	return g
end

gradient(f, x) = gradient!(f, zero(x), x)





################ SECOND ORDER
function gradient_for_hessian(f, x::Flatten)
    # Converts each component to a VectNode
    x_nodes = isa(x, Flatten{<:VectNode}) ? x : Flatten(VectNode.(x.components))
    # Function calculation
    expr = f(x_nodes)
    # Backprop
    backward!(expr)
    
    # Collect all derivatives into a single vector
    grad = vcat([vec(x_nodes.components[i].derivative) for i in eachindex(x_nodes.components)]...)
    
    # Wrap the concatenated derivatives into a single Flatten component
    return Flatten([grad])

end


function zero(x::VectNode)
	return zeros(x.value)
end
function zero(x::Flatten{VectNode})
    return Flatten([
        zeros(size(comp.value)) for comp in x.components
    ])
end

function zero(x::Matrix{Float64})
	return zeros(size(x))
end
function zero(x::Float64)
	return 0
end

function zero(x::Vector{Float64})
	return zeros(size(x))
end

# Jacobian-vector product `J(x) * tx`
function pushforward!(f, x::Flatten, i)
	
    for j in 1:length(x.components)
    if j == i
        x.components[j].derivative = fill!(similar(x.components[j].value), 1.0)
    else
        x.components[j].derivative = fill!(similar(x.components[j].value), 0.0)
    end
end
    return f(x)
end


# We don't know in advance the dimension of the output of `F`
# so we cannot easily redirect to a `jacobian!`
function jacobian(f, x)
    return reduce(hcat, map(i -> pushforward!(f, x, i), eachindex(x.components)))
end

function hessian(f, x)
	x_nodes = Flatten(VectNode.(x.components))
    return jacobian(z -> gradient_for_hessian(f, z), x_nodes)
end


end