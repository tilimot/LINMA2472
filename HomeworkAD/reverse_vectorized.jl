include("flatten.jl")
include("forward.jl")

module VectReverse

using Main.Forward
const relu = Main.relu
const Flatten = Main.Flatten

mutable struct VectNode
	value::Any
	derivative::Any
	parents::Vector{Tuple{VectNode, Function}}
end

import Base: zero
zero(x::VectNode) = VectNode(
    zero(x.value),      # same shape as value, but filled with zeros
    zero(x.derivative), # same shape as derivative, but filled with zeros
    Tuple{VectNode, Function}[]  # no parents
)

# For scalars
VectNode(x::Number) = VectNode(x, zero(x), Vector{Tuple{VectNode,Function}}())
VectNode(value, derivative) = VectNode(value, derivative, Tuple{VectNode, Function}[])
VectNode(x::VectNode) = x

# For vectors / matrix
VectNode(x::AbstractArray) = VectNode(x, zeros(size(x)), Vector{Tuple{VectNode,Function}}())
VectNode(x::Main.Forward.Dual) = VectNode(x, zero(x))

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
    dydx = @. ifelse(x.value >= 0.0, 1.0, 0.0) # The >= ensures gradient is 0 at 0 -> fixed the error for first order

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

# For x * y where both are VectNode 
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


function Base.:*(x::Vector{Float64}, y::Vector{Float64})
    return dot(x, y)
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


Base.ndims(::Type{VectNode}) = 0  
Base.ndims(x::VectNode) = ndims(x.value)  
Base.iterate(n::VectNode) = Base.iterate(n.value)
Base.iterate(n::VectNode, s) = Base.iterate(n.value, s)
Base.getindex(n::VectNode, i...) = getindex(n.value, i...)
Base.eachindex(n::VectNode) = eachindex(n.value)
Base.length(n::VectNode) = length(n.value)
Base.size(n::VectNode) = size(n.value)

# For function ones  with Matrix/Vector inputs
Base.ones(x::Vector{Float64}) = fill(1.0, size(x))
Base.ones(x::Matrix{Float64}) = fill(1.0, size(x))
Base.ones(x::Matrix{Vector}) = fill(1.0, size(x))
# For ones with VectNode
Base.ones(x::VectNode) = VectNode(ones(x.value), zero(x.value), Tuple{VectNode, Function}[])

Base.ones(dims::Tuple{Int,Int}) = fill(1.0, dims)
Base.copy(d::Forward.Dual) = Forward.Dual(d.value, d.derivative)
Base.transpose(d::Forward.Dual) = d
Base.adjoint(d::Forward.Dual) = d

# Implementation of copyto! 
function Base.copyto!(dest::VectNode, bc::Broadcast.Broadcasted{<:Any})
    # Copy values
    copyto!(dest.value, bc)
    
    # Initialize derivative and parents
    dest.derivative = zero(dest.value)
    empty!(dest.parents)
    
    # If arguments contain VectNode, set up the parents
    args = bc.args
    for arg in args
        if arg isa VectNode
            push!(dest.parents, (arg, Δ -> Δ))
        end
    end
    
    return dest
end

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

function flatten_to_vector(x::Flatten)
    return reduce(vcat, [vec(comp) for comp in x.components])
end


gradient(f, x) = gradient!(f, zero(x), x)

# ==============================================================================================
### Second order: Hessian
# ==============================================================================================

function vector_to_flatten(v::AbstractVector, template::Flatten)
    comps = Vector{Any}(undef, length(template.components))
    idx = 1
    for (j, comp_template) in enumerate(template.components)
        n = length(comp_template)                    # nb of elements in this component
        slice = @view v[idx:idx + n - 1]             
        comps[j] = reshape(slice, size(comp_template))  # same shape as the template component
        idx += n
    end
    return Flatten(comps)  
end

function hessian(f, x::Flatten)
    # We flatten parameters into a vector 
    x_vec = flatten_to_vector(x)  

    # We define g : ℝ^n (or Dual^n) -> ℝ^n
    #    g(v) = grad_f( unflatten(v) ), flattened again
    function g(v)
        x_flat_v = vector_to_flatten(v, x)     
        g_flat   = gradient(f, x_flat_v)       # gradient(f, ::Flatten) -> Flatten
        return flatten_to_vector(g_flat)       # Vector (Float64 or Dual)
    end

    # 3. Hessian = Jacobian of g in x_vec
    return Forward.jacobian(g, x_vec)
end

# ==============================================================================================
### Transformers
# ==============================================================================================

# Dans le module VectReverse

# Pour la transposée (Transpose)
function Base.transpose(x::VectNode)
    return VectNode(
        transpose(x.value),
        zero(transpose(x.value)),
        # La dérivée de transpose(f) par rapport à f est transpose(Δ)
        [(x, Δ -> transpose(Δ))] 
    )
end

# Pour l'adjointe (Adjoint), qui est souvent utilisée dans les produits matriciels.
function Base.adjoint(x::VectNode)
    return VectNode(
        adjoint(x.value),
        zero(adjoint(x.value)),
        # La dérivée de adjoint(f) par rapport à f est adjoint(Δ)
        [(x, Δ -> adjoint(Δ))]
    )
end

end
