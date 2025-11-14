include("flatten.jl")



module VectReverse
const relu = Main.relu
const Flatten = Main.Flatten
mutable struct VectNode
	value::Any
	derivative::Any
	parents::Vector{Tuple{VectNode, Function}}
end

import Base: zero
zero(x::VectNode) = VectNode(
    zero(x.value),      # même shape que value, mais rempli de zéros
    zero(x.derivative), # même shape que derivative, mais rempli de zéros
    Tuple{VectNode, Function}[]  # pas de parents
)
# For scalars
VectNode(x::Number) = VectNode(x, zero(x), Vector{Tuple{VectNode,Function}}())
VectNode(value, derivative) = VectNode(value, derivative, Tuple{VectNode, Function}[])
VectNode(x::VectNode) = x
# For vectors / matrix
VectNode(x::AbstractArray) = VectNode(x, zeros(size(x)), Vector{Tuple{VectNode,Function}}())
function VectNode(value)
    return VectNode(
        value,
        zero(value),                           # même type que value (Matrix ou VectDual)
        Tuple{VectNode, Function}[]            # tableau vide de parents
    )
end

# tanh.(X)
function Base.broadcasted(::typeof(tanh), x::VectNode)
    d = 1 .- tanh.(x.value).^2
    VectNode(
        tanh.(x.value),
        zero(x.value),
        [(x, Δ -> d .* Δ)]
    )
end

# ReLU
function Base.broadcasted(::typeof(relu), x::VectNode)
    y = max.(x.value, 0.0)
    dydx = @. ifelse(x.value >= 0.0, 1.0, 0.0)
    VectNode(
        y,
        zero(y),
        [(x, Δ -> dydx .* Δ)]
    )
end

relu_activation(x) = relu.(x)

# X .* Y (VectNode, VectNode)
function Base.broadcasted(::typeof(*), x::VectNode, y::VectNode)
    VectNode(
        x.value .* y.value,
        zero(x.value),
        [(x, Δ -> Δ .* y.value),
         (y, Δ -> x.value .* Δ)]
    )
end

# X .* Y (VectNode, cste)
function Base.broadcasted(::typeof(*), x::VectNode, y::Union{AbstractArray,Number})
    VectNode(
        x.value .* y,
        zero(x.value),
        [(x, Δ -> Δ .* y)]
    )
end

# X .* Y (cste, VectNode)
function Base.broadcasted(::typeof(*), x::Union{AbstractArray,Number}, y::VectNode)
    VectNode(
        x .* y.value,
        zero(y.value),
        [(y, Δ -> x .* Δ)]
    )
end

# A * x, A matrice, x VectNode
function Base.:*(A::AbstractMatrix, x::VectNode)
    VectNode(
        A * x.value,
        zero(A * x.value),
        [(x, Δ -> A' * Δ)]
    )
end

# A * x, A VectNode, x matrice
function Base.:*(A::VectNode, x::AbstractMatrix)
    VectNode(
        A.value * x,
        zero(A.value * x),
        [(A, Δ -> Δ * x')]
    )
end

# x * y, x et y VectNode (produit matrice)
function Base.:*(x::VectNode, y::VectNode)
    VectNode(
        x.value * y.value,
        zero(x.value * y.value),
        [(x, Δ -> Δ * y.value'),
         (y, Δ -> x.value' * Δ)]
    )
end

# x * y, x VectNode, y array/scalaire
function Base.:*(x::VectNode, y::Union{AbstractArray,Number})
    VectNode(
        x.value * y,
        zero(x.value * y),
        [(x, Δ -> Δ * y')]
    )
end

# x * y, x array/scalaire, y VectNode
function Base.:*(x::Union{AbstractArray,Number}, y::VectNode)
    VectNode(
        x * y.value,
        zero(x * y.value),
        [(y, Δ -> x' * Δ)]
    )
end

# A * X où A est Flatten{VectNode}, X matrice
function Base.:*(A::Flatten{<:VectNode}, X::AbstractMatrix)
    Flatten(map(a -> a * X, A.components))
end

# X * A où A est Flatten{VectNode}
function Base.:*(X::AbstractMatrix, A::Flatten{<:VectNode})
    Flatten(map(a -> X * a, A.components))
end

# A * B où A matrice, B Flatten (générique)
function Base.:*(A::AbstractMatrix, B::Flatten)
    Flatten(map(b -> A * b, B.components))
end

# A * B où A Flatten, B matrice
function Base.:*(A::Flatten, B::AbstractMatrix)
    Flatten(map(a -> a * B, A.components))
end


# X ./ Y (VectNode, VectNode)
function Base.broadcasted(::typeof(/), x::VectNode, y::VectNode)
    VectNode(
        x.value ./ y.value,
        zero(x.value),
        [(x, Δ -> Δ ./ y.value),
         (y, Δ -> -Δ .* x.value ./ (y.value .^ 2))]
    )
end

# X ./ Y (VectNode, cste)
function Base.broadcasted(::typeof(/), x::VectNode, y::Union{AbstractArray,Number})
    VectNode(
        x.value ./ y,
        zero(x.value),
        [(x, Δ -> Δ ./ y)]
    )
end

# X ./ Y (cste, VectNode)
function Base.broadcasted(::typeof(/), x::Union{AbstractArray,Number}, y::VectNode)
    VectNode(
        x ./ y.value,
        zero(y.value),
        [(y, Δ -> -Δ .* x ./ (y.value .^ 2))]
    )
end


# x .^ 2
function Base.broadcasted(::typeof(Base.literal_pow), ::typeof(^), x::VectNode, ::Val{y}) where {y}
    Base.broadcasted(^, x, y)
end

# X .^ Y (VectNode, VectNode)
function Base.broadcasted(::typeof(^), x::VectNode, y::VectNode)
    VectNode(
        x.value .^ y.value,
        zero(x.value),
        [(x, Δ -> Δ .* (y.value .* (x.value .^ (y.value .- 1)))),
         (y, Δ -> Δ .* (log.(x.value) .* (x.value .^ y.value)))]
    )
end

# X .^ y (VectNode, cste)
function Base.broadcasted(::typeof(^), x::VectNode, y::Union{AbstractArray,Number})
    VectNode(
        x.value .^ y,
        zero(x.value),
        [(x, Δ -> Δ .* (y .* (x.value .^ (y .- 1))))]
    )
end

# x .^ Y (cste, VectNode)
function Base.broadcasted(::typeof(^), x::Union{AbstractArray,Number}, y::VectNode)
    VectNode(
        x .^ y.value,
        zero(y.value),
        [(y, Δ -> Δ .* (log.(x) .* (x .^ y.value)))]
    )
end

# X .+ Y
function Base.broadcasted(::typeof(+), x::VectNode, y::VectNode)
    VectNode(
        x.value .+ y.value,
        zero(x.value),
        [(x, Δ -> Δ), (y, Δ -> Δ)]
    )
end

# X .+ cste
function Base.broadcasted(::typeof(+), x::VectNode, y::Union{AbstractArray,Number})
    VectNode(
        x.value .+ y,
        zero(x.value),
        [(x, Δ -> Δ)]
    )
end

# cste .+ Y
function Base.broadcasted(::typeof(+), x::Union{AbstractArray,Number}, y::VectNode)
    VectNode(
        x .+ y.value,
        zero(y.value),
        [(y, Δ -> Δ)]
    )
end

# X .- Y
function Base.broadcasted(::typeof(-), x::VectNode, y::VectNode)
    VectNode(
        x.value .- y.value,
        zero(x.value),
        [(x, Δ -> Δ), (y, Δ -> -Δ)]
    )
end

# X .- cste
function Base.broadcasted(::typeof(-), x::VectNode, y::Union{AbstractArray,Number})
    VectNode(
        x.value .- y,
        zero(x.value),
        [(x, Δ -> Δ)]
    )
end

# cste .- Y
function Base.broadcasted(::typeof(-), x::Union{AbstractArray,Number}, y::VectNode)
    VectNode(
        x .- y.value,
        zero(y.value),
        [(y, Δ -> -Δ)]
    )
end

# Non-broadcasted soustraction
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

function Base.:-(x::VectNode)
    VectNode(-x.value, zero(x.value), [(x, Δ -> -Δ)])
end

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
    return VectNode(s, zero(s), [(v, Δ ->
        Δ isa VectDual ?
            VectDual(fill(Δ.value, size(v.value)),
                     fill(Δ.tangent, size(v.value))) :
            fill(Δ, size(v.value)) )])
end

# division par scalaire
function Base.:/(v::VectNode, n::Number)
    res = v.value / n
    return VectNode(res, zero(res), [(v, Δ -> Δ / n)])
end

# exp.(x)
function Base.broadcasted(::typeof(exp), x::VectNode)
    y = exp.(x.value)
    VectNode(y, zero(y), [(x, Δ -> Δ .* y)])
end

# log.(x)
function Base.broadcasted(::typeof(log), x::VectNode)
    y = log.(x.value)
    VectNode(y, zero(y), [(x, Δ -> Δ ./ x.value)])
end

# maximum(x; dims=2) : on applique un masque, il faut dupliquer Δ
function Base.maximum(x::VectNode; dims=2)
    mx = Base.maximum(x.value, dims=dims)
    mask = x.value .== mx
    return VectNode(mx, zero(mx), [(x, Δ ->
        Δ isa VectDual ?
            VectDual(mask .* Δ.value, mask .* Δ.tangent) :
            mask .* Δ)])
end

# sum(x; dims=2) : même idée, duplication de Δ sur chaque colonne
function Base.sum(x::VectNode; dims=2)
    s = Base.sum(x.value, dims=dims)
    return VectNode(s, zero(s), [(x, Δ ->
        Δ isa VectDual ?
            VectDual(repeat(Δ.value, 1, size(x.value, 2)),
                     repeat(Δ.tangent, 1, size(x.value, 2))) :
            repeat(Δ, 1, size(x.value, 2)) )])
end

# elementwise division pour VectNode ./ VectNode 
function Base.broadcasted(::typeof(/), x::VectNode, y::VectNode)
    VectNode(x.value ./ y.value, zero(x.value), [
        (x, Δ -> Δ ./ y.value),
        (y, Δ -> -Δ .* x.value ./ (y.value .^ 2))
    ])
end

# Produit scalaire classique (pour certains tests)
function Base.:*(x::Vector{Float64}, y::Vector{Float64})
    return dot(x, y)
end


function softmax(x::VectNode)
    mx   = Base.maximum(x.value, dims=2)
    exps = exp.(x.value .- mx)
    sums = Base.sum(exps, dims=2)
    s    = exps ./ sums
    return VectNode(
        s,
        zero(s),
        [(x, Δ -> s .* (Δ .- sum(Δ .* s, dims=2)))]
    )
end


Base.ndims(::Type{VectNode}) = 0  
Base.ndims(x::VectNode) = ndims(x.value)  
#=Base.iterate(n::VectNode) = Base.iterate(n.value)
Base.iterate(n::VectNode, s) = Base.iterate(n.value, s)
Base.getindex(n::VectNode, i...) = getindex(n.value, i...) =#
Base.eachindex(n::VectNode) = eachindex(n.value)
Base.length(n::VectNode) = length(n.value)
Base.size(n::VectNode) = size(n.value)

Base.ones(x::Vector{Float64}) = fill(1.0, size(x))
Base.ones(x::Matrix{Float64}) = fill(1.0, size(x))
Base.ones(x::Matrix{Vector}) = fill(1.0, size(x))

Base.ones(x::VectNode) = VectNode(ones(x.value), zero(x.value), Tuple{VectNode, Function}[])

Base.ones(dims::Tuple{Int,Int}) = fill(1.0, dims)


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
Base.one(x::AbstractArray{<:Number}) = ones(size(x))

function backward!(f::VectNode)
	visited = Set{VectNode}()
	topo = VectNode[]
	topo_sort!(visited, topo, f)
	
    for node in topo
        node.derivative = zero(node.derivative)
    end
	if f.value isa VectDual
        f.derivative = VectDual(1.0, f.value.tangent)         # VectDual(1, 0)
    elseif isa(f.value, Number)
        f.derivative = one(f.value)              # 1.0
    else
        f.derivative = ones(size(f.value))       # vecteur / matrice de 1
    end

	for node in reverse(topo)
		_backward!(node)
	end
end

function gradient!(f, g::Flatten, x::Flatten)
    # Convertit chaque composant en VectNode
    x_nodes = Flatten(VectNode.(x.components))
    expr = f(x_nodes)
    backward!(expr)
    for i in eachindex(x.components)
        g.components[i] = x_nodes.components[i].derivative
    end
    return g
end

function gradient(f, x::Flatten)
    g = Flatten([zero(c) for c in x.components])
    return gradient!(f, g, x)
end


function onehot(x::Flatten, i::Integer)
    tx = zero(x)                     # même structure que x, rempli de zéros
    tx.components[i] .= ones(tx.components[i])
    return tx
end


# Second order
struct VectDual
    value::Union{AbstractArray, Number}
    tangent::Union{AbstractArray, Number}
	
    function VectDual(v, t)
        if v isa VectDual || t isa VectDual
            error("Nested VectDual détecté : value=$v, tangent=$t")
        end
        return new(v, t)
    end
end


Base.zero(x::VectDual) = VectDual(zero(x.value), zero(x.tangent))
Base.zero(::Type{VectDual}) = VectDual(0.0, 0.0)
Base.one(x::VectDual) = VectDual(one(x.value), zero(x.tangent))
Base.size(d::VectDual) = size(d.value)
Base.length(d::VectDual) = length(d.value)
Base.size(d::VectDual, dim::Int) = size(d.value, dim)
Base.iterate(d::VectDual) = iterate(d.value)
Base.iterate(d::VectDual, state) = iterate(d.value, state)
Base.repeat(d::VectDual, counts...) =
    VectDual(repeat(d.value, counts...), repeat(d.tangent, counts...))

Base.adjoint(d::VectDual) =
    VectDual(adjoint(d.value), adjoint(d.tangent))

Base.transpose(d::VectDual) =
    VectDual(transpose(d.value), transpose(d.tangent))

# Addition
Base.:+(x::VectDual, y::VectDual) = VectDual(x.value .+ y.value, x.tangent .+ y.tangent)
Base.:+(x::VectDual, y::Union{AbstractArray,Number}) = VectDual(x.value .+ y, x.tangent)
Base.:+(x::Union{AbstractArray,Number}, y::VectDual) = VectDual(x .+ y.value, y.tangent)

Base.broadcasted(::typeof(+), x::VectDual, y::VectDual) = VectDual(x.value .+ y.value, x.tangent .+ y.tangent)
Base.broadcasted(::typeof(+), x::VectDual, y::Union{AbstractArray,Number}) = VectDual(x.value .+ y, x.tangent)
Base.broadcasted(::typeof(+), x::Union{AbstractArray,Number}, y::VectDual) = VectDual(x .+ y.value, y.tangent)

# Soustraction
Base.:-(x::VectDual, y::VectDual) = VectDual(x.value .- y.value, x.tangent .- y.tangent)
Base.:-(x::VectDual, y::Union{AbstractArray,Number}) = VectDual(x.value .- y, x.tangent)
Base.:-(x::Union{AbstractArray,Number}, y::VectDual) = VectDual(x .- y.value, .-y.tangent)
Base.:-(x::VectDual) = VectDual(-x.value, -x.tangent)

Base.broadcasted(::typeof(-), x::VectDual, y::VectDual) = VectDual(x.value .- y.value, x.tangent .- y.tangent)
Base.broadcasted(::typeof(-), x::VectDual, y::Union{AbstractArray,Number}) = VectDual(x.value .- y, x.tangent)
Base.broadcasted(::typeof(-), x::Union{AbstractArray,Number}, y::VectDual) = VectDual(x .- y.value, .-y.tangent)

# Multiplication matricielle
Base.:*(A::AbstractMatrix, x::VectDual) = VectDual(A * x.value, A * x.tangent)
Base.:*(x::VectDual, A::AbstractMatrix) = VectDual(x.value * A, x.tangent * A)
Base.:*(x::VectDual, y::VectDual) = VectDual(x.value * y.value, x.tangent * y.value + x.value * y.tangent)

# Multiplication scalaire
Base.:*(α::Number, x::VectDual) = VectDual(α * x.value, α * x.tangent)
Base.:*(x::VectDual, α::Number) = VectDual(x.value * α, x.tangent * α)

# Multiplication élément par élément
Base.broadcasted(::typeof(*), x::VectDual, y::VectDual) = VectDual(x.value .* y.value, x.tangent .* y.value .+ x.value .* y.tangent)
Base.broadcasted(::typeof(*), x::VectDual, y::Union{AbstractArray,Number}) = VectDual(x.value .* y, x.tangent .* y)
Base.broadcasted(::typeof(*), x::Union{AbstractArray,Number}, y::VectDual) = VectDual(x .* y.value, x .* y.tangent)

# Division
Base.:/(x::VectDual, α::Number) = VectDual(x.value / α, x.tangent / α)

Base.broadcasted(::typeof(/), x::VectDual, y::VectDual) = VectDual(
    x.value ./ y.value,
    (x.tangent .* y.value .- x.value .* y.tangent) ./ (y.value .^ 2)
)
Base.broadcasted(::typeof(/), x::VectDual, y::Union{AbstractArray,Number}) = VectDual(x.value ./ y, x.tangent ./ y)
Base.broadcasted(::typeof(/), x::Union{AbstractArray,Number}, y::VectDual) = VectDual(
    x ./ y.value,
    .-x .* y.tangent ./ (y.value .^ 2)
)

# Puissance
Base.broadcasted(::typeof(^), x::VectDual, n::Number) = VectDual(
    x.value .^ n,
    n .* (x.value .^ (n - 1)) .* x.tangent
)

Base.broadcasted(::typeof(Base.literal_pow), ::typeof(^), x::VectDual, ::Val{n}) where {n} = 
    Base.broadcasted(^, x, n)

# Tanh
Base.broadcasted(::typeof(tanh), x::VectDual) = VectDual(
    tanh.(x.value),
    (1 .- tanh.(x.value).^2) .* x.tangent
)

# ReLU
Base.broadcasted(::typeof(relu), x::VectDual) = VectDual(
    max.(x.value, 0.0),
    @. ifelse(x.value >= 0.0, 1.0, 0.0) * x.tangent
)

# Exponential
Base.broadcasted(::typeof(exp), x::VectDual) = VectDual(
    exp.(x.value),
    exp.(x.value) .* x.tangent
)

# Logarithm
Base.broadcasted(::typeof(log), x::VectDual) = VectDual(
    log.(x.value),
    x.tangent ./ x.value
)

# Sum
Base.sum(x::VectDual) = VectDual(sum(x.value), sum(x.tangent))

Base.sum(x::VectDual; dims=2) = VectDual(
    Base.sum(x.value, dims=dims),
    Base.sum(x.tangent, dims=dims)
)

# Maximum
Base.maximum(x::VectDual; dims=2) = begin
    mx = Base.maximum(x.value, dims=dims)
    mask = x.value .== mx
    VectDual(mx, Base.sum(mask .* x.tangent, dims=dims))
end

# Softmax
function softmax(x::VectDual)
    mx = Base.maximum(x.value, dims=2)
    exps = exp.(x.value .- mx)
    sums = Base.sum(exps, dims=2)
    s = exps ./ sums
    # Jacobien: s .* (tangent - sum(s .* tangent))
    tangent_out = s .* (x.tangent .- sum(s .* x.tangent, dims=2))
    VectDual(s, tangent_out)
end


function Base.:*(A::Flatten{<:VectDual}, X::AbstractMatrix)
    Flatten(map(a -> a * X, A.components))
end

function Base.:*(X::AbstractMatrix, A::Flatten{<:VectDual})
    Flatten(map(a -> X * a, A.components))
end

function flatten_to_vector(x::Flatten)
    return reduce(vcat, [vec(comp) for comp in x.components])
end


Base.ndims(::Type{VectDual}) = 0  
Base.ndims(x::VectDual) = ndims(x.value)  

function create_basis_vector(x::Flatten, global_idx::Int)
    result = zero(x)
    offset = 0
    for i in eachindex(x.components)
        comp_len = length(x.components[i])
        if global_idx <= offset + comp_len
            local_idx = global_idx - offset
            flat_comp = vec(result.components[i])
            flat_comp[local_idx] = 1.0
            result.components[i] .= reshape(flat_comp, size(x.components[i]))
            return result
        end
        offset += comp_len
    end
    error("Index out of bounds: $global_idx > $(offset)")
end


function pushforward(f, x::Flatten, tx::Flatten)
    # Créer un Flatten de VectDual (forward mode)
    dual_input = Flatten([
        VectDual(x.components[i], tx.components[i])
        for i in eachindex(x.components)
    ])

    # forward prop
    y = f(dual_input)
	return Flatten([c.tangent for c in y.components])
end

# i est un index global 
function jacobian_column(f, x::Flatten, i::Integer)
    basis = create_basis_vector(x, i)
    return pushforward(f, x, basis)
end
function jacobian(f, x::Flatten)
    n = length(x) 
    cols = [jacobian_column(f, x, i) for i in 1:n]
    return reduce(hcat, [flatten_to_vector(col) for col in cols])
end

function hessian(f, x)
    return jacobian(z -> gradient(f, z), x)
end

end