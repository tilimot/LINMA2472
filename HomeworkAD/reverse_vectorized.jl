include("flatten.jl")
import ..Forward: Dual, pushforward, jacobian, hessian


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


Base.ndims(::Type{VectNode}) = 0  # Car VectNode est un type wrapper
Base.ndims(x::VectNode) = ndims(x.value)  # Délègue à la valeur contenue
Base.iterate(n::VectNode) = Base.iterate(n.value)
Base.iterate(n::VectNode, s) = Base.iterate(n.value, s)
Base.getindex(n::VectNode, i...) = getindex(n.value, i...)
Base.eachindex(n::VectNode) = eachindex(n.value)
Base.length(n::VectNode) = length(n.value)
Base.size(n::VectNode) = size(n.value)

# Pour la fonction ones avec une matrice
Base.ones(x::Vector{Float64}) = fill(1.0, size(x))
Base.ones(x::Matrix{Float64}) = fill(1.0, size(x))
Base.ones(x::Matrix{Vector}) = fill(1.0, size(x))
# Pour ones avec un VectNode
Base.ones(x::VectNode) = VectNode(ones(x.value), zero(x.value), Tuple{VectNode, Function}[])

# Pour ones avec une taille donnée (si nécessaire)
Base.ones(dims::Tuple{Int,Int}) = fill(1.0, dims)

# Implémentation de copyto! pour le broadcasting avec VectNode
function Base.copyto!(dest::VectNode, bc::Broadcast.Broadcasted{<:Any})
    # Copie les valeurs
    copyto!(dest.value, bc)
    
    # Réinitialise les dérivées et parents
    dest.derivative = zero(dest.value)
    empty!(dest.parents)
    
    # Si le broadcast implique des VectNodes, nous devons ajouter les relations de parenté
    args = bc.args
    for arg in args
        if arg isa VectNode
            # La dérivée dépend de l'opération de broadcast
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


function onehot(x::Flatten, i::Integer)
    tx = zero(x)                     # même structure que x, rempli de zéros
    tx.components[i] .= ones(tx.components[i])
    return tx
end


######################## Second order ###################
# ============================================================================
# DUAL VECTORISÉ POUR FORWARD MODE
# ============================================================================

# Dual vectorisé : contient des valeurs et dérivées qui sont des arrays/matrices
struct VectDual
    value::Union{AbstractArray, Number}
    tangent::Union{AbstractArray, Number}
    
    # Constructeur interne pour éviter la récursion infinie
    VectDual(v, t) = new(v, t)
end

Base.broadcastable(d::VectDual) = Ref(d)
Base.zero(::VectDual) = VectDual(0.0, 0.0)
Base.zero(::Type{VectDual}) = VectDual(0.0, 0.0)
Base.one(::VectDual) = VectDual(1.0, 0.0)
Base.size(d::VectDual) = size(d.value)
Base.length(d::VectDual) = length(d.value)

# ============================================================================
# OPÉRATIONS ARITHMÉTIQUES POUR VectDual
# ============================================================================

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

# ============================================================================
# FONCTIONS SPÉCIALES
# ============================================================================

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

# Identity
Base.broadcasted(::typeof(identity), x::VectDual) = VectDual(identity.(x.value), x.tangent)

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

# ============================================================================
# SUPPORT POUR FLATTEN
# ============================================================================

function Base.:*(A::Flatten{<:VectDual}, X::AbstractMatrix)
    Flatten(map(a -> a * X, A.components))
end

function Base.:*(X::AbstractMatrix, A::Flatten{<:VectDual})
    Flatten(map(a -> X * a, A.components))
end

# ============================================================================
# PUSHFORWARD CORRIGÉ (FORWARD MODE)
# ============================================================================

# Fonction auxiliaire : extraire les valeurs d'un Flatten de VectDual
function extract_values(x::Flatten)
    if eltype(x.components) <: VectDual
        return Flatten([comp.value for comp in x.components])
    else
        return x
    end
end

# Fonction auxiliaire : extraire les tangentes d'un Flatten de VectDual
function extract_tangents(x::Flatten)
    return Flatten([comp.tangent for comp in x.components])
end

# Pushforward avec VectDual (forward mode pur)
function pushforward(f, x::Flatten, tx::Flatten)
    # Créer un Flatten de VectDual (forward mode)
    dual_input = Flatten([
        VectDual(x.components[i], tx.components[i])
        for i in eachindex(x.components)
    ])
    
    # Évaluer f avec les dual numbers (propagation forward)
    y = f(dual_input)
    
    # Extraire les tangentes du résultat
    if y isa VectDual
        return y.tangent
    elseif y isa Flatten
        # Vérifier si les composants sont des VectDual
        if !isempty(y.components) && y.components[1] isa VectDual
            return extract_tangents(y)
        else
            # Si c'est déjà un Flatten de valeurs numériques, le retourner tel quel
            return y
        end
    else
        error("Unexpected output type from f: $(typeof(y))")
    end
end

# ============================================================================
# JACOBIAN ET HESSIAN
# ============================================================================

# Fonction auxiliaire pour aplatir un Flatten en vecteur
function flatten_to_vector(x::Flatten)
    return reduce(vcat, [vec(comp) for comp in x.components])
end

# Fonction auxiliaire : créer un vecteur de base e_i pour un Flatten
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

# Wrapper pour gradient qui accepte VectDual et retourne VectDual
function gradient_dual(f, x::Flatten)
    # Extraire les valeurs pour le gradient (reverse mode sur valeurs)
    x_values = extract_values(x)
    
    # Calculer le gradient (reverse mode) - retourne un Flatten de valeurs numériques
    grad = gradient(f, x_values)
    
    # Si x contient des VectDual, créer un Flatten de VectDual pour le gradient
    if !isempty(x.components) && x.components[1] isa VectDual
        # Propager les tangentes à travers le gradient
        tangents = extract_tangents(x)
        
        # Pour chaque composant du gradient, créer un VectDual
        return Flatten([
            VectDual(grad.components[i], tangents.components[i])
            for i in eachindex(grad.components)
        ])
    else
        return grad
    end
end

# Jacobian : colonne i (i est un index GLOBAL dans le vecteur aplati)
function jacobian_column(f, x::Flatten, i::Integer)
    basis = create_basis_vector(x, i)
    return pushforward(f, x, basis)
end

# Jacobian complète
function jacobian(f, x::Flatten)
    n = length(x)  # Nombre total d'éléments dans le vecteur aplati
    cols = [jacobian_column(f, x, i) for i in 1:n]
    # Aplatir chaque colonne et concaténer
    return reduce(hcat, [flatten_to_vector(col) for col in cols])
end

# Hessian = Jacobian du gradient (forward-on-reverse)
function hessian(f, x)
    # La fonction passée à jacobian doit gérer VectDual
    return jacobian(z -> gradient_dual(f, z), x)
end

# Hessian-vector product
function hvp(f, x, v)
    # v est un Flatten, on calcule H*v via forward-on-reverse
    grad_at_v = pushforward(z -> gradient_dual(f, z), x, v)
    return flatten_to_vector(grad_at_v)
end
end