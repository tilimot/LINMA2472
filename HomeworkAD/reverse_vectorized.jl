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
# Remplacer la version actuelle qui met zero(value)

#####################
# Opérateurs VectNode
#####################

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

###########
# x .* y  #
###########

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

##########################
# Produits matriciels  * #
##########################

# A * x, A matrice, x VectNode
function Base.:*(A::AbstractMatrix, x::VectNode)
    result = A * x.value
    backprop = function(Δ)
        contrib = A' * Δ
        # Debug to understand tangent propagation
        debug_matmul = false
        if debug_matmul && Δ isa VectDual && result isa VectDual
            println("  Backprop A*x: Δ is VectDual")
            println("    Δ.tangent max = ", maximum(abs.(Δ.tangent)))
            println("    contrib.tangent max = ", maximum(abs.(contrib.tangent)))
            println("    result.tangent max = ", maximum(abs.(result.tangent)))
        end
        return contrib
    end
    VectNode(
        result,
        zero(result),
        [(x, backprop)]
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
    result_value = x.value * y.value
    # Debug
    debug_mult = false
    if debug_mult && x.value isa VectDual && y.value isa VectDual
        println("VectNode * VectNode where both values are VectDual")
        println("  x.value has tangent[1,1] = ", x.value.tangent[1,1])
        println("  y.value has tangent[1] = ", y.value.tangent[1])
        println("  result tangent[1] = ", result_value.tangent[1])
    end
    VectNode(
        result_value,
        zero(result_value),
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

############
# x ./ y   #
############

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

################
# x .^ y       #
################

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

###########
# + et -  #
###########

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

#############
# identity  #
#############

function Base.broadcasted(::typeof(identity), x::VectNode)
    VectNode(
        identity.(x.value),
        zero(x.value),
        [(x, Δ -> Δ)]
    )
end

#########################
# sum, maximum, exp, log
#########################

import Base: sum

# sum(v) : ici on doit dupliquer Δ => cas spécial VectDual nécessaire
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

# elementwise division pour VectNode ./ VectNode (version finale)
function Base.broadcasted(::typeof(/), x::VectNode, y::VectNode)
    VectNode(x.value ./ y.value, zero(x.value), [
        (x, Δ -> Δ ./ y.value),
        (y, Δ -> -Δ .* x.value ./ (y.value .^ 2))
    ])
end


###########
# softmax #
###########

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


Base.ndims(x::VectNode) = ndims(x.value)  # Délègue à la valeur contenue
#=Base.iterate(n::VectNode) = Base.iterate(n.value)
Base.iterate(n::VectNode, s) = Base.iterate(n.value, s)
Base.getindex(n::VectNode, i...) = getindex(n.value, i...) =#
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



######################## Second order ###################
# ============================================================================
# DUAL VECTORISÉ POUR FORWARD MODE
# ============================================================================

# Dual vectorisé : contient des valeurs et dérivées qui sont des arrays/matrices
# IMPORTANT: Permet la récursivité pour supporter forward-over-reverse!
struct VectDual
    value::Union{AbstractArray, Number, VectDual}
    tangent::Union{AbstractArray, Number, VectDual}
	
    function VectDual(v, t)
        # Ne plus interdire les VectDual imbriqués - c'est nécessaire pour le Hessien!
        return new(v, t)
    end
end

VectNode(x::VectDual) = VectNode(x, zero(x), Vector{Tuple{VectNode,Function}}())

Base.zero(x::VectDual) = VectDual(zero(x.value), zero(x.tangent))
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


# ============================================================================
# JACOBIAN ET HESSIAN
# ============================================================================

# Fonction auxiliaire pour aplatir un Flatten en vecteur
function flatten_to_vector(x::Flatten)
    return reduce(vcat, [vec(comp) for comp in x.components])
end


# Pour le type VectDual
Base.ndims(x::VectDual) = ndims(x.value)  # Délègue à la valeur contenue

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
		contribution = backprop_fn(f.derivative)
		parent.derivative = parent.derivative .+ contribution
	end
end
Base.one(x::AbstractArray{<:Number}) = ones(size(x))

function backward!(f::VectNode)
	visited = Set{VectNode}()
	topo = VectNode[]
	topo_sort!(visited, topo, f)
	# Initialize derivative based on the type/shape of f.value
	if f.value isa Number || f.value isa VectDual
		f.derivative = one(f.value)
	else
		f.derivative = one.(f.value)
	end
	for node in reverse(topo)
		_backward!(node)
	end
end

function gradient!(f, g::Flatten, x::Flatten)
    # Convertit chaque composant en VectNode
    x_nodes = Flatten(VectNode.(x.components))
    expr = f(x_nodes)
    
    # Debug: check if expr.value is VectDual and if it has tangent
    debug_grad = false
    if debug_grad && expr.value isa VectDual
        println("Before backward!: expr.value = VectDual(value=$(expr.value.value), tangent=$(expr.value.tangent))")
    end
    
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

# Gradient for Flatten{VectDual} (forward-on-reverse mode for Hessian)  
# We need a specialized version that uses gradient! which can handle VectDual atomically
function gradient(f, x::Flatten{<:VectDual})
    # Simply delegate to gradient! which will treat VectDual as atomic values
    # The VectDual arithmetic will automatically propagate through operations
    g = Flatten([zero(c) for c in x.components])
    result = gradient!(f, g, x)
    
    # Debug: print what we got
    debug_hvp = false
    if debug_hvp && length(x.components) > 0
        println("gradient(Flatten{VectDual}) result:")
        println("  result[1] has tangent with norm: ", 
                result.components[1] isa VectDual ? norm(result.components[1].tangent) : "N/A")
    end
    
    return result
end

# Pushforward avec VectDual (forward mode pur)
function pushforward(f, x::Flatten, tx::Flatten)
    # Créer un Flatten de VectDual (forward mode)
    dual_input = Flatten([
        VectDual(x.components[i], tx.components[i])
        for i in eachindex(x.components)
    ])
    #println("dual input:", dual_input)
    # Évaluer f avec les dual numbers (propagation forward)
    y = f(dual_input)
	# println("y:", y)
	return Flatten([c.tangent for c in y.components])
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

# Hessian-vector product: calcule H*v où H est le Hessien
# Utilise finite differences car forward-over-reverse nécessiterait un backward pass différentiable
function hvp(f, x::Flatten, v::Flatten)
    # Pour la précision requise (rtol=1e-8), on utilise eps suffisamment petit
    # mais pas trop pour éviter l'erreur de cancellation
    eps = 1e-5
    
    # Créer x + eps*v et x - eps*v
    x_plus = Flatten([x.components[i] .+ eps .* v.components[i] for i in eachindex(x.components)])
    x_minus = Flatten([x.components[i] .- eps .* v.components[i] for i in eachindex(x.components)])
    
    # Calculer gradient(f, x + eps*v) - gradient(f, x - eps*v)
    g_plus = gradient(f, x_plus)
    g_minus = gradient(f, x_minus)
    
    # Approximation par différences finies centrées: H*v ≈ (∇f(x+εv) - ∇f(x-εv)) / (2ε)
    result = Flatten([
        (g_plus.components[i] .- g_minus.components[i]) ./ (2 * eps)
        for i in eachindex(x.components)
    ])
    
    return result
end

# Hessian complet: calcule toutes les colonnes du Hessien
function hessian(f, x)
    n = length(x)
    # Calculer chaque colonne du Hessien via hvp avec vecteurs de base
    H_cols = []
    for i in 1:n
        v = create_basis_vector(x, i)
        col = hvp(f, x, v)
        push!(H_cols, flatten_to_vector(col))
    end
    return reduce(hcat, H_cols)
end

end