text = read("HomeworkAD/input.txt", String)   # ou autre

LabAD = joinpath(dirname(@__DIR__), "LabAD")

include(joinpath(LabAD, "test", "test.jl"))

# Reference implementation we test against
include(joinpath(LabAD, "solution", "forward.jl"))
include(joinpath(@__DIR__, "train.jl"))
## First order
include(joinpath(@__DIR__, "reverse_vectorized.jl"))

chars = sort(unique(text))

### ENCODING / DECODING

stoi = Dict(ch => i for (i, ch) in enumerate(chars))  # Char -> Int
itos = Dict(i => ch for (i, ch) in enumerate(chars))  # Int  -> Char


encode(s::AbstractString) = [stoi[c] for c in s]          # String -> Vector{Int}
decode(v::AbstractVector{<:Integer}) = join(itos[i] for i in v)  # Vector{Int} -> String

# Encoder tout le texte

# convertir en vecteur d'entiers (déjà fait par encode)
data = encode(text)

# index du split 90%
n = floor(Int, 0.9 * length(data))

# splits
train_data = data[1:n]
val_data   = data[n+1:end]


### Sample random chunks and sending a chunk at a time to the transformer
### The maximym length = block_size

block_size = 8
train_data[1:block_size+1]

# x = train_data[:block_size]
x = train_data[1:block_size]


y = train_data[2:block_size+1]

for t in 1:block_size
    context = x[1:t]          # x[:t+1] en Python
    target  = y[t]            # y[t]
    println("when input is $(context) the target: $(target)")
end

### Création des batch to process multiple chunks at the same time

using Random

Random.seed!(1337)

batch_size = 4      # combien de séquences en parallèle
block_size = 8      # longueur du contexte

function get_batch(split)
    data = split == "train" ? train_data : val_data

    # tirage aléatoire des positions de départ
    ix = rand(1:length(data)-block_size, batch_size)

    # créer la matrice x (batch_size × block_size)
    x = [data[i + t] for i in ix, t in 0:block_size-1]

    # créer la matrice y (batch_size × block_size)
    y = [data[i + t + 1] for i in ix, t in 0:block_size-1]

    return x, y
end

# tester
xb, yb = get_batch("train")

println("inputs:")
println(xb, size(xb))

println("targets:")
println(size(yb))
println(yb)

println("-----")

# reproduction de la boucle Python
for b in 1:batch_size        # dimension batch
    for t in 1:block_size    # dimension temps
        context = xb[b, 1:t]
        target  = yb[b, t]
        println("when input is $(context) the target: $(target)")
    end
end

using Flux
using Flux: onehotbatch
using Random
using StatsBase
Random.seed!(1337)

# exemple d'appel, comme en PyTorch
vocab_size = length(chars)


# xb et yb : matrice d'Int (B,T) avec valeurs dans 1:vocab_size
# B = batch_size
# T = longueur de chaque séquence
println("inputs:")
println("xb", xb)
println("targets:")
println("yb", yb)
println("size of xb: ", size(xb))
println("size of yb: ", size(yb))
println("-----")


# C'est pas bien car total random...
# On regarde que le dernier charac...
# The history is not used

function bigram_activation(Wflat::Flatten, Xb)
    W = Wflat.components[1]          # C×C (Float32 ou VectNode)


    idx = vec(Xb)                    # longueur N = B*T
    N = length(idx)

    # one-hot (Float32 constants)
    OH = zeros(Float32, C, N)
    @inbounds for (j, i) in enumerate(idx)
        OH[i, j] = 1f0
    end

    # logits = W * onehot
    # si W est Matrix{VectNode}, le produit garde le graphe
    return W * OH                    # C×N = C×(B*T)
end


##############################
# Use trained W for generation
##############################


# Petit modèle manuel qui utilise W
struct ManualBigram
    W::Matrix{Float32}  # C×C
end

function (m::ManualBigram)(idx)
    # idx : (B,T) Int
    idx_vec = vec(idx)
    logits2d = m.W[:, idx_vec]          # C×(B*T)
    B, T = size(idx)
    return reshape(logits2d, C, B, T)   # C×B×T
end

function generate(m::ManualBigram, idx, max_new_tokens)
    for _ in 1:max_new_tokens
        logits = m(idx)                 # C×B×T
        last_logits = logits[:, :, end] # C×B
        last_logits = permutedims(last_logits, (2,1))  # B×C

        probs = softmax(last_logits)    # B×C
        Bsize, Csize = size(probs)
        idx_next = similar(idx[:,1:1])  # B×1

        for b in 1:Bsize
            p = probs[b, :]
            idx_next[b, 1] = sample(1:Csize, Weights(p))
        end

        idx = hcat(idx, idx_next)
    end
    return idx
end


C = vocab_size
batch_size = 32

xb, yb = get_batch("train")
W = 0.01f0 .* randn(Float32, C, C)
W = Flatten([W])
targets1d = vec(yb)         # longueur B*T
y_onehot = onehotbatch(targets1d, 1:C)  # C × (B*T)
num_iters = 1
L = loss(mse, bigram_activation, xb, y_onehot)

losses, Wtrained = train!(VectReverse.gradient!, L, W, num_iters)


m_trained = ManualBigram(Wtrained.components[1])
max_new_tokens = 200
idx_gen = generate(m_trained, xb, max_new_tokens)
println(decode(idx_gen[1, :]))

##############################
# ATTENTION MECHANISM
##############################

"""
    scaled_dot_product_attention(Q, K, V, mask=nothing)

Implémente l'attention scaled dot-product compatible avec VectReverse.
- Q : queries (d_k × N) où N = B*T
- K : keys (d_k × N)
- V : values (d_v × N)
- mask : masque optionnel pour l'attention causale

Retourne : (d_v × N)
"""
function scaled_dot_product_attention(Q, K, V, mask=nothing)
    # Q: d_k × N
    # K: d_k × N  
    # V: d_v × N
    
    d_k = size(Q, 1)
    
    # Scores d'attention: K^T * Q (N × N)
    # On transpose K pour obtenir N × d_k, puis multiplie par Q (d_k × N)
    # Pour transposer K quand c'est un VectNode, on doit faire attention
    # K est d_k × N, donc K' sera automatiquement N × d_k
    scores = K' * Q  # (N × d_k) * (d_k × N) = N × N
    
    # Scaling
    scores = scores ./ Float32(sqrt(d_k))
    
    # Application du masque si fourni (pour attention causale)
    if mask !== nothing
        # mask devrait être N × N avec -Inf pour les positions futures
        scores = scores .+ mask
    end
    
    # Softmax sur la dernière dimension (chaque ligne)
    attn_weights = softmax(scores)  # N × N
    
    # Application de l'attention aux valeurs
    # V est d_v × N, attn_weights est N × N
    # On veut: d_v × N
    output = V * attn_weights'  # (d_v × N) * (N × N) = d_v × N
    
    return output
end


"""
    SingleHeadAttention

Structure pour une tête d'attention simple avec ses paramètres.
Compatible avec Flatten pour l'entraînement via VectReverse.grad.
"""
struct SingleHeadAttention
    d_model::Int      # dimension d'entrée/sortie
    d_k::Int          # dimension des queries/keys
    d_v::Int          # dimension des values
end

"""
    attention_forward(params::Flatten, X, model_config)

Fonction forward pour l'attention, compatible avec VectReverse.
- params : Flatten([Wq, Wk, Wv, Wo]) où chaque matrice est (d × d_model)
- X : matrice d'embedding (d_model × N) où N = B*T
- model_config : tuple (d_model, d_k, d_v, block_size, batch_size)

Retourne : (d_model × N)
"""
function attention_forward(params::Flatten, X, model_config)
    d_model, d_k, d_v, block_size, batch_size = model_config
    
    Wq, Wk, Wv, Wo = params.components[1], params.components[2], 
                     params.components[3], params.components[4]
    
    # Projections linéaires
    Q = Wq * X  # (d_k × d_model) * (d_model × N) = d_k × N
    K = Wk * X  # (d_k × d_model) * (d_model × N) = d_k × N  
    V = Wv * X  # (d_v × d_model) * (d_model × N) = d_v × N
    
    # Créer le masque causal pour empêcher de voir le futur
    N = size(X, 2)  # B*T
    mask = create_causal_mask(block_size, batch_size)
    
    # Attention
    attn_output = scaled_dot_product_attention(Q, K, V, mask)  # d_v × N
    
    # Projection de sortie
    output = Wo * attn_output  # (d_model × d_v) * (d_v × N) = d_model × N
    
    return output
end


"""
    create_causal_mask(T, B)

Crée un masque causal pour empêcher l'attention de voir les positions futures.
Retourne une matrice (B*T × B*T) avec -Inf pour les positions futures.
"""
function create_causal_mask(T, B)
    N = B * T
    mask = zeros(Float32, N, N)
    
    # Pour chaque batch
    for b in 0:B-1
        offset = b * T
        # Pour chaque position dans la séquence
        for i in 1:T
            for j in (i+1):T
                # Masquer les positions futures
                mask[offset + i, offset + j] = -1f10
            end
        end
    end
    
    return mask
end


"""
    initialize_attention_params(d_model, d_k, d_v)

Initialise les paramètres de l'attention (Wq, Wk, Wv, Wo) avec Xavier init.
Retourne un Flatten([Wq, Wk, Wv, Wo]).
"""
function initialize_attention_params(d_model, d_k, d_v)
    # Xavier initialization
    scale_qk = sqrt(2.0f0 / (d_model + d_k))
    scale_v = sqrt(2.0f0 / (d_model + d_v))
    scale_o = sqrt(2.0f0 / (d_v + d_model))
    
    Wq = scale_qk .* randn(Float32, d_k, d_model)
    Wk = scale_qk .* randn(Float32, d_k, d_model)
    Wv = scale_v .* randn(Float32, d_v, d_model)
    Wo = scale_o .* randn(Float32, d_model, d_v)
    
    return Flatten([Wq, Wk, Wv, Wo])
end


"""
    attention_activation(Wflat::Flatten, Xb, embedding_table, config)

Fonction d'activation complète incluant embedding + attention.
Compatible avec VectReverse.grad pour l'entraînement.

- Wflat : Flatten([Wq, Wk, Wv, Wo, embedding_table])
- Xb : matrice d'indices (B × T)
- config : (vocab_size, d_model, d_k, d_v, block_size, batch_size)
"""
function attention_activation(Wflat::Flatten, Xb, config)
    vocab_size, d_model, d_k, d_v, block_size, batch_size = config
    
    # Extraire les paramètres
    # Les 4 premières composantes sont Wq, Wk, Wv, Wo
    # La 5ème est la table d'embedding
    attention_params = Flatten(Wflat.components[1:4])
    embedding_table = Wflat.components[5]  # (d_model × vocab_size)
    
    # Embedding : convertir indices en vecteurs
    idx = vec(Xb)  # longueur N = B*T
    N = length(idx)
    
    # One-hot encoding
    OH = zeros(Float32, vocab_size, N)
    @inbounds for (j, i) in enumerate(idx)
        OH[i, j] = 1f0
    end
    
    # Lookup dans la table d'embedding
    X = embedding_table * OH  # (d_model × vocab_size) * (vocab_size × N) = d_model × N
    
    # Attention forward
    model_config = (d_model, d_k, d_v, block_size, batch_size)
    output = attention_forward(attention_params, X, model_config)
    
    return output
end


##############################
# EXEMPLE D'UTILISATION
##############################

"""
Exemple d'entraînement avec attention
"""
function train_attention_model()
    # Hyperparamètres
    vocab_size = length(chars)
    d_model = 32      # dimension des embeddings
    d_k = 8           # dimension queries/keys
    d_v = 8           # dimension values
    block_size = 8
    batch_size = 32
    
    config = (vocab_size, d_model, d_k, d_v, block_size, batch_size)
    
    # Initialiser les paramètres
    attention_params = initialize_attention_params(d_model, d_k, d_v)
    embedding_table = 0.01f0 .* randn(Float32, d_model, vocab_size)
    
    # Combiner tous les paramètres
    W = Flatten([attention_params.components..., embedding_table])
    
    # Préparer les données
    xb, yb = get_batch("train")
    
    # Pour la loss, on doit projeter la sortie vers vocab_size
    # On ajoute une matrice de projection finale
    W_output = 0.01f0 .* randn(Float32, vocab_size, d_model)
    W_full = Flatten([W.components..., W_output])
    
    # Fonction de loss
    function attention_loss(W_params)
        # Attention output: d_model × N
        attn_out = attention_activation(Flatten(W_params.components[1:5]), xb, config)
        
        # Projection vers logits: vocab_size × N
        W_out = W_params.components[6]
        logits = W_out * attn_out
        
        # Loss MSE avec one-hot targets
        targets1d = vec(yb)
        y_onehot = onehotbatch(targets1d, 1:vocab_size)
        
        diff = logits .- y_onehot
        return sum(diff .^ 2) / length(targets1d)
    end
    
    L(w) = attention_loss(w)
    
    # Entraînement
    println("Entraînement du modèle avec attention...")
    losses, W_trained = train!(VectReverse.gradient!, L, W_full, 
                               num_iters=1, 
                               rule=Optimisers.Adam(0.001))
    
    println("Loss finale: ", losses[end])
    
    return W_trained, config
end

w, c = train_attention_model()

##############################
# GÉNÉRATION AVEC ATTENTION
##############################

"""
    ManualAttention

Structure pour la génération avec un modèle d'attention entraîné.
Contient tous les paramètres nécessaires (Wq, Wk, Wv, Wo, embedding, output).
"""
struct ManualAttention
    Wq::Matrix{Float32}
    Wk::Matrix{Float32}
    Wv::Matrix{Float32}
    Wo::Matrix{Float32}
    embedding_table::Matrix{Float32}
    W_output::Matrix{Float32}
    config::Tuple{Int,Int,Int,Int,Int,Int}  # (vocab_size, d_model, d_k, d_v, block_size, batch_size)
end

"""
    ManualAttention(W_trained::Flatten, config)

Constructeur à partir des paramètres entraînés.
"""
function ManualAttention(W_trained::Flatten, config)
    return ManualAttention(
        W_trained.components[1],  # Wq
        W_trained.components[2],  # Wk
        W_trained.components[3],  # Wv
        W_trained.components[4],  # Wo
        W_trained.components[5],  # embedding_table
        W_trained.components[6],  # W_output
        config
    )
end

"""
    (m::ManualAttention)(idx)

Forward pass du modèle d'attention (sans VectNode, pour la génération).
- idx : matrice (B × T) d'indices de tokens
Retourne : logits de forme (vocab_size × B × T)
"""
function (m::ManualAttention)(idx)
    vocab_size, d_model, d_k, d_v, block_size_config, batch_size_config = m.config
    
    B, T = size(idx)
    
    # Embedding
    idx_vec = vec(idx)  # B*T
    N = length(idx_vec)
    
    OH = zeros(Float32, vocab_size, N)
    @inbounds for (j, i) in enumerate(idx_vec)
        OH[i, j] = 1f0
    end
    
    X = m.embedding_table * OH  # d_model × N
    
    # Projections Q, K, V
    Q = m.Wq * X  # d_k × N
    K = m.Wk * X  # d_k × N
    V = m.Wv * X  # d_v × N
    
    # Attention avec masque causal
    scores = K' * Q ./ Float32(sqrt(d_k))  # N × N
    
    # Masque causal
    mask = create_causal_mask(T, B)
    scores = scores .+ mask
    
    # Softmax
    attn_weights = softmax(scores)  # N × N
    
    # Application aux valeurs
    attn_out = V * attn_weights'  # d_v × N
    
    # Projection de sortie
    output = m.Wo * attn_out  # d_model × N
    
    # Projection vers logits
    logits = m.W_output * output  # vocab_size × N
    
    # Reshape en (vocab_size × B × T)
    return reshape(logits, vocab_size, B, T)
end

"""
    generate(m::ManualAttention, idx, max_new_tokens; temperature=1.0)

Génère de nouveaux tokens avec le modèle d'attention.
- m : modèle ManualAttention
- idx : contexte initial (B × T)
- max_new_tokens : nombre de tokens à générer
- temperature : contrôle l'aléatoire (1.0 = normal, <1 = plus déterministe, >1 = plus aléatoire)

Retourne : séquence augmentée (B × (T + max_new_tokens))
"""
function generate(m::ManualAttention, idx, max_new_tokens; temperature=1.0)
    vocab_size, d_model, d_k, d_v, block_size, batch_size_config = m.config
    
    for _ in 1:max_new_tokens
        # Tronquer au block_size si nécessaire (attention causale)
        T_current = size(idx, 2)
        if T_current > block_size
            idx_cond = idx[:, end-block_size+1:end]
        else
            idx_cond = idx
        end
        
        # Forward pass
        logits = m(idx_cond)  # vocab_size × B × T
        
        # Prendre les logits du dernier token
        last_logits = logits[:, :, end]  # vocab_size × B
        last_logits = permutedims(last_logits, (2, 1))  # B × vocab_size
        
        # Appliquer la température
        if temperature != 1.0
            last_logits = last_logits ./ temperature
        end
        
        # Softmax pour obtenir les probabilités
        probs = softmax(last_logits)  # B × vocab_size
        
        Bsize, Vsize = size(probs)
        idx_next = similar(idx[:, 1:1])  # B × 1
        
        # Échantillonner pour chaque élément du batch
        for b in 1:Bsize
            p = probs[b, :]
            idx_next[b, 1] = sample(1:Vsize, Weights(p))
        end
        
        # Ajouter le nouveau token
        idx = hcat(idx, idx_next)
    end
    
    return idx
end

"""
    generate_text(W_trained, config, prompt=""; max_new_tokens=100, temperature=1.0)

Fonction helper pour générer du texte à partir d'un prompt.
- W_trained : paramètres entraînés (Flatten)
- config : configuration du modèle
- prompt : texte de départ (vide par défaut)
- max_new_tokens : nombre de tokens à générer
- temperature : contrôle de l'aléatoire

Retourne : texte généré
"""
function generate_text(W_trained, config, prompt=""; max_new_tokens=100, temperature=1.0)
    vocab_size, d_model, d_k, d_v, block_size, batch_size_config = config
    
    # Créer le modèle
    m = ManualAttention(W_trained, config)
    
    # Encoder le prompt ou commencer avec un token aléatoire
    if isempty(prompt)
        # Commencer avec un token aléatoire
        idx = rand(1:vocab_size, 1, 1)
    else
        # Encoder le prompt
        encoded = encode(prompt)
        # Limiter au block_size si nécessaire
        if length(encoded) > block_size
            encoded = encoded[end-block_size+1:end]
        end
        idx = reshape(encoded, 1, :)  # 1 × T (batch size = 1)
    end
    
    # Générer
    idx_generated = generate(m, idx, max_new_tokens, temperature=temperature)
    
    # Décoder
    return decode(vec(idx_generated[1, :]))
end

# Après avoir entraîné le modèle
w, c = train_attention_model()

# Créer le modèle de génération
m_attention = ManualAttention(w, c)

# Option 1: Générer à partir d'un batch existant
xb_test, _ = get_batch("val")
generated_idx = generate(m_attention, xb_test[1:1, :], 200)
println("Génération à partir du batch:")
println(decode(vec(generated_idx[1, :])))

# Option 2: Générer à partir d'un prompt (plus simple)
println("\nGénération avec prompt vide:")
text1 = generate_text(w, c, "", max_new_tokens=200, temperature=1.0)
println(text1)

# Option 3: Générer à partir d'un prompt spécifique
if length(chars) > 0 && haskey(stoi, first(chars))
    prompt = string(first(chars))
    println("\nGénération avec prompt '$prompt':")
    text2 = generate_text(w, c, prompt, max_new_tokens=200, temperature=0.8)
    println(text2)
end