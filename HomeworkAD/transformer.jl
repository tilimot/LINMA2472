text = read("HomeworkAD/input.txt", String)   # ou autre
text = lowercase(text)

LabAD = joinpath(dirname(@__DIR__), "LabAD")

include(joinpath(LabAD, "test", "test.jl"))

# Reference implementation we test against
include(joinpath(LabAD, "solution", "forward.jl"))
include(joinpath(@__DIR__, "definitions.jl"))
include(joinpath(@__DIR__, "train.jl"))

## First order
include(joinpath(@__DIR__, "reverse_vectorized.jl"))

import Base.size

function Base.size(v::Main.VectReverse.VectNode, d::Int64)
    return size(v.value, d)
end

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

# tester
xb, yb = get_batch("train")


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

##############################
# Use trained W for generation
##############################

C = vocab_size
batch_size = 32

##############################
# ATTENTION MECHANISM
##############################

struct SingleHeadAttention
    d_model::Int      # dimension d'entrée/sortie
    d_k::Int          # dimension des queries/keys
    d_v::Int          # dimension des values
end


function attention_forward(params::Flatten, X, model_config)
    d_model, d_k, d_v, block_size, batch_size = model_config
    
    Wq, Wk, Wv, Wo = params.components[1], params.components[2], 
                     params.components[3], params.components[4]
    
    # Projections linéaires
    # Q = Wq * X  # (d_k × d_model) * (d_model × N) = d_k × N
    # K = Wk * X  # (d_k × d_model) * (d_model × N) = d_k × N  
    Q = Wq * X # (d_k × d_model) * (d_model × N) = d_k × N
    K = Wk * X # (d_k × d_model) * (d_model × N) = d_k × N
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


function attention_activation(Wflat::Flatten, Xb, config)
    vocab_size, d_model, d_k, d_v, block_size, batch_size = config
    
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
    
    # 1. Générer le PE (d_model × block_size)
    PE_base = positional_encoding(block_size, d_model)
    
    # 2. Répéter le PE pour chaque élément du batch
    PE_full = zeros(Float32, d_model, batch_size * block_size)
    for b in 0:batch_size-1
        # Copier le PE dans le bloc correspondant
        PE_full[:, (b * block_size + 1):((b + 1) * block_size)] = PE_base
    end

    # 3. Ajouter le PE à l'embedding d'entrée X
    X = X .+ PE_full # X est maintenant un VectNode si embedding_table est un VectNode

    # Attention forward
    model_config = (d_model, d_k, d_v, block_size, batch_size)
    output = attention_forward(attention_params, X, model_config)
    
    return output
end


function train_attention_model()
    # Hyperparamètres
    vocab_size = length(chars)
    d_model = 512      # dimension des embeddings
    d_k = 64           # dimension queries/keys
    d_v = 64           # dimension values
    block_size = 8
    batch_size = 32
    
    config = (vocab_size, d_model, d_k, d_v, block_size, batch_size)
    

    attention_params = initialize_attention_params(d_model, d_k, d_v)
    embedding_table = randn(Float32, d_model, vocab_size) .* sqrt(1.0f0 / d_model)
    
    W = Flatten([attention_params.components..., embedding_table])
    
    xb, yb = get_batch("train")
    
    # Pour la loss, on doit projeter la sortie vers vocab_size
    # On ajoute une matrice de projection finale
    W_output = 0.001f0 .* randn(Float32, vocab_size, d_model)
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
        
        probs = softmax(logits)
        log_probs = log.(probs .+ 1f-8) 
        
        log_target_prob = y_onehot .* log_probs 
        

        loss_sum = -sum(log_target_prob)
        
        # Normaliser par le nombre de tokens (N = B * T)
        return loss_sum / length(targets1d) # N est la taille totale du batch
    end
    
    L(w) = attention_loss(w)
    num_iters = 10
    println("Entraînement du modèle avec attention...")
    losses, W_trained = train!(VectReverse.gradient!, L, W_full, num_iters, 
                               )
    
    println("Loss finale: ", losses[end])
    
    return W_trained, config
end

w, c = train_attention_model()

##############################
# GÉNÉRATION AVEC ATTENTION
##############################


struct ManualAttention
    Wq::Matrix{Float32}
    Wk::Matrix{Float32}
    Wv::Matrix{Float32}
    Wo::Matrix{Float32}
    embedding_table::Matrix{Float32}
    W_output::Matrix{Float32}
    config::Tuple{Int,Int,Int,Int,Int,Int}  # (vocab_size, d_model, d_k, d_v, block_size, batch_size)
end

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


function generate(m::ManualAttention, idx, max_new_tokens; temperature = 0.8)
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



# Après avoir entraîné le modèle
w, c = train_attention_model()

# Créer le modèle de génération
m_attention = ManualAttention(w, c)

# Option 1: Générer à partir d'un batch existant
xb_test, _ = get_batch("val")
generated_idx = generate(m_attention, xb_test[1:1, :], 200)
println("Génération à partir du batch:")
println(decode(vec(generated_idx[1, :])))
