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


### Création des batch

using Random

Random.seed!(1337)

block_size = 8  # consider increasing to 32 or 64 later

function get_batch(split; batch_size=32)
    data = split == "train" ? train_data : val_data
    ix = rand(1:length(data)-block_size, batch_size)
    x = [data[i + t] for i in ix, t in 0:block_size-1]
    y = [data[i + t + 1] for i in ix, t in 0:block_size-1]
    return x, y
end

using Flux
using Flux: onehotbatch
using Random
using StatsBase
Random.seed!(1337)

vocab_size = length(chars)

function bigram_activation(Wflat::Flatten, Xb)
    W = Wflat.components[1]
    idx = vec(Xb)
    N = length(idx)
    OH = zeros(Float32, C, N)
    @inbounds for (j, i) in enumerate(idx)
        OH[i, j] = 1f0
    end
    return W * OH
end


##############################
# BIGRAM MODEL FOR GENERATION
##############################

struct ManualBigram
    W::Matrix{Float32}
end

function (m::ManualBigram)(idx)
    idx_vec = vec(idx)
    logits2d = m.W[:, idx_vec]
    B, T = size(idx)
    return reshape(logits2d, C, B, T)
end

function generate(m::ManualBigram, idx, max_new_tokens)
    for _ in 1:max_new_tokens
        logits = m(idx)
        last_logits = logits[:, :, end]
        last_logits = permutedims(last_logits, (2,1))
        probs = softmax(last_logits)
        Bsize, Csize = size(probs)
        idx_next = similar(idx[:,1:1])
        for b in 1:Bsize
            p = probs[b, :]
            idx_next[b, 1] = sample(1:Csize, Weights(p))
        end
        idx = hcat(idx, idx_next)
    end
    return idx
end

C = vocab_size

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
    d_k = size(Q, 1)
    scores = K' * Q
    scores = scores ./ Float32(sqrt(d_k))
    if mask !== nothing
        scores = scores .+ mask
    end
    attn_weights = softmax(scores)
    output = V * attn_weights'
    return output
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
    d_model, d_k, d_v, block_size, batch_size, n_heads = model_config
    Wq, Wk, Wv, Wo = params.components[1], params.components[2], params.components[3], params.components[4]
    # If X is a VectNode (training with custom AD), fall back to single-head to keep gradient flow simple
    if X isa VectReverse.VectNode
        Q = Wq * X
        K = Wk * X
        V = Wv * X
        mask = create_causal_mask(block_size, batch_size)
        attn_output = scaled_dot_product_attention(Q, K, V, mask)
        return Wo * attn_output
    end
    # Multi-head path for plain arrays (generation / evaluation)
    Q = Wq * X
    K = Wk * X
    V = Wv * X
    N = size(X, 2)
    @assert d_model == d_k * n_heads "d_model must equal d_k * n_heads"
    Qh = reshape(Q, d_k, n_heads, N)
    Kh = reshape(K, d_k, n_heads, N)
    Vh = reshape(V, d_k, n_heads, N)
    mask = create_causal_mask(block_size, batch_size)
    head_outputs = zeros(Float32, d_model, N)
    for h in 1:n_heads
        Qhh = view(Qh, :, h, :)
        Khh = view(Kh, :, h, :)
        Vhh = view(Vh, :, h, :)
        out_h = scaled_dot_product_attention(Qhh, Khh, Vhh, mask)
        head_outputs[(h-1)*d_k+1:h*d_k, :] .= out_h
    end
    return Wo * head_outputs
end


"""
    create_causal_mask(T, B)

Crée un masque causal pour empêcher l'attention de voir les positions futures.
Retourne une matrice (B*T × B*T) avec -Inf pour les positions futures.
"""
function create_causal_mask(T, B)
    N = B * T
    mask = zeros(Float32, N, N)
    for b in 0:B-1
        offset = b * T
        for i in 1:T
            for j in (i+1):T
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
function initialize_attention_params(d_model, d_k, d_v, n_heads)
    @assert d_model == d_k * n_heads "d_model must equal d_k * n_heads"
    # Use combined projection matrices (d_model × d_model)
    scale = sqrt(2.0f0 / (d_model + d_model))
    Wq = scale .* randn(Float32, d_model, d_model)
    Wk = scale .* randn(Float32, d_model, d_model)
    Wv = scale .* randn(Float32, d_model, d_model)
    Wo = scale .* randn(Float32, d_model, d_model)
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
    vocab_size, d_model, d_k, d_v, block_size, batch_size, n_heads = config
    attention_params = Flatten(Wflat.components[1:4])
    embedding_table = Wflat.components[5]
    pos_emb = length(Wflat.components) >= 6 ? Wflat.components[6] : nothing
    idx = vec(Xb)
    N = length(idx)
    OH = zeros(Float32, vocab_size, N)
    @inbounds for (j, i) in enumerate(idx)
        OH[i, j] = 1f0
    end
    X = embedding_table * OH
    if pos_emb !== nothing
        # Repeat positional indices per batch
        T_used = min(block_size, size(pos_emb, 2))
        pos_indices = repeat(1:T_used, batch_size)[1:N]
        X = X .+ pos_emb[:, pos_indices]
    end
    model_config = (d_model, d_k, d_v, block_size, batch_size, n_heads)
    output = attention_forward(attention_params, X, model_config)
    return output
end


##############################
# TRAINING
##############################

"""
Exemple d'entraînement avec attention
"""
function train_attention_model(; num_iters=1000, d_model=64, d_k=32, d_v=32, n_heads=2, learning_rate=0.001, batch_size_train=32, eval_interval=200)
    @assert d_model == d_k * n_heads "d_model must equal d_k * n_heads"
    config = (vocab_size, d_model, d_k, d_v, block_size, batch_size_train, n_heads)
    
    println("Initializing model...")
    println("  d_model=$d_model, d_k=$d_k, d_v=$d_v")
    println("  batch_size=$batch_size_train, block_size=$block_size")
    attention_params = initialize_attention_params(d_model, d_k, d_v, n_heads)
    embedding_table = 0.01f0 .* randn(Float32, d_model, vocab_size)
    pos_emb = 0.01f0 .* randn(Float32, d_model, block_size)  # learnable positional embeddings
    W_output = 0.01f0 .* randn(Float32, vocab_size, d_model)
    W_full = Flatten([attention_params.components..., embedding_table, pos_emb, W_output])
    
    # Fonction de loss qui échantillonne un NOUVEAU batch à chaque appel
    function attention_loss(W_params)
        xb_new, yb_new = get_batch("train", batch_size=batch_size_train)
        # Components: 1:4 attention, 5 embedding, 6 pos_emb, 7 output
        attn_out = attention_activation(Flatten(W_params.components[1:6]), xb_new, config)
        logits = W_params.components[7] * attn_out
        targets1d = vec(yb_new)
        Ntargets = length(targets1d)
        # log-softmax (VectReverse aware)
        log_probs = logits isa VectReverse.VectNode ? VectReverse.logsoftmax(logits; dims=1) : Flux.logsoftmax(logits; dims=1)
        Y = zeros(Float32, vocab_size, Ntargets)
        @inbounds for (j, t) in enumerate(targets1d)
            Y[t, j] = 1f0
        end
        if log_probs isa VectReverse.VectNode
            # Keep as VectNode for AD
            ce = -(sum(log_probs .* Y) / Ntargets)
            return ce  # DO NOT unwrap value
        else
            return -(sum(log_probs .* Y) / Ntargets)
        end
    end
    
    L(w) = attention_loss(w)
    
    println("\nTraining for $num_iters iterations (lr=$learning_rate)...")
    println("Progress will be shown every $eval_interval iterations\n")
    
    # Entraînement par blocs pour afficher la progression
    losses = Float64[L(W_full)]
    W_current = W_full
    
    for block_start in 1:eval_interval:num_iters
        block_iters = min(eval_interval, num_iters - block_start + 1)
        block_losses, W_current = train!(VectReverse.gradient!, L, W_current, 
                                         num_iters=block_iters, 
                                         rule=Optimisers.Adam(learning_rate),
                                         losses=Float64[])
        append!(losses, block_losses)
        
        iter_num = block_start + block_iters - 1
        println("Iter $iter_num/$num_iters - Loss: $(round(block_losses[end], digits=4))")
    end
    
    W_trained = W_current
    
    println("\n" * "="^50)
    println("Training complete!")
    println("  Initial loss: ", round(losses[1], digits=4))
    println("  Final loss: ", round(losses[end], digits=4))
    improvement = (losses[1] - losses[end]) / losses[1] * 100
    println("  Improvement: ", round(improvement, digits=2), "%")
    println("="^50)
    
    return W_trained, config, losses
end

# Perplexity / validation evaluation helper
function evaluate_perplexity(W_params, config; num_batches=20, batch_size_eval=32)
    vocab_size, d_model, d_k, d_v, block_size_cfg, batch_size_train_cfg, n_heads = config
    total_loss = 0.0
    total_tokens = 0
    for _ in 1:num_batches
        xb_val, yb_val = get_batch("val", batch_size=batch_size_eval)
        attn_out = attention_activation(Flatten(W_params.components[1:6]), xb_val, config)
        logits = W_params.components[7] * attn_out
        targets = vec(yb_val)
        Ntargets = length(targets)
        if logits isa VectReverse.VectNode
            log_probs = VectReverse.logsoftmax(logits; dims=1)
        else
            log_probs = Flux.logsoftmax(logits; dims=1)
        end
        Y = zeros(Float32, vocab_size, Ntargets)
        @inbounds for (j, t) in enumerate(targets)
            Y[t, j] = 1f0
        end
        if log_probs isa VectReverse.VectNode
            batch_loss = -(sum(log_probs .* Y).value / Ntargets)
        else
            batch_loss = -(sum(log_probs .* Y) / Ntargets)
        end
        total_loss += batch_loss * Ntargets
        total_tokens += Ntargets
    end
    avg_nll = total_loss / total_tokens
    ppl = exp(avg_nll)
    return avg_nll, ppl
end

##############################
# GENERATION
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
    pos_emb::Matrix{Float32}
    W_output::Matrix{Float32}
    config::Tuple{Int,Int,Int,Int,Int,Int}
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
        W_trained.components[6],  # pos_emb
        W_trained.components[7],  # W_output
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
    vocab_size, d_model, d_k, d_v, block_size_config, batch_size_config, n_heads = m.config
    B, T = size(idx)
    idx_vec = vec(idx)
    N = length(idx_vec)
    OH = zeros(Float32, vocab_size, N)
    @inbounds for (j, i) in enumerate(idx_vec)
        OH[i, j] = 1f0
    end
    X = m.embedding_table * OH
    # Add positional embeddings (repeat positions per batch)
    T_used = min(T, size(m.pos_emb, 2))
    pos_indices = repeat(1:T_used, size(idx,1))[1:N]
    X = X .+ m.pos_emb[:, pos_indices]
    Q = m.Wq * X
    K = m.Wk * X
    V = m.Wv * X
    N = size(X,2)
    @assert d_model == d_k * n_heads
    Qh = reshape(Q, d_k, n_heads, N)
    Kh = reshape(K, d_k, n_heads, N)
    Vh = reshape(V, d_k, n_heads, N)
    mask = create_causal_mask(T, B)
    head_outputs = similar(Q)
    for h in 1:n_heads
        Qhh = view(Qh, :, h, :)
        Khh = view(Kh, :, h, :)
        Vhh = view(Vh, :, h, :)
        scores = Khh' * Qhh ./ Float32(sqrt(d_k))
        scores = scores .+ mask
        attn_weights = softmax(scores)
        out_h = Vhh * attn_weights'
        head_outputs[(h-1)*d_k+1:h*d_k, :] .= out_h
    end
    output = m.Wo * head_outputs
    logits = m.W_output * output
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
        T_current = size(idx, 2)
        idx_cond = T_current > block_size ? idx[:, end-block_size+1:end] : idx
        logits = m(idx_cond)
        last_logits = logits[:, :, end]
        last_logits = permutedims(last_logits, (2, 1))
        if temperature != 1.0
            last_logits = last_logits ./ temperature
        end
        probs = softmax(last_logits)
        Bsize, Vsize = size(probs)
        idx_next = similar(idx[:, 1:1])
        for b in 1:Bsize
            p = probs[b, :]
            idx_next[b, 1] = sample(1:Vsize, Weights(p))
        end
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
function _topk_probs(logits_row::AbstractVector{<:Real}, k::Int)
    k = min(k, length(logits_row))
    idxs = partialsortperm(logits_row, rev=true, 1:k)
    top_logits = logits_row[idxs]
    shifted = top_logits .- maximum(top_logits)
    exps = exp.(shifted)
    probs = exps ./ sum(exps)
    return idxs, probs
end

function generate_text(W_trained, config, prompt=""; max_new_tokens=100, temperature=1.0, top_k=nothing)
    vocab_size, d_model, d_k, d_v, block_size, batch_size_config, n_heads = config
    m = ManualAttention(W_trained, config)
    if isempty(prompt)
        idx = rand(1:vocab_size, 1, 1)
    else
        encoded = encode(prompt)
        if length(encoded) > block_size
            encoded = encoded[end-block_size+1:end]
        end
        idx = reshape(encoded, 1, :)
    end
    # If top_k provided, override sampling loop locally
    if top_k === nothing
        idx_generated = generate(m, idx, max_new_tokens, temperature=temperature)
    else
        for _ in 1:max_new_tokens
            T_current = size(idx, 2)
            idx_cond = T_current > block_size ? idx[:, end-block_size+1:end] : idx
            logits = m(idx_cond)[:, :, end]  # vocab_size × B
            logits = permutedims(logits, (2,1))  # B × vocab_size
            if temperature != 1.0
                logits = logits ./ temperature
            end
            Bsize, Vsize = size(logits)
            idx_next = similar(idx[:, 1:1])
            for b in 1:Bsize
                row = logits[b, :]
                kidxs, kprobs = _topk_probs(row, top_k)
                choice = sample(kidxs, Weights(kprobs))
                idx_next[b, 1] = choice
            end
            idx = hcat(idx, idx_next)
        end
        idx_generated = idx
    end
    return decode(vec(idx_generated[1, :]))
end
# Exemple d'utilisation:
# Pour un entraînement rapide (test):
w, c, losses = train_attention_model(num_iters=500)

# Pour un bon résultat (recommandé):
w, c, losses = train_attention_model(num_iters=5000, d_model=64, d_k=32, d_v=32, n_heads=2)

# Pour générer du texte:
text = generate_text(w, c, "", max_new_tokens=200, temperature=0.8)
println(text)

nll, ppl = evaluate_perplexity(w, c; num_batches=50)
println("Val NLL=$(round(nll, digits=4))  Val PPL=$(round(ppl, digits=2))")