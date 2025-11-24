# Assuming the necessary AD/training utility includes are present:
# LabAD = joinpath(dirname(@__DIR__), "LabAD")
# include(joinpath(LabAD, "test", "test.jl"))
# include(joinpath(LabAD, "solution", "forward.jl"))
include(joinpath(@__DIR__, "train.jl"))
include(joinpath(@__DIR__, "reverse_vectorized.jl"))

using Flux
using Flux: onehotbatch, softmax
using Random
using StatsBase
using BSON # Used to load preprocessed data

Random.seed!(1337)

# --- Load Preprocessed Data ---

const DATA_FILE = joinpath(@__DIR__,"preprocessed_data.bson")
data_loaded = BSON.load(DATA_FILE)

train_data = data_loaded[:"train_data"]
val_data   = data_loaded[:"val_data"]
stoi       = data_loaded[:"stoi"]
itos       = data_loaded[:"itos"]
vocab_size = data_loaded[:"vocab_size"]

# --- ENCODING / DECODING (Now for Word Tokens) ---

# String -> Vector{Int} (uses the loaded stoi)
function encode(s::AbstractString; stoi=stoi)
    # This simplified version does not handle tokenization/unk, 
    # but assumes input is pre-tokenized if used for prompting.
    # For actual prompting, we'd need to tokenize the prompt first.
    return [get(stoi, lowercase(t), stoi["[UNK]"]) for t in split(s)]
end

# Vector{Int} -> String (uses the loaded itos)
function decode(v::AbstractVector{<:Integer}; itos=itos)
    # Tokens à ignorer lors de l'affichage
    ignored_tokens = Set(["[EOS]", "[BOS]", "[PAD]", "[UNK]"]) 
    
    # 1. Convertir les indices en mots (strings)
    words = [itos[i] for i in v]
    
    # 2. Filtrer les tokens spéciaux
    filtered_words = filter(w -> w ∉ ignored_tokens, words)
    
    # 3. Joindre les mots avec un espace
    return join(filtered_words, " ")
end

# --- Batch Creation ---

batch_size = 16      # Number of sequences processed in parallel (reduced for CPU)
block_size = 32      # Length of the context (increased for word-level models)

function get_batch(split)
    data = split == "train" ? train_data : val_data

    # Randomly draw starting positions
    ix = rand(1:length(data)-block_size, batch_size)

    # Create the input matrix x (batch_size × block_size)
    x = [data[i + t] for i in ix, t in 0:block_size-1]

    # Create the target matrix y (batch_size × block_size)
    y = [data[i + t + 1] for i in ix, t in 0:block_size-1]

    return x, y
end

# Test the batch creation
xb, yb = get_batch("train")

println("inputs size: ", size(xb))
println("targets size: ", size(yb))
println("Vocabulary Size (V): ", vocab_size)

# --- Placeholder for Manual Bigram (kept for structure, not used in Transformer) ---
# ... (Bigram functions remain conceptually the same but use word indices)
# ...
##############################
# ATTENTION MECHANISM
##############################

"""
    scaled_dot_product_attention(Q, K, V, mask=nothing)

Implements scaled dot-product attention compatible with VectReverse (AD system).
- Q : queries (d_k × N) where N = B*T
- K : keys (d_k × N)
- V : values (d_v × N)
- mask : optional mask for causal attention

Returns : (d_v × N)
"""
function scaled_dot_product_attention(Q, K, V, mask=nothing)
    # Q: d_k × N
    # K: d_k × N  
    # V: d_v × N
    
    d_k = size(Q, 1)
    
    # Attention Scores: K^T * Q (N × N)
    scores = K' * Q  # (N × d_k) * (d_k × N) = N × N
    
    # Scaling
    scores = scores ./ Float32(sqrt(d_k))
    
    # Apply mask if provided (for causal attention)
    if mask !== nothing
        # mask should be N × N with -Inf for future positions
        scores = scores .+ mask
    end
    
    # Softmax over the last dimension (rows/query dimension)
    attn_weights = softmax(scores)  # N × N
    
    # Apply attention to values
    # V is d_v × N, attn_weights is N × N
    output = V * attn_weights'  # (d_v × N) * (N × N) = d_v × N
    
    return output
end


"""
    SingleHeadAttention

Structure for a single attention head with its parameters.
Compatible with Flatten for training via VectReverse.grad.
"""
struct SingleHeadAttention
    d_model::Int      # input/output dimension
    d_k::Int          # queries/keys dimension
    d_v::Int          # values dimension
end

"""
    attention_forward(params::Flatten, X, model_config)

Forward function for the attention head, compatible with VectReverse.
- params : Flatten([Wq, Wk, Wv, Wo]) where each matrix is (d × d_model)
- X : embedding matrix (d_model × N) where N = B*T
- model_config : tuple (d_model, d_k, d_v, block_size, batch_size)

Returns : (d_model × N)
"""
function attention_forward(params::Flatten, X, model_config)
    d_model, d_k, d_v, block_size, batch_size = model_config
    
    Wq, Wk, Wv, Wo = params.components[1], params.components[2], 
                     params.components[3], params.components[4]
    
    # Linear Projections
    Q = Wq * X  # (d_k × d_model) * (d_model × N) = d_k × N
    K = Wk * X  # (d_k × d_model) * (d_model × N) = d_k × N  
    V = Wv * X  # (d_v × d_model) * (d_model × N) = d_v × N
    
    # Create the causal mask to prevent looking into the future
    N = size(X, 2)  # B*T
    mask = create_causal_mask(block_size, batch_size)
    
    # Attention
    attn_output = scaled_dot_product_attention(Q, K, V, mask)  # d_v × N
    
    # Output Projection
    output = Wo * attn_output  # (d_model × d_v) * (d_v × N) = d_model × N
    
    return output
end


"""
    create_causal_mask(T, B)

Creates a causal mask to prevent attention from seeing future positions.
Returns an (B*T × B*T) matrix with -Inf for future positions.
"""
function create_causal_mask(T, B)
    N = B * T
    mask = zeros(Float32, N, N)
    
    # For each batch
    for b in 0:B-1
        offset = b * T
        # For each position in the sequence (i: query position)
        for i in 1:T
            # For each token position in the sequence (j: key position)
            for j in (i+1):T
                # Mask future positions (keys j > queries i)
                # Apply masking only within the same batch slice
                mask[offset + i, offset + j] = -1f10
            end
        end
    end
    
    return mask
end


"""
    initialize_attention_params(d_model, d_k, d_v)

Initializes attention parameters (Wq, Wk, Wv, Wo) using Xavier initialization.
Returns a Flatten([Wq, Wk, Wv, Wo]).
"""
function initialize_attention_params(d_model, d_k, d_v)
    # Xavier initialization scaling factors
    scale_qk = Float32(sqrt(2.0 / (d_model + d_k)))
    scale_v = Float32(sqrt(2.0 / (d_model + d_v)))
    scale_o = Float32(sqrt(2.0 / (d_v + d_model)))
    
    Wq = scale_qk .* randn(Float32, d_k, d_model)
    Wk = scale_qk .* randn(Float32, d_k, d_model)
    Wv = scale_v .* randn(Float32, d_v, d_model)
    Wo = scale_o .* randn(Float32, d_model, d_v)
    
    return Flatten([Wq, Wk, Wv, Wo])
end


"""
    attention_activation(Wflat::Flatten, Xb, config)

Complete activation function including embedding + attention.
Compatible with VectReverse.grad for training.

- Wflat : Flatten([Wq, Wk, Wv, Wo, embedding_table])
- Xb : index matrix (B × T)
- config : (vocab_size, d_model, d_k, d_v, block_size, batch_size)
"""
function attention_activation(Wflat::Flatten, Xb, config)
    vocab_size, d_model, d_k, d_v, block_size, batch_size = config
    
    # Extract parameters
    # The first 4 components are Wq, Wk, Wv, Wo
    # The 5th is the embedding table
    attention_params = Flatten(Wflat.components[1:4])
    embedding_table = Wflat.components[5]  # (d_model × vocab_size)
    
    # Embedding: convert indices to vectors (Lookup)
    idx = vec(Xb)  # length N = B*T
    N = length(idx)
    
    # One-hot encoding (used for efficient lookup via matrix multiplication)
    OH = zeros(Float32, vocab_size, N)
    @inbounds for (j, i) in enumerate(idx)
        OH[i, j] = 1f0
    end
    
    # Lookup in the embedding table
    # (d_model × vocab_size) * (vocab_size × N) = d_model × N
    X = embedding_table * OH  
    
    # Attention forward pass
    model_config = (d_model, d_k, d_v, block_size, batch_size)
    output = attention_forward(attention_params, X, model_config)
    
    return output
end


##############################
# USAGE EXAMPLE
##############################

"""
Example of training with attention
"""
function train_attention_model(;embedding_dimension=16, dim_k=4,dim_v=4, num_iters_to_run=100, learning_rate=0.001, dir="Transformers")
    # Hyperparameters (Optimized for low compute)
    d_model = embedding_dimension      # embedding dimension (reduced from 32)
    d_k = dim_k           # queries/keys dimension (reduced from 8)
    d_v = dim_v           # values dimension (reduced from 8)
    
    # Reuse global config
    global vocab_size, block_size, batch_size
    config = (vocab_size, d_model, d_k, d_v, block_size, batch_size)
    
    # Initialize parameters
    attention_params = initialize_attention_params(d_model, d_k, d_v)
    # Embedding table initialization (d_model × vocab_size)
    embedding_table = Float32(0.01) .* randn(Float32, d_model, vocab_size)
    
    # Combine all parameters before the output projection
    W_partial = Flatten([attention_params.components..., embedding_table])
    
    # Prepare data for loss calculation
    xb, yb = get_batch("train")
    
    # Final output projection matrix (d_model -> vocab_size)
    W_output = Float32(0.01) .* randn(Float32, vocab_size, d_model)
    # Combine all parameters for training
    W_full = Flatten([W_partial.components..., W_output])
    
    # Loss function closure
    function attention_loss(W_params)
        # Attention output: d_model × N
        # Components 1:5 are the attention weights and embedding table
        attn_out = attention_activation(Flatten(W_params.components[1:5]), xb, config)
        
        # Projection to logits: vocab_size × N
        W_out = W_params.components[6]
        logits = W_out * attn_out
        
        # MSE Loss with one-hot targets
        targets1d = vec(yb)
        y_onehot = onehotbatch(targets1d, 1:vocab_size)
        
        # Ensure targets are Float32 for AD compatibility
        y_onehot_f32 = convert(AbstractArray{Float32}, y_onehot)

        diff = logits .- y_onehot_f32
        return sum(diff .^ 2) / length(targets1d)
    end
    
    # ... (Dans la fonction train_attention_model) ...

    L(w) = attention_loss(w)
    
    # Configuration du training
    # num_iters_to_run = 3000 
    # learning_rate = 0.01
    
    # Training
    println("Training the model with Single Head Attention for $num_iters_to_run iterations...")
    
    # CORRECTION : num_iters_to_run est passé comme 4ème argument positionnel
    losses, W_trained = train!(VectReverse.gradient!, # 1. gradient!
                                L,                    # 2. L
                                W_full,               # 3. w
                                num_iters_to_run;     # 4. num_iters (positionnel, puis le ; pour les keywords)
                                rule=Optimisers.Adam(learning_rate))
    
    println("Final Loss after $num_iters_to_run iterations: ", losses[end])

    # --- NOUVEAU : Sauvegarde du Modèle ---
    MODEL_OUTPUT_FILE = joinpath(dir,"single_head_transformer_weights.bson")
    
    # Créer le dictionnaire de sauvegarde
    model_data = Dict(
        "W_trained" => W_trained,
        "config" => config
    )
    
    BSON.bson(MODEL_OUTPUT_FILE, model_data)
    println(" Model successfully saved to $(MODEL_OUTPUT_FILE)")
    # ---------------------------------------
    
    return W_trained, config
end

# Placeholder call: w, c = train_attention_model()

##############################
# GENERATION WITH ATTENTION
##############################

"""
    ManualAttention

Structure for generation using a trained attention model.
Contains all necessary parameters (Wq, Wk, Wv, Wo, embedding, output).
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

Constructor from trained parameters.
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

Forward pass of the attention model (without VectNode, for fast generation).
- idx : token index matrix (B × T)
Returns : logits of shape (vocab_size × B × T)
"""
function (m::ManualAttention)(idx)
    vocab_size, d_model, d_k, d_v, block_size_config, batch_size_config = m.config
    
    B, T = size(idx)
    
    # Embedding (Manual Lookup via Matmul)
    idx_vec = vec(idx)  # B*T
    N = length(idx_vec)
    
    OH = zeros(Float32, vocab_size, N)
    @inbounds for (j, i) in enumerate(idx_vec)
        OH[i, j] = 1f0
    end
    
    X = m.embedding_table * OH  # d_model × N
    
    # Q, K, V Projections
    Q = m.Wq * X  # d_k × N
    K = m.Wk * X  # d_k × N
    V = m.Wv * X  # d_v × N
    
    # Attention with Causal Mask
    scores = K' * Q ./ Float32(sqrt(d_k))  # N × N
    
    # Causal Mask application
    mask = create_causal_mask(T, B)
    scores = scores .+ mask
    
    # Softmax
    attn_weights = softmax(scores)  # N × N
    
    # Apply to values
    attn_out = V * attn_weights'  # d_v × N
    
    # Output Projection
    output = m.Wo * attn_out  # d_model × N
    
    # Projection to logits
    logits = m.W_output * output  # vocab_size × N
    
    # Reshape to (vocab_size × B × T)
    return reshape(logits, vocab_size, B, T)
end

"""
    generate(m::ManualAttention, idx, max_new_tokens; temperature=1.0)

Generates new tokens with the attention model.
- m : ManualAttention model
- idx : initial context (B × T)
- max_new_tokens : number of tokens to generate
- temperature : controls randomness (1.0 = normal)

Returns : augmented sequence (B × (T + max_new_tokens))
"""
function generate(m::ManualAttention, idx, max_new_tokens; temperature=1.0)
    vocab_size, d_model, d_k, d_v, block_size, batch_size_config = m.config
    
    for _ in 1:max_new_tokens
        # Truncate to block_size if necessary (causal attention)
        T_current = size(idx, 2)
        if T_current > block_size
            idx_cond = idx[:, end-block_size+1:end]
        else
            idx_cond = idx
        end
        
        # Forward pass
        logits = m(idx_cond)  # vocab_size × B × T
        
        # Take the logits of the last token
        last_logits = logits[:, :, end]  # vocab_size × B
        last_logits = permutedims(last_logits, (2, 1))  # B × vocab_size
        
        # Apply temperature
        if temperature != 1.0
            last_logits = last_logits ./ temperature
        end
        
        # Softmax to get probabilities
        probs = softmax(last_logits)  # B × vocab_size
        
        Bsize, Vsize = size(probs)
        idx_next = similar(idx[:, 1:1])  # B × 1
        
        # Sample for each batch element
        for b in 1:Bsize
            p = probs[b, :]
            # Sample index (requires StatsBase.sample and Weights)
            idx_next[b, 1] = sample(1:Vsize, Weights(p))
        end
        
        # Append the new token
        idx = hcat(idx, idx_next)
    end
    
    return idx
end


"""
    generate_text(W_trained, config, prompt=""; max_new_tokens=100, temperature=1.0)

Helper function to generate text from a prompt.
"""
function generate_text(W_trained, config, prompt=""; max_new_tokens=100, temperature=1.0)
    
    # Create the model structure
    m = ManualAttention(W_trained, config)
    
    # Encode the prompt or start with the [BOS] token
    bos_token = stoi["[BOS]"]
    
    if isempty(prompt)
        # Start with [BOS]
        idx = reshape([bos_token], 1, 1)
    else
        # Encode the prompt tokens
        encoded = encode(prompt)
        
        # Prepend [BOS] and truncate if necessary
        full_context = vcat(bos_token, encoded)
        if length(full_context) > config[4] # config[4] is block_size
             # Truncate to block_size
            full_context = full_context[end-config[4]+1:end]
        end
        idx = reshape(full_context, 1, :)  # 1 × T (batch size = 1)
    end
    
    # Generate
    idx_generated = generate(m, idx, max_new_tokens, temperature=temperature)
    
    # Decode
    return decode(vec(idx_generated[1, :]))
end

using BSON

"""
    load_trained_model(file_path::String)

Charges les poids et la configuration du modèle à partir d'un fichier BSON.
Crée ensuite une instance de ManualAttention.
"""
function load_trained_model(file_path::String)
    println("⏳ Loading model from $file_path...")
    
    # 1. Charger les données
    loaded_data = BSON.load(file_path)
    
    W_trained = loaded_data[:"W_trained"]
    config = loaded_data[:"config"]
    
    # 2. Reconstruire le modèle de génération
    m = ManualAttention(W_trained, config)
    
    println("✅ Model loaded successfully.")
    
    return m, W_trained, config
end

# train a new model #
N_ITER = 100
L_RATE = 0.01
EMB_DIM = 64
DIM_K = 16
DIM_V = 16
w_trained, c_config = train_attention_model(embedding_dimension=EMB_DIM, dim_k=DIM_K, dim_v= DIM_V, num_iters_to_run=N_ITER, learning_rate=L_RATE)
text_output = generate_text(w_trained, c_config, "The boy said", max_new_tokens=50, temperature=1.2)
println(text_output)


# Uncomment to load an already trained model 
# load_trained_model("Transformers/single_head_transformer_weights.bson")
# text_output = generate_text(w_trained, c_config, "The king said", max_new_tokens=50, temperature=1.6)
# println(text_output)