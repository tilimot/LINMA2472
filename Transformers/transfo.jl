# Assuming the necessary AD/training utility includes are present:
# LabAD = joinpath(dirname(@__DIR__), "LabAD")
# include(joinpath(LabAD, "test", "test.jl"))
# include(joinpath(LabAD, "solution", "forward.jl"))

include(joinpath(@__DIR__, "train.jl"))
include(joinpath(@__DIR__, "reverse_vectorized.jl"))
include(joinpath(@__DIR__, "utils.jl"))

using Flux
using Flux: onehotbatch, softmax
using Random
using StatsBase
using BSON # load preprocessed data

Random.seed!(1337)

# Load Preprocessed Data 

const DATA_FILE = joinpath(@__DIR__,"preprocessed_data.bson")
data_loaded = BSON.load(DATA_FILE)

train_data = data_loaded[:"train_data"]
val_data   = data_loaded[:"val_data"]
stoi       = data_loaded[:"stoi"]
itos       = data_loaded[:"itos"]
vocab_size = data_loaded[:"vocab_size"]

# ENCODING / DECODING 

# String -> Vector{Int} (uses the loaded stoi)
function encode(s::AbstractString; stoi=stoi)
    # with input is pre-tokenized if used for prompting.
    return [get(stoi, lowercase(t), stoi["[UNK]"]) for t in split(s)]
end

# Vector -> String 
function decode(v::AbstractVector{<:Integer}; itos=itos)
    ignored_tokens = Set(["[EOS]", "[BOS]", "[PAD]", "[UNK]"]) 
    
    words = [itos[i] for i in v]
    filtered_words = filter(w -> w ∉ ignored_tokens, words)

    return join(filtered_words, " ")
end


batch_size = 16      # Number of sequences processed in parallel 
block_size = 32      # Length of the context 



xb, yb = get_batch("train")

println("inputs size: ", size(xb))
println("targets size: ", size(yb))
println("Vocabulary Size (V): ", vocab_size)

# ATTENTION MECHANISM

function scaled_dot_product_attention(Q, K, V, mask=nothing)
    # Q: d_k × N
    # K: d_k × N  
    # V: d_v × N
    d_k = size(Q, 1)
    
    # Attention Scores: K^T * Q (N × N)
    scores = K' * Q  # (N × d_k) * (d_k × N) = N × N
    scores = scores ./ Float32(sqrt(d_k))
    
    if mask !== nothing
        # mask: N × N with -Inf for future positions
        scores = scores .+ mask
    end
    
    # Softmax over the last dimension (rows/query dimension)
    attn_weights = softmax(scores)  # N × N
 
    # V is d_v × N, attn_weights is N × N
    output = V * attn_weights'  # (d_v × N) * (N × N) = d_v × N
    
    return output
end

struct SingleHeadAttention
    d_model::Int      # input/output dimension
    d_k::Int          # queries/keys dimension
    d_v::Int          # values dimension
end


function attention_forward(params::Flatten, X, model_config)
    d_model, d_k, d_v, block_size, batch_size = model_config
    
    Wq, Wk, Wv, Wo = params.components[1], params.components[2], 
                     params.components[3], params.components[4]
    
    # Linear Projections
    Q = Wq * X  # (d_k × d_model) * (d_model × N) = d_k × N
    K = Wk * X  # (d_k × d_model) * (d_model × N) = d_k × N  
    V = Wv * X  # (d_v × d_model) * (d_model × N) = d_v × N
    
    N = size(X, 2)  # B*T
    mask = create_causal_mask(block_size, batch_size) # prevent looking into the future
 
    attn_output = scaled_dot_product_attention(Q, K, V, mask)  # d_v × N
    
    # Output Projection
    output = Wo * attn_output  # (d_model × d_v) * (d_v × N) = d_model × N
    
    return output
end


function create_causal_mask(T, B)
    N = B * T
    mask = zeros(Float32, N, N)

    for b in 0:B-1
        offset = b * T
        for i in 1:T
            for j in (i+1):T
                mask[offset + i, offset + j] = -1f10    # Mask future positions (keys j > queries i)
            end
        end
    end
    
    return mask
end

function initialize_attention_params(d_model, d_k, d_v)
    scale_qk = Float32(sqrt(2.0 / (d_model + d_k)))
    scale_v = Float32(sqrt(2.0 / (d_model + d_v)))
    scale_o = Float32(sqrt(2.0 / (d_v + d_model)))
    Wq = scale_qk .* randn(Float32, d_k, d_model)
    Wk = scale_qk .* randn(Float32, d_k, d_model)
    Wv = scale_v .* randn(Float32, d_v, d_model)
    Wo = scale_o .* randn(Float32, d_model, d_v)
    
    return Flatten([Wq, Wk, Wv, Wo])
end


function attention_activation(Wflat::Flatten, Xb, config)
    vocab_size, d_model, d_k, d_v, block_size, batch_size = config
    
    # components: Wq, Wk, Wv, Wo
    # The 5th is the embedding table
    attention_params = Flatten(Wflat.components[1:4])
    embedding_table = Wflat.components[5]  # (d_model × vocab_size)

    idx = vec(Xb)  # length N = B*T
    N = length(idx)
    
    # One-hot encoding (used for efficient lookup via matrix multiplication)
    OH = zeros(Float32, vocab_size, N)
    @inbounds for (j, i) in enumerate(idx)
        OH[i, j] = 1f0
    end

    # (d_model × vocab_size) * (vocab_size × N) = d_model × N
    X = embedding_table * OH  

    model_config = (d_model, d_k, d_v, block_size, batch_size)
    output = attention_forward(attention_params, X, model_config)
    
    return output
end

################# USAGE EXAMPLE ##################################

function train_attention_model(;embedding_dimension=16, dim_k=4,dim_v=4, num_iters_to_run=100, learning_rate=0.001, dir="Transformers")
    d_model = embedding_dimension 
                          # embedding dimension (reduced from 32)
    d_k = dim_k           # queries/keys dimension (reduced from 8)
    d_v = dim_v           # values dimension (reduced from 8)
    
    global vocab_size, block_size, batch_size
    config = (vocab_size, d_model, d_k, d_v, block_size, batch_size)

    attention_params = initialize_attention_params(d_model, d_k, d_v)
    
    embedding_table = Float32(0.01) .* randn(Float32, d_model, vocab_size) # (d_model × vocab_size)
    
    W_partial = Flatten([attention_params.components..., embedding_table])

    xb, yb = get_batch("train")
    
    W_output = Float32(0.01) .* randn(Float32, vocab_size, d_model) # output (d_model -> vocab_size)
    W_full = Flatten([W_partial.components..., W_output])
    
    function attention_loss(W_params)
        # Attention output: d_model × N
        # components 1:5 are the attention weights and embedding table
        attn_out = attention_activation(Flatten(W_params.components[1:5]), xb, config)
        
        # Projection to logits: vocab_size × N
        W_out = W_params.components[6]
        logits = W_out * attn_out
        
        # MSE Loss with one-hot targets
        targets1d = vec(yb)
        y_onehot = onehotbatch(targets1d, 1:vocab_size)
 
        y_onehot_f32 = convert(AbstractArray{Float32}, y_onehot)

        diff = logits .- y_onehot_f32
        return sum(diff .^ 2) / length(targets1d)
    end
    

    L(w) = attention_loss(w)
    
    println("Training the model with Single Head Attention for $num_iters_to_run iterations...")
    
    losses, W_trained = train!(VectReverse.gradient!, 
                                L,                    
                                W_full,               
                                num_iters_to_run;     
                                rule=Optimisers.Adam(learning_rate))
    
    println("Final Loss after $num_iters_to_run iterations: ", losses[end])

    MODEL_OUTPUT_FILE = joinpath(dir,"single_head_transformer_weights.bson")
    
    # dictionnaire de sauvegarde
    model_data = Dict(
        "W_trained" => W_trained,
        "config" => config
    )
    
    BSON.bson(MODEL_OUTPUT_FILE, model_data)
    println(" Model successfully saved to $(MODEL_OUTPUT_FILE)")

    return W_trained, config
end



## GENERATION WITH ATTENTION ##

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

    idx_vec = vec(idx)  # B*T
    N = length(idx_vec)
    
    OH = zeros(Float32, vocab_size, N)
    @inbounds for (j, i) in enumerate(idx_vec)
        OH[i, j] = 1f0
    end
    
    X = m.embedding_table * OH  # d_model × N

    Q = m.Wq * X  # d_k × N
    K = m.Wk * X  # d_k × N
    V = m.Wv * X  # d_v × N
    
    scores = K' * Q ./ Float32(sqrt(d_k))  # N × N
    
    mask = create_causal_mask(T, B)
    scores = scores .+ mask
    attn_weights = softmax(scores)  # N × N

    attn_out = V * attn_weights'  # d_v × N
    
    output = m.Wo * attn_out  # d_model × N
    
    logits = m.W_output * output  # vocab_size × N

    return reshape(logits, vocab_size, B, T)  # (vocab_size × B × T)
end

function generate(m::ManualAttention, idx, max_new_tokens; temperature=1.0)
    vocab_size, d_model, d_k, d_v, block_size, batch_size_config = m.config
    
    for _ in 1:max_new_tokens
        # Truncate to block_size if necessary 
        T_current = size(idx, 2)
        if T_current > block_size
            idx_cond = idx[:, end-block_size+1:end]
        else
            idx_cond = idx
        end
        
        logits = m(idx_cond)  # vocab_size × B × T
        
        # Take the logits of the last token
        last_logits = logits[:, :, end]  # vocab_size × B
        last_logits = permutedims(last_logits, (2, 1))  # B × vocab_size

        if temperature != 1.0
            last_logits = last_logits ./ temperature
        end
        
        # Softmax to get probabilities
        probs = softmax(last_logits)  # B × vocab_size
        
        Bsize, Vsize = size(probs)
        idx_next = similar(idx[:, 1:1])  # B × 1

        for b in 1:Bsize
            p = probs[b, :]
            idx_next[b, 1] = sample(1:Vsize, Weights(p))
        end

        idx = hcat(idx, idx_next) # concaténer 
    end
    
    return idx
end



function generate_text(W_trained, config, prompt=""; max_new_tokens=100, temperature=1.0)
    m = ManualAttention(W_trained, config)

    bos_token = stoi["[BOS]"]
    
    if isempty(prompt)
        # Start with [BOS]
        idx = reshape([bos_token], 1, 1)
    else
        encoded = encode(prompt)
        
        # Prepend [BOS] and truncate if necessary
        full_context = vcat(bos_token, encoded)
        if length(full_context) > config[4] # config[4] is block_size
             # Truncate to block_size
            full_context = full_context[end-config[4]+1:end]
        end
        idx = reshape(full_context, 1, :)  # 1 × T (batch size = 1)
    end
    
    idx_generated = generate(m, idx, max_new_tokens, temperature=temperature)
    
    return decode(vec(idx_generated[1, :]))
end

using BSON


function load_trained_model(file_path::String)
    println("Loading model from $file_path...")
    
    loaded_data = BSON.load(file_path)
    
    W_trained = loaded_data[:"W_trained"]
    config = loaded_data[:"config"]
    m = ManualAttention(W_trained, config)
    
    println("Model loaded successfully.")
    
    return m, W_trained, config
end

# train a new model 
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