include(joinpath(@__DIR__,"train.jl"))
include(joinpath(@__DIR__,"reverse_vectorized.jl"))

include(joinpath(@__DIR__, "config.jl"))
using .Config

using Flux
using Flux: onehotbatch, softmax
using Random
using StatsBase
using BSON # load preprocessed data

Random.seed!(1337)

###################### Global var ######################

# load preprocessed data 
const DATA_FILE = joinpath("Transformers/","BSON_files/","preprocessed_data.bson")



###################### Utils ######################

####### Loading data

function load_preprocessed_data()
    global train_data, val_data, stoi, itos, vocab_size

    # Utiliser les chemins corrigés de Config.jl
    data_loaded = BSON.load(Config.PREPROCESSED_FILE)

    train_data = data_loaded[:"train_data"]
    val_data   = data_loaded[:"val_data"]
    stoi       = data_loaded[:"stoi"]
    itos       = data_loaded[:"itos"]
    vocab_size = data_loaded[:"vocab_size"]
    
    return
end



####### Encoding / Decoding 

function encode(s::AbstractString; stoi=stoi)
    """
    Encode a token(string) as a integer relative to the given dictionnary building during Preprocessing
    Ex: encode("attention") -->  42
    """
    return [get(stoi, lowercase(t), stoi["[UNK]"]) for t in split(s)]
end


function decode(v::AbstractVector{<:Integer}; itos=itos)
    """
    Decode an int into the corresponding token given a the dictionnary build during Preprocessing
    Ex: decode(157) --> "need"
    """

    ignored_tokens = Set(["[EOS]", "[BOS]", "[PAD]", "[UNK]"]) 
    
    words = [itos[i] for i in v]
    filtered_words = filter(w -> w ∉ ignored_tokens, words)

    return join(filtered_words, " ")
end


####### Attention mechanism

struct SingleHeadAttention
    d_model::Int      # input/output dimension
    d_k::Int          # queries/keys dimension
    d_v::Int          # values dimension
end


function scaled_dot_product_attention(Q, K, V, mask=nothing)
    """ 
    Compute head of attention with the given matrix Q, K, V 
    
    @param Q : query matrix
    @param K : key matrix
    @param V : value matrix
    @return: head of attention
    """

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


function create_causal_mask(T, B)
    """
    Create a matrix NxN ( N = B *T ) with:
        - upper triangle = -inf
        - lower triangle = 0 
    """

    N = B * T
    mask = zeros(Float32, N, N)

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



###################### USAGE  ######################

function get_batch(split)
    data = split == "train" ? train_data : val_data
    # Randomly draw starting positions
    ix = rand(1:length(data)-Config.BLOCK_SIZE, Config.BATCH_SIZE)

    # Create the input matrix x (batch_size × BLOCK_SIZE)
    x = [data[i + t] for i in ix, t in 0:Config.BLOCK_SIZE-1]

    # Create the target matrix y (batch_size × BLOCK_SIZE)
    y = [data[i + t + 1] for i in ix, t in 0:Config.BLOCK_SIZE-1]
    return x, y
end


function train_attention_model(;embedding_dimension=16, dim_k=4,dim_v=4, num_iters_to_run=100, learning_rate=0.001, output_file="Transformers/")
    d_model = embedding_dimension 
                          # embedding dimension (reduced from 32)
    d_k = dim_k           # queries/keys dimension (reduced from 8)
    d_v = dim_v           # values dimension (reduced from 8)
    
    global vocab_size
    config = (vocab_size, d_model, d_k, d_v, Config.BLOCK_SIZE, Config.BATCH_SIZE)

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
    
    # dictionnaire de sauvegarde
    model_data = Dict(
        "W_trained" => W_trained,
        "config" => config
    )
    
    BSON.bson(Config.MODEL_FILE, model_data)
    println(" Model successfully saved to $(Config.MODEL_FILE)")

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
    """
    Load a trained model stored in file path 
    """
    println("Loading model from $file_path...")
    
    loaded_data = BSON.load(file_path)
    
    W_trained = loaded_data[:"W_trained"]
    config = loaded_data[:"config"]
    m = ManualAttention(W_trained, config)
    
    println("Model loaded successfully.")
    
    return m, W_trained, config
end


function run_train()
    # --- TRAINING MODE ---
        
    println("\n##################################")
    println("    Starting Training (Single Head Attention)")
    println("##################################")

    load_preprocessed_data()

    train_attention_model(embedding_dimension=Config.EMB_DIM, 
                            dim_k=Config.DIM_K, 
                            dim_v=Config.DIM_V, 
                            num_iters_to_run=Config.N_ITER_DEFAULT, 
                            learning_rate=Config.L_RATE_DEFAULT,
                             
                            )


end

function run_generate(prompt)
    println("\n##################################")
    println("    Starting Generation")
    println("##################################")

    load_preprocessed_data()
    
    m_attention, w_trained, c_config = load_trained_model(Config.MODEL_FILE)
    text_output = generate_text(w_trained, c_config, prompt, max_new_tokens=Config.MAX_TOKEN, temperature=Config.TEMP)
    
    println("prompt: ", prompt, "\n")
    println("\nGenerated Text:\n")
    println(text_output)

end
