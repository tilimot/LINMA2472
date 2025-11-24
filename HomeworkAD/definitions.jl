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



function scaled_dot_product_attention(Q, K, V, mask=nothing)
    # Q: d_k × N
    # K: d_k × N  
    # V: d_v × N
    
    d_k = size(Q, 1)
    
    # Scores d'attention: K^T * Q (N × N)
    # On transpose K pour obtenir N × d_k, puis multiplie par Q (d_k × N)
    # Pour transposer K quand c'est un VectNode, on doit faire attention
    # K est d_k × N, donc K' sera automatiquement N × d_k
    scores = transpose(Q) * K  # (N × d_k) * (d_k × N) = N × N
    
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
    output_prime = attn_weights' * V'  # (d_v × N) * (N × N) = d_v × N
    output = transpose(output_prime)

    return output
end



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

function initialize_attention_params(d_model, d_k, d_v)
    scale_qk = sqrt(2.0f0 / (d_model + d_k))
    scale_v = sqrt(2.0f0 / (d_model + d_v))
    scale_o = sqrt(2.0f0 / (d_v + d_model))
    
    Wq = scale_qk .* randn(Float32, d_k, d_model)
    Wk = scale_qk .* randn(Float32, d_k, d_model)
    Wv = scale_v .* randn(Float32, d_v, d_model)
    Wo = scale_o .* randn(Float32, d_model, d_v)
    
    return Flatten([Wq, Wk, Wv, Wo])
end


function positional_encoding(T, d_model)
    PE = zeros(Float32, d_model, T)
    
    # Calculer l'argument de la fonction sin/cos
    for k in 1:d_model
        # k est l'indice de la dimension (paire/impaire)
        div_term = 1f0 / Float32(10000^((k-1) / d_model))
        
        # t est l'indice de position
        for t in 1:T
            if k % 2 == 1 # Indices impairs (sin)
                PE[k, t] = sin(t * div_term)
            else # Indices pairs (cos)
                PE[k, t] = cos(t * div_term)
            end
        end
    end
    return PE
end

