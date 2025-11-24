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

function create_causal_mask(T, B)
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


