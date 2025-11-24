include("transformers.jl")

# Parameters
N_ITER = 100
L_RATE = 0.01
EMB_DIM = 64
DIM_K = 16
DIM_V = 16


# Train a models
println("\n\n\n")
println("##################################")
println("    Beginning of training")
println("##################################")

w_trained, c_config = train_attention_model(embedding_dimension=EMB_DIM, dim_k=DIM_K, dim_v= DIM_V, num_iters_to_run=N_ITER, learning_rate=L_RATE)
