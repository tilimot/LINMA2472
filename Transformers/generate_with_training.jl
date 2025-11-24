include(joinpath("utils/", "preprocess.jl"))
include(joinpath("utils/", "transformers.jl"))

# Parameters
N_ITER = 100
L_RATE = 0.01
EMB_DIM = 64
DIM_K = 16
DIM_V = 16
STARTER = "The king said"


# Preprocess the data
println("\n\n\n")
println("##################################")
println("    Beginning of Preprocess")
println("##################################")

run_preprocess()



# Train a models
println("\n\n\n")
println("##################################")
println("    Beginning of training")
println("##################################")

w_trained, c_config = train_attention_model(embedding_dimension=EMB_DIM, dim_k=DIM_K, dim_v= DIM_V, num_iters_to_run=N_ITER, learning_rate=L_RATE)


# Generate results
println("\n\n\n")
println("##################################")
println("    Generation")
println("##################################")

println("")
println("Starter prompt: '", STARTER, "'\n generated text: \n")

text_output = generate_text(w_trained, c_config, STARTER, max_new_tokens=50, temperature=1)
println(text_output)
