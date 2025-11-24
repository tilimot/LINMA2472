include(joinpath("utils/", "transformers.jl"))


STARTER = "The king said"

# Generate results
println("\n\n\n")
println("##################################")
println("    Generation")
println("##################################")

println("")
println("Starter prompt: '", STARTER, "'\n generated text: \n")

text_output = generate_text(w_trained, c_config, STARTER, max_new_tokens=50, temperature=1)
println(text_output)
