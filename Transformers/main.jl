# utils/run.jl
using BSON
using Optimisers
using Printf

# --- Project Dependencies (Ensure paths are correct) ---

include(joinpath("utils/","preprocess.jl"))
include(joinpath("utils/","transformers.jl"))

include(joinpath("utils/", "config.jl"))
using .Config


function main(args)
    
    if isempty(args) || args[1] ∉ ["train", "generate", "preprocess"]
        println("Usage: julia main.jl [preprocess  | train | generate | all ] [prompt_for_generate]")
        println("Example (Train): julia main.jl train")
        println("Example (Generate): julia main.jl generate \"The queen said\"")
        println("Exemple (All) julia main.jl all \"The king said\"")
        return
    end

    command = lowercase(args[1])
    
    if command == "preprocess"
        run_preprocess()
        return
    end

    # Ensure Preprocessing is done for TRAIN/GENERATE 
    if !isfile(Config.PREPROCESSED_FILE)
        println("Error: Preprocessed data not found at $(Config.PREPROCESSED_FILE).")
        println("Run: julia run.jl preprocess")
        return
    end
    
    # --- Execute Command ---
    if command == "train"
        # --- TRAINING MODE ---
        
        println("\n##################################")
        println("    Starting Training (Single Head Attention)")
        println("##################################")

        train_attention_model(Config.EMB_DIM, 
                              Config.DIM_K, 
                              Config.DIM_V, 
                              Config.N_ITER_DEFAULT, 
                              Config.L_RATE_DEFAULT,
                              Config.MODEL_FILE)

    elseif command == "generate"
        # --- GENERATION MODE ---
        
        if !isfile(Config.MODEL_FILE)
            println("Error: Trained model weights not found at $(Config.MODEL_FILE).")
            println("Run: julia run.jl train")
            return
        end
        
        # Load model and get weights/config
        m_attention, w_trained, c_config = load_trained_model(Config.MODEL_FILE)
        
        # Define prompt
        prompt = length(args) > 1 ? join(args[2:end], " ") : ""
        
        println("\n##################################")
        println("    Starting Generation")
        println("##################################")
        println("Prompt: '$(prompt)'")
        
        # Generation
        text_output = generate_text(w_trained, c_config, prompt, max_new_tokens=100, temperature=1.0)
        
        println("\nGenerated Text:\n")
        println(text_output)

    end
end

main(ARGS)