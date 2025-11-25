# utils/run.jl
using BSON
using Optimisers
using Printf
#using JoinPaths

include(joinpath("utils/","models.jl"))
include(joinpath("utils/","train.jl"))
include(joinpath("utils/","flatten.jl"))
include(joinpath("utils/","forward.jl"))
include(joinpath("utils/","reverse_vectorized.jl"))


include(joinpath("utils/","preprocess.jl"))
include(joinpath("utils/","transformers.jl"))

include(joinpath("utils/", "config.jl"))
using .Config


function main(args)
    
    if isempty(args) || args[1] ∉ ["train", "generate", "preprocess", "all"]
        println("Usage: julia run.jl [preprocess  | train | generate | all ] [prompt_for_generate]")
        println("Example (Train): julia run.jl train")
        println("Example (Generate): julia run.jl generate \"the queen said\"")
        println("Exemple (All) julia run.jl all \"the king said\"")
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
        run_train()
        return

    elseif command == "generate"
        # GENERATION MODE 
        
        if !isfile(Config.MODEL_FILE)
            println("Error: Trained model weights not found at $(Config.MODEL_FILE).")
            println("Run: julia run.jl train")
            return
        end
        
        # Define prompt
        prompt = length(args) > 1 ? join(args[2:end], " ") : ""
        run_generate(prompt)

    end

    if command == "all"
        prompt = length(args) > 1 ? join(args[2:end], " ") : ""
        
        run_preprocess()
        run_train()
        run_generate(prompt)
    end
end

main(ARGS)