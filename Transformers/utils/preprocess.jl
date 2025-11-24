using WordTokenizers
using StatsBase
using BSON 


#################### parameters ##################

const DIR = "Transformers/"
const INPUT_FILE = joinpath(DIR,"corpus","input.txt")
const OUTPUT_FILE = joinpath(DIR,"BSON_files","preprocessed_data.bson")

# Minimum frequency threshold for a word to be included in the vocabulary
const MIN_FREQ = 10

# Train/validation split ratio
const TRAIN_RATIO = 0.8

const SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]




#################### utils ##################

function tokenize_and_clean(text::String)
    """
    Take an input text and return it as list of tokens
    Example: 
    tokenize_and_clean("Attention is all you need.") 
        --> ["attention, is", "all", "you", "need", "[EOS]"]

    @param text : a corpus as a long String
    @return list of String tokenized
    """

    text = lowercase(text)
    tokens = tokenize(text)

    # Replace end of sentence symbols by [EOS]
    tokens_processed = String[]
    for token in tokens
        if token in (".", "!", "?", "\n")
            push!(tokens_processed, "[EOS]")
        elseif !isempty(strip(token)) # Ignore spaces and empty tokens
            push!(tokens_processed, token)
        end
    end
    
    return tokens_processed
end


function build_vocab(tokens::Vector{String})
    """
    Build a vocabulary based on the token list received.
    Steps: 
        1. Count the frequency of each tokens
        2. Drop tokens with frequency < MIN_FREQ
        3. Create a filtered vocabulary with Special tokens added
        4. Associate each token to an indice in a dict , and revertebly each indice to its token
    Return the 2 dictionaries and len(vocab)
    """

    # Count token frequencies
    token_counts = countmap(tokens)
    
    # Filter tokens by minimum frequency threshold
    filtered_tokens = Set(k for (k, v) in token_counts if v >= MIN_FREQ)
    
    # Build the final vocabulary (Special Tokens + Filtered Tokens)
    vocab = unique(vcat(SPECIAL_TOKENS, sort(collect(filtered_tokens))))

    # Create encoding dictionaries
    stoi = Dict(s => i for (i, s) in enumerate(vocab))
    itos = Dict(i => s for (i, s) in enumerate(vocab))

    return stoi, itos, length(vocab)
end




function encode_tokens(tokens::Vector{String}, stoi::Dict{String, Int})
    """
    Replace each token by its encoding in the stoi dict 
    Example: 
        ["attention", "is", "all", "you", "need"]
            |           |     |       |      |
        [   42,         4 ,   336,    1024,   157 ] 
    """

    # Index of [UNK] to replace rare tokens
    unk_index = stoi["[UNK]"]
    
    # Encode the full sequence. Tokens not in 'stoi' become [UNK]
    encoded_data = [get(stoi, token, unk_index) for token in tokens]
    
    return encoded_data
end


function run_preprocess()

    """
    Read an input text -, preprocess it and store the results into a .BSON file 
    """

    println("Starting Preprocessing...")
    
    # 1. Read file
    text = read(INPUT_FILE, String)

    # 2. Tokenization and Cleaning
    println("Begin tokenizing...")
    tokens = tokenize_and_clean(text)
    println("Total number of tokens: $(length(tokens))")

    # 3. Vocabulary Building
    println("Begin vocabulary building...")
    stoi, itos, vocab_size = build_vocab(tokens)
    println("Final vocabulary size (V): $vocab_size (after filtering Min Freq = $MIN_FREQ)")

    # 4. Encoding
    println("Begin encoding...")
    data = encode_tokens(tokens, stoi)

    # 5. Train/Val Split
    println("Spliting of data into train and val set...")
    n = floor(Int, TRAIN_RATIO * length(data))
    train_data = data[1:n]
    val_data   = data[n+1:end]
    
    println("Size of training data (tokens): $(length(train_data))")

    # 6. Save results
    println("Saving the result")
    data_to_save = Dict(
        "train_data" => train_data,
        "val_data" => val_data,
        "stoi" => stoi,
        "itos" => itos,
        "vocab_size" => vocab_size
    )
    
    BSON.bson(OUTPUT_FILE, data_to_save)
    println( "Preprocessing finished. Data saved in $OUTPUT_FILE")
end



#################### Preprocessing ##################

# Execute the function
# run_preprocess()