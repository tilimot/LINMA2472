using WordTokenizers # Requires Pkg.add("WordTokenizers")
using StatsBase
using BSON # Requires Pkg.add("BSON")

# --- Configuration ---
const DIR = "Transformers/"
const INPUT_FILE = joinpath(DIR,"input.txt")
const OUTPUT_FILE = joinpath(DIR,"preprocessed_data.bson")

# Minimum frequency threshold for a word to be included in the vocabulary
const MIN_FREQ = 10

# Train/validation split ratio
const TRAIN_RATIO = 0.8

# --- Special Tokens ---
# Special tokens must be the first in our vocabulary to have low indices
const SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]

# --- 1. Load and Tokenize (using WordTokenizers) ---

function tokenize_and_clean(text::String)
    # 1. Lowercase and Tokenize
    text = lowercase(text)
    # Use WordTokenizers for robust word/punctuation separation
    tokens = tokenize(text)

    # 2. Replace end-of-sentence symbols with [EOS]
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

# --- 2. Vocabulary Building and Filtering ---

function build_vocab(tokens::Vector{String})
    # Count token frequencies
    token_counts = countmap(tokens)
    
    # Filter tokens by minimum frequency threshold
    # Ensure special tokens are not filtered
    filtered_tokens = Set(k for (k, v) in token_counts if v >= MIN_FREQ)
    
    # Build the final vocabulary (Special Tokens + Filtered Tokens)
    vocab = unique(vcat(SPECIAL_TOKENS, sort(collect(filtered_tokens))))

    # Create encoding dictionaries
    stoi = Dict(s => i for (i, s) in enumerate(vocab))
    itos = Dict(i => s for (i, s) in enumerate(vocab))

    return stoi, itos, length(vocab)
end

# --- 3. Text Encoding ---

function encode_tokens(tokens::Vector{String}, stoi::Dict{String, Int})
    # Index of [UNK] to replace rare tokens
    unk_index = stoi["[UNK]"]
    
    # Encode the full sequence. Tokens not in 'stoi' become [UNK]
    encoded_data = [get(stoi, token, unk_index) for token in tokens]
    
    return encoded_data
end

# --- Main Preprocessing and Saving Function ---

function run_preprocess()
    println("--- Starting Preprocessing (WordTokenizers) ---")
    
    # 1. Read file
    text = read(INPUT_FILE, String)

    # 2. Tokenization and Cleaning
    tokens = tokenize_and_clean(text)
    println("Total number of tokens: $(length(tokens))")

    # 3. Vocabulary Building
    stoi, itos, vocab_size = build_vocab(tokens)
    println("Final vocabulary size (V): $vocab_size (after filtering Min Freq = $MIN_FREQ)")
    if vocab_size > 10000
        println("⚠️ WARNING: Vocabulary size ($vocab_size) is high. This will significantly slow down training on CPU.")
        println("Consider increasing MIN_FREQ to reduce V.")
    end

    # 4. Encoding
    data = encode_tokens(tokens, stoi)

    # 5. Train/Val Split
    n = floor(Int, TRAIN_RATIO * length(data))
    train_data = data[1:n]
    val_data   = data[n+1:end]
    
    println("Size of training data (tokens): $(length(train_data))")

    # 6. Save results
    data_to_save = Dict(
        "train_data" => train_data,
        "val_data" => val_data,
        "stoi" => stoi,
        "itos" => itos,
        "vocab_size" => vocab_size
    )
    
    BSON.bson(OUTPUT_FILE, data_to_save)
    println("--- Preprocessing finished. Data saved in $OUTPUT_FILE ---")
end

# Execute the function (this should be run once before the main training script)
run_preprocess()