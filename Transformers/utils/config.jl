# utils/config.jl
module Config

# Preprocessing parameters 
const MIN_FREQ = 10
const TRAIN_RATIO = 0.8

# Model parameters
const EMB_DIM = 32          # d_model (Embedding Dimension)
const DIM_K = 8             # d_k (Keys/Queries Dimension)
const DIM_V = 8             # d_v (Values Dimension)

# Training parameters
const BLOCK_SIZE = 32       # T (Context Length)
const BATCH_SIZE = 16       # B (Number of sequences per batch)
const N_ITER_DEFAULT = 100 # Default number of iterations
const L_RATE_DEFAULT = 0.01 # Learning rate

# File Paths 
const BASE_DIR = joinpath(@__DIR__, "..")
const CORPUS_FILE = joinpath(BASE_DIR, "corpus", "input.txt")
const PREPROCESSED_FILE = joinpath(BASE_DIR, "BSON_files", "preprocessed_data.bson")
const MODEL_FILE = joinpath(BASE_DIR, "BSON_files", "single_head_transformer_weights.bson")

# Text generation
const MAX_TOKEN = 50
const TEMP = 1.0

end