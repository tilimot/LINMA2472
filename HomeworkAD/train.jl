


text = read("HomeworkAD/input.txt", String)   # ou autre

LabAD = joinpath(dirname(@__DIR__), "LabAD")

include(joinpath(LabAD, "test", "test.jl"))

# Reference implementation we test against
include(joinpath(LabAD, "solution", "forward.jl"))

## First order
include(joinpath(@__DIR__, "reverse_vectorized.jl"))

chars = sort(unique(text))

### ENCODING / DECODING

stoi = Dict(ch => i for (i, ch) in enumerate(chars))  # Char -> Int
itos = Dict(i => ch for (i, ch) in enumerate(chars))  # Int  -> Char


encode(s::AbstractString) = [stoi[c] for c in s]          # String -> Vector{Int}
decode(v::AbstractVector{<:Integer}) = join(itos[i] for i in v)  # Vector{Int} -> String

# Encoder tout le texte

# convertir en vecteur d'entiers (déjà fait par encode)
data = encode(text)

# index du split 90%
n = floor(Int, 0.9 * length(data))

# splits
train_data = data[1:n]
val_data   = data[n+1:end]


### Sample random chunks and sending a chunk at a time to the transformer
### The maximym length = block_size

block_size = 8
train_data[1:block_size+1]

# x = train_data[:block_size]
x = train_data[1:block_size]


y = train_data[2:block_size+1]

for t in 1:block_size
    context = x[1:t]          # x[:t+1] en Python
    target  = y[t]            # y[t]
    println("when input is $(context) the target: $(target)")
end

### Création des batch to process multiple chunks at the same time

using Random

Random.seed!(1337)

batch_size = 4      # combien de séquences en parallèle
block_size = 8      # longueur du contexte

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

# tester
xb, yb = get_batch("train")

println("inputs:")
println(xb)

println("targets:")
println(size(yb))
println(yb)

println("-----")

# reproduction de la boucle Python
for b in 1:batch_size        # dimension batch
    for t in 1:block_size    # dimension temps
        context = xb[b, 1:t]
        target  = yb[b, t]
        println("when input is $(context) the target: $(target)")
    end
end

#### SIMPLE NEURAL NETWORK TO FEED THE INPUT

using Flux
using Flux: onehotbatch
using Random

Random.seed!(1337)

# Chaque token lit directement les logits du token suivant
struct BigramLanguageModel
    token_embedding_table::Embedding
end

Flux.@functor BigramLanguageModel

function BigramLanguageModel(vocab_size::Int)
    # Créer une token_embedding_table (VocabSize x VocabSize) 
    emb = Embedding(vocab_size, vocab_size) # V token possibles de dim d'embedding V
    # emb = matrice de poids ou chaque ligne corresponf à un token d'entrée. 
    # Si je vois le token i le score pour le prochain token à la ième ligne de W
    return BigramLanguageModel(emb)
end

# idx et targets sont des tenseurs (B,T) d'entiers
# B = batch_size
# T = longueur de chaque séquence
function my_cross_entropy_from_logits(logits::AbstractMatrix, y_onehot::AbstractMatrix)
    # logits : C × N   (C = vocab_size, N = batch_size*T)
    # y_onehot : C × N (onehot)
    
    # log-sum-exp pour stabilité numérique
    # max_logits : 1 × N
    max_logits = maximum(logits, dims=1)

    # denom = log( sum_i exp(logits_i - max) )
    lse = max_logits .+ log.(sum(exp.(logits .- max_logits), dims=1))

    # perte = - sum_i y_i * (logits_i - lse)
    # y_onehot .* logits => sélectionne les bons logits
    ce = -sum(y_onehot .* (logits .- lse))

    # normalisation par N (optionnel comme PyTorch)
    return ce / size(logits, 2)
end
function (m::BigramLanguageModel)(idx, targets=nothing)
    # On appelle idx on arrache la idx row of the embedding table
    logits = m.token_embedding_table(idx)      # taille ~ (C, B*T) ou (C, N)
    # chaque entrée dans idx est un entier représentant un token
    # Si W de taille VxV et idx N => VxN
    # si idx matrice (B, T) => VxBxT
    if targets === nothing
        return logits, nothing
    end
    C, B, T = size(logits)
    # print("size logits", size(logits), "\n")
    # on aplatit pour faire comme PyTorch: (C, B*T)
    logits2d = reshape(logits, C, B*T)
    print("size logits_2d: ", size(logits2d), "\n")
    # targets: (B, T) -> vecteur longueur B*T
    targets1d = vec(targets)
    # println("size targets1d: ", size(targets1d))
    # one-hot des classes (C, B*T)
    y = onehotbatch(targets1d, 1:C)
    println("size target_onehot: ", size(y))
    # cross entropy sur logits bruts (numériquement stable)
    loss_ce = my_cross_entropy_from_logits(logits2d, y)

    return logits, loss_ce
    end

using StatsBase

# take a BxT and transform it into a Bx(T+1), Bx(T+2),... jusque max_new_tokens
function generate(m::BigramLanguageModel, idx, max_new_tokens)
    # idx est (B, T)
    for _ in 1:max_new_tokens

        # forward pass
        logits, _ = m(idx)
        # println("size logits: ", size(logits))
        # on garde uniquement le dernier time-step : logits[:, :, end]
        # en PyTorch: logits = logits[:, -1, :]
        last_logits = logits[:, :, end]      # (C, B)
        # println("last logits dans generate:", size(last_logits))
        # transpose pour avoir (B, C) comme PyTorch
        last_logits = permutedims(last_logits, (2,1))   # (B, C)
        # println("last logits dans generate permuté:", size(last_logits))
        # softmax
        probs = softmax(last_logits)            # (B, C)
        # println("size probs: ", size(probs))
        # sample : pour chaque batch on tire un token selon la distribution
        B, C = size(probs)
        # print("size idx: ", size(idx))
        idx_next = similar(idx[:,1:1])  # (B,1)

        for b in 1:B
            p = probs[b, :]
            # choisir un élément de 1 à C en respectant la proba
            idx_next[b, 1] = sample(1:C, Weights(p))
        end

        # concat avec la séquence
        idx = hcat(idx, idx_next)   # (B, T+1)
    end

    return idx
end


# exemple d'appel, comme en PyTorch
vocab_size = length(chars)
m = BigramLanguageModel(vocab_size)

# xb et yb : matrice d'Int (B,T) avec valeurs dans 1:vocab_size
# B = batch_size
# T = longueur de chaque séquence
println("inputs:")
println("xb", xb)
println("targets:")
println("yb", yb)
println("size of xb: ", size(xb))
println("size of yb: ", size(yb))
println("-----")
logits, loss_ = m(xb, yb)
# println("size out (C, B, T): ", size(out))
println("loss: ", loss_)

max_new_tokens = 100
idx = generate(m, xb, max_new_tokens)

print(decode(idx[1,:]))

# C'est pas bien car total random...
# On regarde que le dernier charac...
# The history is not used



#=
batch_size = 32
for step in 1:100
    xb, yb = get_batch("train")

    logits, loss = m(xb, yb)

    

    print(loss)
end
=#