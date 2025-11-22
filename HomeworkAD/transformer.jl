text = read("HomeworkAD/input.txt", String)   # ou autre

LabAD = joinpath(dirname(@__DIR__), "LabAD")

include(joinpath(LabAD, "test", "test.jl"))

# Reference implementation we test against
include(joinpath(LabAD, "solution", "forward.jl"))
include(joinpath(@__DIR__, "train.jl"))
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
println(xb, size(xb))

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

using Flux
using Flux: onehotbatch
using Random

Random.seed!(1337)

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

function bigram_activation(Wflat::Flatten, Xb)
    W = Wflat.components[1]          # C×C (Float32 ou VectNode)


    idx = vec(Xb)                    # longueur N = B*T
    N = length(idx)

    # one-hot (Float32 constants)
    OH = zeros(Float32, C, N)
    @inbounds for (j, i) in enumerate(idx)
        OH[i, j] = 1f0
    end

    # logits = W * onehot
    # si W est Matrix{VectNode}, le produit garde le graphe
    return W * OH                    # C×N = C×(B*T)
end
C = vocab_size
W = 0.01f0 .* randn(Float32, C, C)
W = Flatten([W])
targets1d = vec(yb)         # longueur B*T
y_onehot = onehotbatch(targets1d, 1:C)  # C × (B*T)

L = loss(mse, bigram_activation, xb, y_onehot)

losses, Wtrained = train!(VectReverse.gradient!, L, W)

##############################
# Use trained W for generation
##############################


# Petit modèle manuel qui utilise W
struct ManualBigram
    W::Matrix{Float32}  # C×C
end

function (m::ManualBigram)(idx)
    # idx : (B,T) Int
    idx_vec = vec(idx)
    logits2d = m.W[:, idx_vec]          # C×(B*T)
    B, T = size(idx)
    return reshape(logits2d, C, B, T)   # C×B×T
end

function generate(m::ManualBigram, idx, max_new_tokens)
    for _ in 1:max_new_tokens
        logits = m(idx)                 # C×B×T
        last_logits = logits[:, :, end] # C×B
        last_logits = permutedims(last_logits, (2,1))  # B×C

        probs = softmax(last_logits)    # B×C
        Bsize, Csize = size(probs)
        idx_next = similar(idx[:,1:1])  # B×1

        for b in 1:Bsize
            p = probs[b, :]
            idx_next[b, 1] = sample(1:Csize, Weights(p))
        end

        idx = hcat(idx, idx_next)
    end
    return idx
end

print(Wtrained)
m_trained = ManualBigram(Wtrained.components[1])

max_new_tokens = 200
idx_gen = generate(m_trained, xb, max_new_tokens)
println(decode(idx_gen[1, :]))
