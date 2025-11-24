text = read("HomeworkAD/input.txt", String)   # ou autre

LabAD = joinpath(dirname(@__DIR__), "LabAD")

include(joinpath(LabAD, "test", "test.jl"))

# Reference implementation we test against
include(joinpath(LabAD, "solution", "forward.jl"))
include(joinpath(@__DIR__, "train.jl"))
## First order
include(joinpath(@__DIR__, "reverse_vectorized.jl"))

chars = sort(unique(text))

print(text[10])