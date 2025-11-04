using CSV
using DataFrames
using BenchmarkTools

# Charger le module VectReverse
include("reverse_vectorized.jl")
using .VectReverse: VectNode, relu_activation, backward!, Flatten

# ---------------------------
# Fonction de test pour un réseau de Flatten
# ---------------------------
function test_network(flat_layers::Vector{Flatten})
    # On part du premier Flatten
    current = flat_layers[1]
    
    # Propagation couche par couche
    for layer in flat_layers[2:end]
        # Multiplication élément-wise entre chaque composant
        current = Flatten([VectNode(sum(relu_activation.(a.value .* b.value))) 
                            for (a,b) in zip(current.components, layer.components)])
    end
    
    # Somme finale pour produire un VectNode unique
    return VectNode(sum([c.value for c in current.components]))
end

# ---------------------------
# Générateur de Flatten aléatoire
# ---------------------------
function random_flatten(n_components::Int, comp_size::Int)
    comps = [VectNode(randn(comp_size)) for _ in 1:n_components]
    return Flatten(comps)
end

# ---------------------------
# Paramètres des tests
# ---------------------------
n_layers_list = [2, 3, 5]               # Nombre de couches
n_components_list = [1, 5, 10]         # Nombre de VectNode par couche
component_sizes = [10, 100, 500]       # Taille de chaque composant

# Préparer DataFrame pour stocker les résultats
results = DataFrame(
    n_layers = Int[],
    n_components = Int[],
    comp_size = Int[],
    total_size = Int[],
    time_ms = Float64[]
)

# ---------------------------
# Boucle de benchmark
# ---------------------------
for n_layers in n_layers_list
    for n_comp in n_components_list
        for c_size in component_sizes
            println("Testing network: $n_layers layers, $n_comp components, size $c_size")

            # Générer un réseau de Flatten
            network = [random_flatten(n_comp, c_size) for _ in 1:n_layers]
            total_size = sum(length(c.value) for flt in network for c in flt.components)

            # Mesure forward + backward
            t = @belapsed backward!(test_network($network))

            println("Time: $(t*1000) ms, total elements = $total_size")
            push!(results, (n_layers, n_comp, c_size, total_size, t*1000))
        end
    end
end

# ---------------------------
# Sauvegarde CSV
# ---------------------------
CSV.write("vectreverse_perf_network.csv", results)
println("Results saved to vectreverse_perf_network.csv")
