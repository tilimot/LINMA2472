include(joinpath(@__DIR__, "flatten.jl"))

# See details about these optimisers in the following course:
# "LINMA2474 - High-Dimensional Data Analysis and Optimization"
import Optimisers

function update!(rule, state, W, ∇)
	state, Δ = Optimisers.apply!(rule, state, W, ∇)
	W .= W .- Δ
end

function update!(rule, states, w::Flatten, ∇::Flatten)
	for (state, W, ∇i) in zip(states.components, w.components, ∇.components)
		update!(rule, state, W, ∇i)
	end
end

function Optimisers.init(rule::Optimisers.AbstractRule, w::Flatten)
	return Flatten(map(W -> Optimisers.init(rule, W), w.components))
end

function train!(gradient!, L, w, num_iters,; rule = Optimisers.Descent(),  losses = [L(w)], states = Optimisers.init(rule, w))
    g = zero(w) # preallocation
    
    # Calculate the step size for 10% progress display
    log_interval = max(1, floor(Int, num_iters / 20))

    for i in 1:num_iters
        ∇ = gradient!(L, g, w)
        update!(rule, states, w, ∇)        
        current_loss = L(w)
        push!(losses, current_loss)

        # Check if the current iteration is a multiple of the log_interval (or the last iteration)
        if (i % log_interval == 0) || (i == num_iters)
            percentage = round(i / num_iters * 100, digits=1)
            println("Iteration $(i)/$(num_iters) [$(percentage)%] - Loss: $(current_loss)")
        end
    end
    return losses, w
end