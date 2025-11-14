
# Forward mode: pushforward
function pushforward_simple(f, x, tx)
    dW = [Dual(x[i], tx[i]) for i in eachindex(x)]
    result = f(dW)
    return [r.derivative for r in result]
end

# Reverse mode: gradient simple
function gradient_simple(f, x)
    # Juste pour l'exemple
    # Calcul numérique du gradient
    return [2*xi for xi in x]  # gradient de sum(x^2)
end

# Jacobienne du Forward
function jacobian_forward(f, x)
    cols = []
    for i in eachindex(x)
        tx = zeros(length(x))
        tx[i] = 1.0
        col = pushforward_simple(f, x, tx)
        push!(cols, col)
    end
    return hcat(cols...)
end

# MAINTENANT LA HESSIENNE:
f(x) = sum(x.^2)
x = [2.0, 3.0]

# Méthode: Forward-on-Reverse
grad_f = z -> gradient_simple(f, z)
H = jacobian_forward(grad_f, x)