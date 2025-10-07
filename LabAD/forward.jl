module Forward

struct Dual
    value::Float64
    derivative::Float64
end
Dual(x::Number, y::Number) = Dual(Float64(x), Float64(y))

Base.broadcastable(d::Dual) = Ref(d)
Base.zero(::Dual) = Dual(0, 0)
Base.zero(::Type{Dual}) = Dual(0, 0)
# Addition and subtraction
Base.:+(x::Dual, y::Dual) = Dual(x.value + y.value, x.derivative + y.derivative)
Base.:-(x::Dual, y::Number) = Dual(x.value - y, x.derivative)
Base.:-(x::Dual, y::Dual) = Dual(x.value-y.value, x.derivative-y.derivative)
Base.:-(x::Dual) = Dual(-x.value, -x.derivative)
# Scalar multiplication and division
Base.:*(α::Number, x::Dual) = Dual(α * x.value, α * x.derivative)
Base.:*(x::Dual, α::Number) = Dual(x.value * α, x.derivative * α)
Base.:/(x::Dual, α::Number) = Dual(x.value / α, x.derivative / α)
Base.:/(x::Dual, y::Dual) = Dual(x.value/y.value, (x.derivative*y.value-y.derivative*x.value)/(y.value^2))
# Dual multiplication, division and power
Base.:*(x::Dual, y::Dual) = Dual(x.value * y.value, x.value * y.derivative + x.derivative * y.value)
Base.:^(x::Dual, n::Integer) = Base.power_by_squaring(x, n)
# Specific functions and operations
Base.tanh(d::Dual) = Dual(tanh(d.value), (1-tanh(d.value)^2)*d.derivative)
Base.exp(d::Dual) = Dual(exp(d.value), exp(d.value) * d.derivative)
Base.log(d::Dual) = Dual(log(d.value), d.derivative / d.value)

Base.isless(x::Dual, y::Real) = x.value < y
Base.isless(x::Real, y::Dual) = x < y.value
Base.isless(x::Dual, y::Dual) = x.value < y.value


Base.show(io::IO, d::Dual) = print(io, "Dual(", d.value, ", ", d.derivative, ")")

# ReLU pour un Dual
function relu(d::Dual)
    if d.value > 0
        return Dual(d.value, d.derivative)
    else
        return Dual(0, 0)
    end
end


function onehot(v, i)
    z = zero(similar(v, Float64))
    z[i] = 1.0
    return z
end

function gradient(f, x, i::Integer)
    dx = map(Dual, x, onehot(x, i))
    return f(dx).derivative
end

function gradient!(f, g, x)
    return map!(g, eachindex(x)) do i
        gradient(f, x, i)
    end
end

gradient(f, x) = gradient!(f, zero(x), x)

end
