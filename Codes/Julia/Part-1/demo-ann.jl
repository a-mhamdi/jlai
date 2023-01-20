#= ANN APPROXIMATER OF LINEAR FUNCTION `y = f(x) = 2x-1` =#

x = rand(256)
y = 2 .* x .+ 1

using Flux

f = Dense( 1 => 1, relu)

f.weight, f.bias

using Zygote

for i in 1:256
        g = gradient(Params([f.weight, f.bias]) do
                         sum(f([x[i
    end

        f.weight -= g[f.weight]
        f.bias -= g[f.bias]
    end


