###########################################################
#= ANN APPROXIMATER OF LINEAR FUNCTION `y = f(x) = αx+β` =#
###########################################################

using Markdown

md"Import the `Zygote` and `Flux` librairies."
using Zygote
using Flux
md"`Random` to shuffle the data when training."
using Random
md"`Plots` to plot values of the loss function."
using Plots

n, η, epochs = 512, 1f-2, 16
α, β = 32f-1, -23f-1

md"Generate some synthetic data" 
x = 5 .+ randn(Float32, n)
y = α .* x .+ β

md"Construct the **SLP**: single neuron, *(i.e, 1 input, 1 output)*"
f = Dense( 1 => 1, relu)
f.weight, f.bias

md"Setup the optimizer and train the **SLP**"
st = Flux.setup(Adam(η), f)

vec_loss = []
for epoch in 1:epochs
    println("Epoch # $epoch")
    for i in randperm(n)
        loss, ∇ = withgradient(f) do m
            ŷ = m([x[i]]);
            Flux.mse(ŷ, y[i]);
        end
        Flux.update!(st, f, ∇[1]);
        push!(vec_loss, loss)
    end

    println("Coefficient α̂ is $(f.weight[1])")
    println("Coefficient β̂ is $(f.bias[1])")
end

md"Plot the loss vector `vec_loss`"
plot(1:n*epochs, vec_loss, legend=false, title="Loss Function J")
