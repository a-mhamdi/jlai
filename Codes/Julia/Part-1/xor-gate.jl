#= XOR GATE =#

# This code considers the XOR problem. In order to be able to run it, simply execute `julia -e "import Pkg; Pkg.activate(\"\"); include(\"xor-gate.jl\")"`

using Flux

#=
x = -1:.1:1
n = length(x)
x1 = x2 = Array{Float64}[];
for i in 1:n
	x1 = vcat(x1, x[i].*ones(n, 1))
	x2 = vcat(x2, x)
end
X = hcat(x1, x2)
=#

## Create the dataset for an "XOR" problem
X = rand(Float32, 2, 1_024);
# vscodedisplay(X, "X")
y = [xor(col[1]>.5, col[2]>.5) for col in eachcol(X)]
# vscodedisplay(y, "y")
yoe = Flux.onehotbatch(y, [true, false])

using Plots; # unicodeplots()

sc = scatter(X[1,:], X[2,:], group=y; labels=["False" "True"])
loader = Flux.Data.DataLoader((X, yoe), batchsize=64, shuffle=true)

## `mdl` is the model to be built 
mdl = Chain(Dense(2 => 3, tanh),
        BatchNorm(3),
	Dense(3 => 2),
        softmax)

## Raw output before training
y_raw = mdl(X)

## `opt` designates the optimizer
opt = Adam(.01)

## `state` contains all trainable parameters
state = Flux.setup(opt, mdl)

#= TRAINING PHASE =#
vec_loss = []

using ProgressMeter
@showprogress for epoch in 1:1_000
    for (Features, target) in loader
		# Begin a gradient context session
        loss, grads = Flux.withgradient(mdl) do m
            # Evaluate model:
            target_hat = m(Features)
			# Evaluate loss:
            Flux.crossentropy(target_hat, target)
        end
        Flux.update!(state, mdl, grads[1])
        push!(vec_loss, loss)  # Log `loss` to `losses` vector `vec_loss`
    end
end

## Predicted output after being trained
y_hat = mdl(X)
y_pred = (y_hat[1, :] .> .5)

## Accuracy: How much we got right over all cases, i.e., (TP+TN)/(TP+TN+FP+FN)
accuracy = Flux.Statistics.mean( (y_pred .> .5) .== y )

## Plot loss vs. iteration
plot(vec_loss; 
    xaxis=(:log10, "Iteration"),
    yaxis="Loss",
    label="Per Batch")

sc1 = scatter(X[1,:], X[2,:], group=yoe[1,:];
    title="TRUTH", labels=["False" "True"])
sc2 = scatter(X[1,:], X[2,:], zcolor=y_raw[1,:];
    title="BEFORE", label=:none, clims=(0,1))
sc3 = scatter(X[1,:], X[2,:], group=y_pred;
    title="AFTER", labels=["False" "True"])

plot(sc1, sc3, layout=(1,2), size=(512,512))
