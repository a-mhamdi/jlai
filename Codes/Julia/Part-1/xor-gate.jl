##############
#= XOR GATE =#
##############
# `versioninfo()` -> v"1.11.1"

using Markdown

md"This code considers the XOR problem. In the terminal, run `julia -e 'import Pkg; Pkg.activate(\".\"); include(\"Part-1/xor-gate.jl\")'`"

using Flux

md"Create the dataset for an \"XOR\" problem"
X = rand(Float32, 2, 1_024);
# vscodedisplay(X, "X")
y = [xor(col[1]>.5, col[2]>.5) for col in eachcol(X)]
# vscodedisplay(y, "y")

md"Scatter plot of `X`"
using Plots; # unicodeplots()
sc = scatter(X[1,:], X[2,:], group=y; labels=["False" "True"])
loader = Flux.DataLoader((X, y), batchsize=32, shuffle=true)

md"`mdl` is the model to be built"
mdl = Chain(Dense( 2 => 4, tanh ),
            Dense( 4 => 4, tanh ),
            Dense( 4 => 1, σ ),
            )

md"Raw output before training"
y_raw = mdl(X)

md"`opt` designates the optimizer"
opt = Adam(.01)

md"`state` contains all trainable parameters"
state = Flux.setup(opt, mdl)

md"## TRAINING PHASE"
vec_loss = []
using ProgressMeter
@showprogress for epoch in 1:1_000
    for (Features, target) in loader
		# Begin a gradient context session
        loss, grads = Flux.withgradient(mdl) do m
            # Evaluate model:
            target_hat = m(Features) |> vec # loss function expects size(ŷ) = (1, :) to match size(y) = (:,)
			# Evaluate loss:
            Flux.binarycrossentropy(target_hat, target)
        end
        Flux.update!(state, mdl, grads[1])
        push!(vec_loss, loss)  # Log `loss` to `losses` vector `vec_loss`
    end
end

md"Predicted output after being trained"
y_hat = mdl(X)
y_pred = (y_hat[1, :] .> .5)

md"Accuracy: How much we got right over all cases _(i.e., (TP+TN)/(TP+TN+FP+FN))_"
accuracy = Flux.Statistics.mean( (y_pred .> .5) .== y )

md"Plot loss vs. iteration"
plot(vec_loss; xaxis=(:log10, "Iteration"), yaxis="Loss", label="Per Batch")
sc1 = scatter(X[1,:], X[2,:], group=y; title="TRUTH", labels=["False" "True"])
sc2 = scatter(X[1,:], X[2,:], zcolor=y_raw[1,:]; title="BEFORE", label=:none, clims=(0,1))
sc3 = scatter(X[1,:], X[2,:], group=y_pred; title="AFTER", labels=["False" "True"])

md"Plot of both ground truth and results after training"
plot(sc1, sc3, layout=(1,2), size=(512,512))
