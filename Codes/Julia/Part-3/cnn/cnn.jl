##############################################
#= HANDWRITTEN DIGITS RECOGNITION USING CNN =#
##############################################
# `versioninfo()` -> 1.11.1

using Markdown
md"Handwritten digits classification using **CNN**. This solution is implemented in `Julia` using the `Flux.jl` library"

using Statistics
using ProgressMeter: Progress, next!
using Plots

md"Import the machine learning library `Flux`" 
using Flux # v0.14.25
using Flux: DataLoader
using Flux: onecold, onehotbatch

using MLDatasets
d = MNIST()

Base.@kwdef mutable struct HyperParams
    η = 3f-3                # Learning rate
    batchsize = 64          # Batch size
    epochs = 8              # Number of epochs
    split = :train          # Split data into `train` and `test`
end

md"Load the **MNIST** dataset"
function get_data(; kws...)
    args = HyperParams(; kws...);
    md"Split and normalize data"
    data = MNIST(split=args.split);
    X, y = data.features ./ 255, data.targets;
    X = reshape(X, (28, 28, 1, :));
    y = onehotbatch(y, 0:9);
    loader = DataLoader((X, y); batchsize=args.batchsize, shuffle=true);
    return loader
end

train_loader = get_data();
test_loader = get_data(split=:test);

md"Transform sample training data to an image. View the image and check the corresponding digit value."
idx = rand(1:6_000);
using ImageShow, ImageInTerminal # ImageView
convert2image(d, idx) # |> imshow
md"**Digit is $(d.targets[idx])**"

md"""
## **CNN** ARCHITECTURE"
The input `X` is a batch of images with dimensions `(width=28, height=28, channels=1, batchsize)`
"""

fc = prod(Int.(floor.([28/4 - 2, 28/4 - 2, 16]))) # 2^{\# max-pool}
model = Chain(
            Conv((5, 5), 1 => 16, relu), # (28-5+1)x(28-5+1)x16 = 24x24x16
            MaxPool((2, 2)), # 12x12x16
            Conv((3, 3), 16 => 16, relu), # (12-3+1)x(12-3+1)x16 = 10x10x16
            MaxPool((2, 2)), # 5x5x16
            Flux.flatten, # 400
            Dense(fc => 64, relu),
            Dense(64 => 32, relu),
            Dense(32 => 10)
)

function train(; kws...)
    args = HyperParams(; kws...)
    md"Define the loss function"
    l(α, β) = Flux.logitcrossentropy(α, β)
    md"Define the accuracy metric"
    acc(α, β) = mean(onecold(α) .== onecold(β))
    md"Optimizer"
    optim_state = Flux.setup(Adam(args.η), model);

    vec_loss = []
    vec_acc = []

    for epoch in 1:args.epochs
        printstyled("\t***\t === EPOCH $(epoch) === \t*** \n", color=:magenta, bold=true)
        @info "TRAINING"
        prg_train = Progress(length(train_loader))
        for (X, y) in train_loader
            loss, grads = Flux.withgradient(model) do m
                ŷ = m(X);
                l(ŷ, y);
            end
            Flux.update!(optim_state, model, grads[1]); # Upd `W` and `b`
            # Show progress meter
            next!(prg_train, showvalues=[(:loss, loss)])
        end
        @info "TESTING"
        prg_test = Progress(length(test_loader))
        for (X, y) in test_loader
            ŷ = model(X);
            push!(vec_loss, l(ŷ, y));  # log `loss` value -> `vec_loss` vector
            push!(vec_acc, acc(ŷ, y)); # log `accuracy` value -> `vec_acc` vector
          	# Show progress meter
            next!(prg_test, showvalues=[(:loss, vec_loss[end]), (:accuracy, vec_acc[end])])
        end
    end
    return vec_loss, vec_acc     
end

vec_loss, vec_acc = train()

# Plot results
plot(vec_loss, label="Test Loss")
plot(vec_acc, label="Test Accuracy")

# Let's make some predictions
idx = rand(1:1000, 16)
xs, ys = test_loader.data[1][:,:,:,idx], onecold(test_loader.data[2][:, idx]) .- 1
yp = onecold(model(xs)) .- 1

for i ∈ eachindex(yp)
    @info "**Prediction is $(yp[i]). Label is $(ys[i]).**"
end

# Save the model
using BSON: @save
@save "cnn.bson" model
