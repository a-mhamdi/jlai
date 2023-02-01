#########################################
#= Transfer Learning (CIFAR10 DATASET) =#
#########################################

using Markdown

using Metalhead
md"Load the pre-trained model"
md"[API Reference](https://fluxml.ai/Metalhead.jl/dev/api/reference/#API-Reference)"
vgg = VGG(19)

using Flux
using Flux: onecold, onehotbatch

mdl = Chain(
    resnet.layers[1:end-1],
    # Dense(16 => 12, relu),
    # Replace the last layer
    Dense(512, 10),
    # softmax
)

using MLDatasets
md"Load the CIFAR10 dataset"
function get_data(split)
    data = CIFAR10(split)
    X, y = data.features ./ 255, onehotbatch(data.targets, 0:9);
    loader = Flux.Data.DataLoader((X, y); batchsize=512, shuffle=true);
    return loader
end

train_loader = get_data(:train);
test_loader = get_data(:test);

md"Define a setup of the optimizer"
loss(X, y) = Flux.Losses.logitcrossentropy(mdl(X), y)
opt = Adam(3e-3)
ps = Flux.params(mdl)

@epochs 5 Flux.train!(loss, ps, train_loader, opt) # , cb = throttle(() -> println("training"), 10))

#=
opt_state = Optimisers.setup(Adam(3e-3), model[end]) # Freeze the weights of the pre-trained layers
using ProgressMeter
epochs = 3
# Fine-tune the model
for epoch in 1:epochs
    @showprogress for (X, y) in train_loader
        # Compute the gradient of the loss with respect to the model's parameters
        ∇ = Flux.gradient( m -> loss(m, X, y), mdl)
        # Update the `mdl`'s parameters
        Flux.update!(opt_state, mdl, ∇[1])
    end
    @info "Calculate the accuracy on the test set"
    for (X, y) in test_loader
        accuracy = sum(onecold(mdll(X)) .== onecold(y)) / length(y)
        println("Epoch: $epoch, Accuracy: $accuracy")
    end
end
=#
