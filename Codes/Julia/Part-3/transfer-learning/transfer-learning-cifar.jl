#########################################
#= Transfer Learning (CIFAR10 DATASET) =#
#########################################
# `versioninfo()` -> 1.11.1

using Markdown

using Metalhead
md"Load the pre-trained model"
md"[API Reference](https://fluxml.ai/Metalhead.jl/dev/api/reference/#API-Reference)"
resnet = ResNet(18; pretrain=true).layers;

using Flux
using Flux: onecold, onehotbatch

mdl = Chain(
    resnet[1:end-1],
    resnet[end][1:end-1],
    # Replace the last layer
    Dense(512 => 256, relu),
    Dense(256 => 10)
)

using MLDatasets
md"Load the CIFAR10 dataset"
function get_data(split, lm::Integer=1024)
    data = CIFAR10(split)
    X, y = data.features[:, :, :, 1:lm] ./ 255, onehotbatch(data.targets[1:lm], 0:9)
    loader = Flux.Data.DataLoader((X, y); batchsize=16, shuffle=true)
    return loader
end

train_loader = get_data(:train, 512);
test_loader = get_data(:test, 128);

md"Define a setup of the optimizer"
loss(X, y) = Flux.Losses.logitcrossentropy(mdl(X), y)
opt = Adam(3e-3)
ps = Flux.params(mdl[3:end])

using Flux: @epochs
@epochs 5 Flux.train!(loss, ps, train_loader, opt, cb=Flux.throttle(() -> println("Training"), 10))

using ImageShow, ImageInTerminal
idx = rand(1:50000)
convert2image(d, idx)
printstyled("Label is $(d.targets[idx])"; bold=true, color=:red)

#=
using Optimisers
opt_state = Optimisers.setup(Adam(3e-3), mdl[3:end]) # Freeze the weights of the pre-trained layers
using ProgressMeter
epochs = 5
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
