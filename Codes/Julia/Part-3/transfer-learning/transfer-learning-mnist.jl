### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# ╔═╡ b9318360-c3de-49ee-9512-c2fe35d92f28
# NOT WORKING YET

#######################################
#= Transfer Learning (MNIST DATASET) =#
#######################################
# Gray Color Images , i.e., n° of channnels = 1
# `versioninfo()` -> 1.11.1

using Markdown

# ╔═╡ f96fceaf-1c64-4b57-bbe4-1bf61dcfcc2d
using Metalhead

# ╔═╡ 56823b33-28c7-44e3-ac47-bcad3f926533
using DataAugmentation

# ╔═╡ 4ca61be4-c2e0-48f4-8a41-a42b7e46ad18
using Flux

# ╔═╡ 94931cdd-0cea-4b03-b72c-9c427827ce3f
using Flux: onecold, onehotbatch

# ╔═╡ be79e64f-3563-4f30-8898-819b3e404043
using MLDatasets

# ╔═╡ f069c872-8b2e-42e8-b7b9-6ed2a9a86527
using Images

# ╔═╡ 050f9987-b0fe-4cc8-9e2c-2eb330a2524f
using ProgressMeter

# ╔═╡ 6c26901a-6d9d-46a8-a71f-a4aa784314fd
md"Load the pre-trained model"

# ╔═╡ b25dc1d7-e774-4841-b2b7-5d45499aad46
md"[API Reference](https://fluxml.ai/Metalhead.jl/dev/api/reference/#API-Reference)"

# ╔═╡ 971420b9-89eb-49e7-aa2c-f77b4005be80
vgg = VGG(16; pretrain=true).layers;

# ╔═╡ ff01b1f5-dc08-4c91-9cc6-2c869b150d79
tfm = CenterCrop((224, 224)) |> ImageToTensor()

# ╔═╡ 72e32070-1e73-4ac4-9877-e4b33ad30f75
model = Chain(
    vgg[1:end-1],
    vgg[end][1:end-1],
    # Replace the last layer
    Dense(4096, 256),
    Dense(256, 10)
)

# ╔═╡ 0cccc901-72e6-422c-819f-a51a06c53af0
md"Load the MNIST dataset"

# ╔═╡ fd67d27e-79fb-4d87-891f-76389f5007a7
function get_data(split)
    data = MNIST(split)
    imgs, y = data.features ./ 255, onehotbatch(data.targets, 0:9);
    X = []
    for i in 1:length(y)
        img = apply(tfm, Image(RGB.(imgs[:,:,i]))) |> itemdata
        push!(X, img)
    end
    loader = Flux.Data.DataLoader((X, y); batchsize=64, shuffle=true);
    return loader
end

# ╔═╡ 567a10e4-1452-42f9-8963-c509e359f849
train_loader = get_data(:train);

# ╔═╡ c1a846db-f1a4-41d6-8d47-e7f2e972941a
test_loader = get_data(:test);

# Define a loss function and an optimizer

# ╔═╡ 081bf283-d13e-415a-a2a8-f5e56cf786a4
loss_fn = Flux.logitcrossentropy

# ╔═╡ 64849012-8098-430d-9d4c-5178cf36b9ac
opt_state = Flux.setup(Adam(3e-3), model[end]) # Freeze the weights of the pre-trained layers

# ╔═╡ 97c8b17e-a1c8-4f96-a1b2-3476183418b1
epochs = 3
# Fine-tune the model

# ╔═╡ f41f22d7-ebb8-4383-814e-1962909af5b9
for epoch in 1:epochs
    @showprogress for (X, y) in train_loader
        # Compute the gradient of the loss with respect to the model's parameters
        loss, ∇ = Flux.withgradient(model) do m
            ŷ = m(X)
            loss_fn(ŷ, y)
        end
        # Update the model's parameters using the optimizer
        Flux.update!(opt_state, model, ∇[1])
    end
    @info "Calculate the accuracy on the test set"
    for (X, y) in test_loader
        accuracy = sum(onecold(model(X)) .== onecold(y)) / length(y)
        println("Epoch: $epoch, Accuracy: $accuracy")
    end
end

# ╔═╡ Cell order:
# ╠═b9318360-c3de-49ee-9512-c2fe35d92f28
# ╠═f96fceaf-1c64-4b57-bbe4-1bf61dcfcc2d
# ╠═6c26901a-6d9d-46a8-a71f-a4aa784314fd
# ╠═b25dc1d7-e774-4841-b2b7-5d45499aad46
# ╠═971420b9-89eb-49e7-aa2c-f77b4005be80
# ╠═56823b33-28c7-44e3-ac47-bcad3f926533
# ╠═ff01b1f5-dc08-4c91-9cc6-2c869b150d79
# ╠═4ca61be4-c2e0-48f4-8a41-a42b7e46ad18
# ╠═94931cdd-0cea-4b03-b72c-9c427827ce3f
# ╠═72e32070-1e73-4ac4-9877-e4b33ad30f75
# ╠═be79e64f-3563-4f30-8898-819b3e404043
# ╠═0cccc901-72e6-422c-819f-a51a06c53af0
# ╠═fd67d27e-79fb-4d87-891f-76389f5007a7
# ╠═f069c872-8b2e-42e8-b7b9-6ed2a9a86527
# ╠═567a10e4-1452-42f9-8963-c509e359f849
# ╠═c1a846db-f1a4-41d6-8d47-e7f2e972941a
# ╠═081bf283-d13e-415a-a2a8-f5e56cf786a4
# ╠═64849012-8098-430d-9d4c-5178cf36b9ac
# ╠═050f9987-b0fe-4cc8-9e2c-2eb330a2524f
# ╠═97c8b17e-a1c8-4f96-a1b2-3476183418b1
# ╠═f41f22d7-ebb8-4383-814e-1962909af5b9
