### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ 6be46628-a26b-4557-b425-b4f386d420a2
begin
	import Pkg;
	Pkg.activate(".")
	Pkg.status()
end

# ╔═╡ b4bfd97f-3159-4a20-8659-08aacd51adbb
using Metalhead

# ╔═╡ 7667536f-f734-4fe5-b0af-e21963f17ce5
using Flux

# ╔═╡ b1e0de19-d9ae-4255-b114-524b0d606842
using Flux: onecold, onehotbatch

# ╔═╡ 5aa757ee-d8eb-4bc3-bbad-7bf0bd1b15de
using MLDatasets: CIFAR10

# ╔═╡ 482fc576-8808-4734-ad7d-5545964fe5fb
using ImageShow

# ╔═╡ 991dc357-1c32-4b38-a694-422fe9f543da
using MLDatasets: convert2image

# ╔═╡ 16d225ab-bedd-42a6-9b02-5ec8fcca0d1d
md"# Transfer Learning _(CIFAR'10 DATASET)_"

# ╔═╡ 04f5c8f9-2af7-4d83-9051-8e4209b70fd6
versioninfo() # -> v"1.11.2"

# ╔═╡ f87d699d-c7a4-400c-8798-93257b8e69b3
md"Load the pre-trained model"

# ╔═╡ 75236965-94f7-4224-a0bc-9ef4e6c9d6ff
md"[API Reference](https://fluxml.ai/Metalhead.jl/dev/api/reference/#API-Reference)"

# ╔═╡ a28e98f9-04bb-4967-bd80-8bb3078d6301
resnet = ResNet(18; pretrain=true).layers;

# ╔═╡ fded6d2f-1e76-4649-b5cc-4f7b230381de
mdl = Chain(
    resnet[1:end-1],
    resnet[end][1:end-1],
    # Replace the last layer
    Dense(512 => 256, relu),
    Dense(256 => 10)
)

# ╔═╡ a42f0c09-d601-40e0-b844-9b58ba867292
md"Load the CIFAR'10 dataset"

# ╔═╡ fcbfb38d-9b55-4b70-949c-6f4a48c5d24f
d = CIFAR10()

# ╔═╡ 37276043-a28e-4a4b-806b-7abc8a37c2cf
idx = rand(1:50000)

# ╔═╡ 9376c600-2f92-4e79-be02-8d5bac546932
convert2image(d, idx)

# ╔═╡ 70ea7725-04bf-461a-a3ef-95e5d7d2759b
printstyled("Label is $(d.targets[idx])"; bold=true, color=:red)

# ╔═╡ 0666417c-e70b-445e-b9da-52657b50927a
function get_data(split, lm::Integer=1024)
    data = CIFAR10(split)
    X, y = data.features[:, :, :, 1:lm] ./ 255, onehotbatch(data.targets[1:lm], 0:9)
    loader = Flux.DataLoader((X, y); batchsize=16, shuffle=true)
    return loader
end

# ╔═╡ 7a7bf4ad-9438-47a4-88b7-6c0389a7b4c1
train_loader = get_data(:train, 512);

# ╔═╡ 28a4be79-daea-41e8-8f22-e077e83f3bfb
test_loader = get_data(:test, 128);

# ╔═╡ 3ce1a460-dc4b-4eb8-9558-362da1a49de8
md"Define a setup of the optimizer"

# ╔═╡ b2b1aaac-fd7b-4301-ae18-afc9f7a0f3e5
loss(X, y) = Flux.Losses.logitcrossentropy(mdl(X), y)

# ╔═╡ 6c2555fa-47f6-4a05-9d49-e6b18313cd5c
opt = Adam(3e-3)

# ╔═╡ 594fea64-d080-4366-9a90-0c8f6badf01c
ps = Flux.params(mdl[3:end])

# ╔═╡ b53cdab1-1a54-49e2-8fdf-c3d47fc38b34
for epoch in 1:8
    Flux.train!(
		loss, 
		ps, 
		train_loader, 
		opt, 
		cb=Flux.throttle(() -> println("Training"), 10)
	)
end

# ╔═╡ 41a26c99-f97f-4dc9-8dff-9d6421b831fe
begin
	sample = d.features[:, :, :, idx]
	println(size(sample))
	mb_sample = reshape(sample, 32, 32, 3, 1)
	println(size(mb_sample))
end

# ╔═╡ 3319a48f-fc7c-49e3-b7fd-876ce39bdf88
mb_sample |> mdl |> softmax |> onecold

# ╔═╡ e04b3a47-13f2-485e-a03c-178d5358d77f
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

# ╔═╡ Cell order:
# ╠═16d225ab-bedd-42a6-9b02-5ec8fcca0d1d
# ╠═04f5c8f9-2af7-4d83-9051-8e4209b70fd6
# ╠═6be46628-a26b-4557-b425-b4f386d420a2
# ╠═b4bfd97f-3159-4a20-8659-08aacd51adbb
# ╠═f87d699d-c7a4-400c-8798-93257b8e69b3
# ╠═75236965-94f7-4224-a0bc-9ef4e6c9d6ff
# ╠═a28e98f9-04bb-4967-bd80-8bb3078d6301
# ╠═7667536f-f734-4fe5-b0af-e21963f17ce5
# ╠═b1e0de19-d9ae-4255-b114-524b0d606842
# ╠═fded6d2f-1e76-4649-b5cc-4f7b230381de
# ╠═a42f0c09-d601-40e0-b844-9b58ba867292
# ╠═5aa757ee-d8eb-4bc3-bbad-7bf0bd1b15de
# ╠═fcbfb38d-9b55-4b70-949c-6f4a48c5d24f
# ╠═37276043-a28e-4a4b-806b-7abc8a37c2cf
# ╠═482fc576-8808-4734-ad7d-5545964fe5fb
# ╠═991dc357-1c32-4b38-a694-422fe9f543da
# ╠═9376c600-2f92-4e79-be02-8d5bac546932
# ╠═70ea7725-04bf-461a-a3ef-95e5d7d2759b
# ╠═0666417c-e70b-445e-b9da-52657b50927a
# ╠═7a7bf4ad-9438-47a4-88b7-6c0389a7b4c1
# ╠═28a4be79-daea-41e8-8f22-e077e83f3bfb
# ╠═3ce1a460-dc4b-4eb8-9558-362da1a49de8
# ╠═b2b1aaac-fd7b-4301-ae18-afc9f7a0f3e5
# ╠═6c2555fa-47f6-4a05-9d49-e6b18313cd5c
# ╠═594fea64-d080-4366-9a90-0c8f6badf01c
# ╠═b53cdab1-1a54-49e2-8fdf-c3d47fc38b34
# ╠═41a26c99-f97f-4dc9-8dff-9d6421b831fe
# ╠═3319a48f-fc7c-49e3-b7fd-876ce39bdf88
# ╠═e04b3a47-13f2-485e-a03c-178d5358d77f
