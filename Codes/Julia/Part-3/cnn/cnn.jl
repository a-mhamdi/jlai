### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ cd0d7da6-4b16-41c5-9e85-64c01fd0e799
import Pkg; Pkg.activate(".")

# ╔═╡ e63339ed-3e4c-41d4-b47c-a962838ee9a1
using Statistics

# ╔═╡ fc5fdf2d-b83b-4416-a261-cdff48028b9c
using ProgressMeter: Progress, next!

# ╔═╡ 4dc6674f-a81d-4ed4-b242-a797cb7955b1
using Plots; theme(:dracula)

# ╔═╡ 07c27657-9e71-40fd-8a23-ee5e0b319572
using Flux # v"0.16.0"

# ╔═╡ a600067a-d9e9-49d9-b9dc-2fb884e3878e
using Flux: DataLoader

# ╔═╡ 08d58075-de21-4889-abf3-43a8e4e753cc
using Flux: onecold, onehotbatch

# ╔═╡ e91d06f3-ac17-48c8-86a5-76c99badf839
using MLDatasets

# ╔═╡ e27c363a-fdf7-46d9-97ab-fc02b7800713
using ImageShow, ImageInTerminal # ImageView

# ╔═╡ 770d2663-7edf-4eba-ad03-eecf9f0a078e
using BSON: @save

# ╔═╡ 4a1270e4-c751-48f2-af82-380d97ed2800
md"# HANDWRITTEN DIGITS RECOGNITION USING CNN"

# ╔═╡ 5721858c-52df-44df-a258-e727e645cdd6
versioninfo() # -> v"1.11.2"

# ╔═╡ e8cefb2a-bdd2-4626-bbe8-04772e1f169f
md"Handwritten digits classification using **CNN**. This solution is implemented in `Julia` using the `Flux.jl` library"

# ╔═╡ 0ab3f83a-bb49-4da0-9c3d-8ba987ea785e
md"Import the machine learning library `Flux`"

# ╔═╡ e9b447a2-3787-4cb2-b3cd-6bc078e89025
d = MNIST()

# ╔═╡ 2f3fb0c3-6fc1-4e42-ad9c-48160adb797d
Base.@kwdef mutable struct HyperParams
    η = 3f-3                # Learning rate
    batchsize = 64          # Batch size
    epochs = 8              # Number of epochs
    split = :train          # Split data into `train` and `test`
end

# ╔═╡ 149faa4a-52b6-4567-bd59-c4284fc57a85
md"Load the **MNIST** dataset"

# ╔═╡ 2a45a3e4-82f9-45d1-b895-68195849b88c
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

# ╔═╡ eac8ccc8-1912-4b47-8b94-2183cdb0e9e9
train_loader = get_data();

# ╔═╡ a1b726b3-3187-4324-b406-971ed1becf05
test_loader = get_data(split=:test);

# ╔═╡ 6d2ab2d2-4445-4283-aae8-ae762a0e42f9
md"Transform sample training data to an image. View the image and check the corresponding digit value."

# ╔═╡ 3181037a-1679-4bdb-9bde-e5f7fe4560a2
imdx = rand(1:6_000);

# ╔═╡ 0f0b60a4-3b7a-45af-a1af-b48e492fd52f
convert2image(d, imdx) # |> imshow

# ╔═╡ 35ccac3c-bbaf-44f3-9c9c-f99803f9000f
md"**Digit is $(d.targets[imdx])**"

# ╔═╡ 921517f8-727c-4392-9952-42098a894023
md"""
## **CNN** ARCHITECTURE"
The input `X` is a batch of images with dimensions `(width=28, height=28, channels=1, batchsize)`
"""

# ╔═╡ 884d81c9-4c74-4b63-ba5d-e480f41833e8
fc = prod(Int.(floor.([28/4 - 2, 28/4 - 2, 16]))) # 2^{\# max-pool}

# ╔═╡ 24e8e53d-1d9e-4229-bfaa-751d45b76dee
model = Chain(
            Conv((5, 5), 1 => 16, relu),  # (28-5+1)x(28-5+1)x16 = 24x24x16
            MaxPool((2, 2)),              # 12x12x16
            Conv((3, 3), 16 => 16, relu), # (12-3+1)x(12-3+1)x16 = 10x10x16
            MaxPool((2, 2)),              # 5x5x16
            Flux.flatten,                 # 400
            Dense(fc => 64, relu),
            Dense(64 => 32, relu),
            Dense(32 => 10)
)

# ╔═╡ fbf4e3fa-0105-4910-b208-194240d2ee6d
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

# ╔═╡ ea8783f1-2075-4375-89e8-911b9027e84a
vec_loss, vec_acc = train()

# ╔═╡ 8ced717e-6454-4e00-b88d-b9ebb59d6463
md"Plot results"

# ╔═╡ 0ea41544-c09a-4e84-bf93-c0a102c60b9c
plot(vec_loss, label="Test Loss")

# ╔═╡ f0dc1c14-c88f-4e1f-873b-06a6306016df
plot(vec_acc, label="Test Accuracy")

# ╔═╡ 756699be-37e6-4ee5-876f-04f82bc34c15
md"Let's make some predictions"

# ╔═╡ 6a5cff02-2263-4224-a793-f0067775b040
idx = rand(1:1000, 16)

# ╔═╡ 53ccffc2-0e69-4dbe-ad96-228ddffa2b0d
xs, ys = test_loader.data[1][:,:,:,idx], onecold(test_loader.data[2][:, idx]) .- 1

# ╔═╡ a4430df1-51fd-471f-ab28-85fab075c835
yp = onecold(model(xs)) .- 1

# ╔═╡ f1685a5d-dd11-4e23-b84c-7ee626d3148f
for i ∈ eachindex(yp)
    @info "**Prediction is $(yp[i]). Label is $(ys[i]).**"
end

# ╔═╡ f969e980-555c-4870-963c-095c75e0ff79
md"**Save the model**"

# ╔═╡ c8516e32-5fd2-4000-ac96-7ce87132d3b0
@save "cnn.bson" model

# ╔═╡ 548bb92a-b1e2-4245-b234-eb94b941c296
html"""
<style>
	main {
		margin: 0 auto;
		max-width: 2000px;
		padding-left: max(160px, 10%);
		padding-right: max(160px, 10%);
	}
</style>
"""

# ╔═╡ Cell order:
# ╠═4a1270e4-c751-48f2-af82-380d97ed2800
# ╠═5721858c-52df-44df-a258-e727e645cdd6
# ╠═cd0d7da6-4b16-41c5-9e85-64c01fd0e799
# ╠═e8cefb2a-bdd2-4626-bbe8-04772e1f169f
# ╠═e63339ed-3e4c-41d4-b47c-a962838ee9a1
# ╠═fc5fdf2d-b83b-4416-a261-cdff48028b9c
# ╠═4dc6674f-a81d-4ed4-b242-a797cb7955b1
# ╠═0ab3f83a-bb49-4da0-9c3d-8ba987ea785e
# ╠═07c27657-9e71-40fd-8a23-ee5e0b319572
# ╠═a600067a-d9e9-49d9-b9dc-2fb884e3878e
# ╠═08d58075-de21-4889-abf3-43a8e4e753cc
# ╠═e91d06f3-ac17-48c8-86a5-76c99badf839
# ╠═e9b447a2-3787-4cb2-b3cd-6bc078e89025
# ╠═2f3fb0c3-6fc1-4e42-ad9c-48160adb797d
# ╠═149faa4a-52b6-4567-bd59-c4284fc57a85
# ╠═2a45a3e4-82f9-45d1-b895-68195849b88c
# ╠═eac8ccc8-1912-4b47-8b94-2183cdb0e9e9
# ╠═a1b726b3-3187-4324-b406-971ed1becf05
# ╠═6d2ab2d2-4445-4283-aae8-ae762a0e42f9
# ╠═3181037a-1679-4bdb-9bde-e5f7fe4560a2
# ╠═e27c363a-fdf7-46d9-97ab-fc02b7800713
# ╠═0f0b60a4-3b7a-45af-a1af-b48e492fd52f
# ╠═35ccac3c-bbaf-44f3-9c9c-f99803f9000f
# ╠═921517f8-727c-4392-9952-42098a894023
# ╠═884d81c9-4c74-4b63-ba5d-e480f41833e8
# ╠═24e8e53d-1d9e-4229-bfaa-751d45b76dee
# ╠═fbf4e3fa-0105-4910-b208-194240d2ee6d
# ╠═ea8783f1-2075-4375-89e8-911b9027e84a
# ╠═8ced717e-6454-4e00-b88d-b9ebb59d6463
# ╠═0ea41544-c09a-4e84-bf93-c0a102c60b9c
# ╠═f0dc1c14-c88f-4e1f-873b-06a6306016df
# ╠═756699be-37e6-4ee5-876f-04f82bc34c15
# ╠═6a5cff02-2263-4224-a793-f0067775b040
# ╠═53ccffc2-0e69-4dbe-ad96-228ddffa2b0d
# ╠═a4430df1-51fd-471f-ab28-85fab075c835
# ╠═f1685a5d-dd11-4e23-b84c-7ee626d3148f
# ╠═f969e980-555c-4870-963c-095c75e0ff79
# ╠═770d2663-7edf-4eba-ad03-eecf9f0a078e
# ╠═c8516e32-5fd2-4000-ac96-7ce87132d3b0
# ╟─548bb92a-b1e2-4245-b234-eb94b941c296
