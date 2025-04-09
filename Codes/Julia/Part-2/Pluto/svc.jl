### A Pluto.jl notebook ###
# v0.20.5

using Markdown
using InteractiveUtils

# ╔═╡ 5ae532ad-2454-413f-b4de-0d4dbac809d3
begin
	cd(@__DIR__)
	import Pkg; Pkg.activate(".."); Pkg.status()
end

# ╔═╡ 316c62ec-deb1-48e0-a29b-fdec424e3851
using CSV, DataFrames

# ╔═╡ 5bdc592b-4f17-461f-8792-991b49664d34
using MLJ

# ╔═╡ 5e4f2880-e058-48f0-80e6-70317c801e14
using Plots

# ╔═╡ 89bb4654-93c8-4db8-8d24-108446b0c2f6
md"# SUPPORT VECTOR MACHINE FOR CLASSIFICATION"

# ╔═╡ efd07dc7-ca06-4f1c-8dd4-a5f8be367d69
versioninfo()

# ╔═╡ 19ae1eb6-9a1a-4cc5-a315-64d43a49855f
md"This an example of how we implemented an `SVM` in _Julia_ for classification task using the `LIBSVM` package interfacing with `MLJ` module."

# ╔═╡ 9a98ff5e-8e82-456f-85d8-28b312c07c8d
md"Import librairies"

# ╔═╡ e618de37-70f1-4da0-95e7-666970f8fe79
md"Read data from CSV file"

# ╔═╡ 9effc32b-7c9f-434b-97b0-1f5bae762cbb
df = CSV.read("../../Datasets/Social_Network_Ads.csv", DataFrame)

# ╔═╡ 2a8c82a9-32f6-4b90-9d65-85393d0eaa30
schema(df)

# ╔═╡ d74f8f38-147b-4e95-b397-cea1d8cb1f4c
coerce!(df, 
        :Age => Continuous,
        :EstimatedSalary => Continuous,
        :Purchased => OrderedFactor)

# ╔═╡ a25abea7-82b9-4681-b256-896a45d97eef
schema(df)

# ╔═╡ b0714705-eef3-4064-9748-626793bcc881
md"Unpack features & target"

# ╔═╡ 202d524a-256b-4f31-8184-d9d977b609bf
target, features = unpack(df, ==(:Purchased))

# ╔═╡ c165decb-3264-436b-b06b-4d4a98e06ba4
md"Scatter plot"

# ╔═╡ 7ac707b1-7fb3-4dbd-89c7-e42b9d008421
scatter(features.Age, features.EstimatedSalary; group=target)

# ╔═╡ 94f8f2ed-c5e7-45e2-a826-0f00f7cd114b
md"Split the data into train & test sets"

# ╔═╡ d1dccb20-0533-4a91-ae0d-fd2e9b36f7bf
train, test = partition(eachindex(target), 0.8, rng=123)

# ╔═╡ ffd8ac0c-a336-4f06-95c5-a05acc62c7c3
Xtrain, Xtest = features[train, :], features[test, :]

# ╔═╡ 95e3507e-c3d0-4b69-9cde-c2f06c76d8ce
ytrain, ytest = target[train], target[test]

# ╔═╡ 98338274-9fb0-47c0-973f-9dac0a95e61e
md"Standardizer"

# ╔═╡ 440b7a2b-4bf9-4018-a832-7f38c39ce132
sc_ = Standardizer()

# ╔═╡ 975d57c2-f4e6-48fc-86f9-b623a005cccc
md"Import SVC and bind it to SVM"

# ╔═╡ d0bf2dad-6a61-4f11-bff8-48b0279a6159
SVM = @load SVC pkg=LIBSVM

# ╔═╡ fb9eeb40-c274-4068-b2ef-0b30e5cceb3e
svm_ = SVM() # If you want to change the `kernel`, please install `LIBSVM` pkg.

# ╔═╡ 17319fa1-9a67-47dc-90ab-2ddda6bc5114
md"Fit a pipeline to the training data"

# ╔═╡ d9435df3-a95a-4d2f-be62-1e636411e9be
pipe_ = Pipeline(sc_, svm_)

# ╔═╡ 92fa9db5-bcae-453d-a177-3fe0d0fe6c6f
pipe = machine(pipe_, Xtrain, ytrain) |> fit!

# ╔═╡ b4b314b9-442e-439b-b828-4404fa461b81
md"Let's make some predictions"

# ╔═╡ 1da818c6-15b7-4291-8d39-97d46572ee00
ŷ = predict(pipe, Xtest)

# ╔═╡ fcd347e1-d09a-43f3-b6db-bd0aa9458988
md"Confusion Matrix"

# ╔═╡ 27793133-f3bd-40ce-bf27-120bbcdd9e5d
confusion_matrix(ŷ, ytest)

# ╔═╡ d33d5277-46eb-43f9-9994-6838d76ae51f
md"Evaluation Metrics"

# ╔═╡ 1ffaf18f-a975-46ea-a25c-cc3ef0111feb
accuracy(ŷ, ytest)

# ╔═╡ dc02cf8e-2c9e-47cb-9f76-00ca873063ef
specificity(ŷ, ytest) # specificity, true negative rate: TN/(TN+FP)

# ╔═╡ d059badb-76bb-46df-99f4-62096b334034
sensitivity(ŷ, ytest) # sensitivity, true positive rate: TP/(TP+FN)

# ╔═╡ 130bd140-3637-43d8-b155-5271e8a88fea
f1score(ŷ, ytest)

# ╔═╡ 41368512-6706-4313-91ec-268fa7315c26
md"We can estimate the performance of `pipe` through the `evaluate!` command."

# ╔═╡ 4039c2ff-ee9a-4398-bdd6-158954bd7068
evaluate!(pipe, operation=predict)

# ╔═╡ Cell order:
# ╠═89bb4654-93c8-4db8-8d24-108446b0c2f6
# ╠═efd07dc7-ca06-4f1c-8dd4-a5f8be367d69
# ╠═19ae1eb6-9a1a-4cc5-a315-64d43a49855f
# ╠═5ae532ad-2454-413f-b4de-0d4dbac809d3
# ╠═9a98ff5e-8e82-456f-85d8-28b312c07c8d
# ╠═316c62ec-deb1-48e0-a29b-fdec424e3851
# ╠═5bdc592b-4f17-461f-8792-991b49664d34
# ╠═e618de37-70f1-4da0-95e7-666970f8fe79
# ╠═9effc32b-7c9f-434b-97b0-1f5bae762cbb
# ╠═2a8c82a9-32f6-4b90-9d65-85393d0eaa30
# ╠═d74f8f38-147b-4e95-b397-cea1d8cb1f4c
# ╠═a25abea7-82b9-4681-b256-896a45d97eef
# ╠═b0714705-eef3-4064-9748-626793bcc881
# ╠═202d524a-256b-4f31-8184-d9d977b609bf
# ╠═c165decb-3264-436b-b06b-4d4a98e06ba4
# ╠═5e4f2880-e058-48f0-80e6-70317c801e14
# ╠═7ac707b1-7fb3-4dbd-89c7-e42b9d008421
# ╠═94f8f2ed-c5e7-45e2-a826-0f00f7cd114b
# ╠═d1dccb20-0533-4a91-ae0d-fd2e9b36f7bf
# ╠═ffd8ac0c-a336-4f06-95c5-a05acc62c7c3
# ╠═95e3507e-c3d0-4b69-9cde-c2f06c76d8ce
# ╠═98338274-9fb0-47c0-973f-9dac0a95e61e
# ╠═440b7a2b-4bf9-4018-a832-7f38c39ce132
# ╠═975d57c2-f4e6-48fc-86f9-b623a005cccc
# ╠═d0bf2dad-6a61-4f11-bff8-48b0279a6159
# ╠═fb9eeb40-c274-4068-b2ef-0b30e5cceb3e
# ╠═17319fa1-9a67-47dc-90ab-2ddda6bc5114
# ╠═d9435df3-a95a-4d2f-be62-1e636411e9be
# ╠═92fa9db5-bcae-453d-a177-3fe0d0fe6c6f
# ╠═b4b314b9-442e-439b-b828-4404fa461b81
# ╠═1da818c6-15b7-4291-8d39-97d46572ee00
# ╠═fcd347e1-d09a-43f3-b6db-bd0aa9458988
# ╠═27793133-f3bd-40ce-bf27-120bbcdd9e5d
# ╠═d33d5277-46eb-43f9-9994-6838d76ae51f
# ╠═1ffaf18f-a975-46ea-a25c-cc3ef0111feb
# ╠═dc02cf8e-2c9e-47cb-9f76-00ca873063ef
# ╠═d059badb-76bb-46df-99f4-62096b334034
# ╠═130bd140-3637-43d8-b155-5271e8a88fea
# ╠═41368512-6706-4313-91ec-268fa7315c26
# ╠═4039c2ff-ee9a-4398-bdd6-158954bd7068
