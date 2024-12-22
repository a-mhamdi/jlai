### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ cebea174-69cc-420f-95e4-da9577115d25
begin
	cd(@__DIR__)
	import Pkg; Pkg.activate("..")
end

# ╔═╡ 48ab2092-42c3-410f-b428-242eb6449c11
using CSV, DataFrames

# ╔═╡ 0a76bdbc-da06-43fb-a7da-7ac51b064eeb
using MLJ

# ╔═╡ 408d9f6e-efb4-41c7-b148-b80efe45600c
using Plots; theme(:dracula)

# ╔═╡ d5f343da-9f77-49f3-a2b9-ced4e9664c3b
using PlutoUI

# ╔═╡ 4df16c37-19d8-4cc8-8df5-9ea4b91e986a
md"# k-NEAREST NEIGHBORS"

# ╔═╡ 2d2ad47c-e8c9-4420-a527-fe74ed9cd99f
md"
```julia
versioninfo() # -> v\"1.11.2\"
```
"

# ╔═╡ 56680207-6d9f-499d-aa74-9793bff22153
md"Import librairies"

# ╔═╡ 56e132c4-4616-4bed-84e1-b3fe806db6c4
md"Read data from CSV file"

# ╔═╡ c08c470e-a0ee-4b7c-8707-583607d94b98
df = CSV.read("../../Datasets/Social_Network_Ads.csv", DataFrame)

# ╔═╡ bc6142ae-7080-4cd4-b0b0-866e028ccbb8
schema(df)

# ╔═╡ 2c95678f-16e5-415f-99f6-6be19e9271d1
coerce!(df, 
        :Age => Continuous,
        :EstimatedSalary => Continuous,
        :Purchased => OrderedFactor)

# ╔═╡ b4af69af-367c-42a4-809f-9c9fca1f4f6a
schema(df)

# ╔═╡ 3a118d0c-3e11-4399-9af7-97c1bd9787de
md"Unpack features & target"

# ╔═╡ 1070caa5-2781-4bfd-b948-8be52a467a35
target, features = unpack(df, ==(:Purchased))

# ╔═╡ 08b855d9-2c72-48bb-9019-291b7fa4bad0
md"Scatter plot"

# ╔═╡ 04191fcf-ba34-4956-8eeb-d65b7c8f4a83
scatter(features.Age, features.EstimatedSalary; group=target)

# ╔═╡ 4df16605-fd55-4111-90b8-46c9cacbc0d7
md"Split the data into train & test sets"

# ╔═╡ 4ebe3b71-bd64-4f69-82fc-28ad0353c529
train, test = partition(eachindex(target), 0.8, rng=123)

# ╔═╡ 1eb5ebc8-4181-4443-82ad-870efda9efea
Xtrain, Xtest = features[train, :], features[test, :]

# ╔═╡ 159184b2-e301-4808-ba01-a9c178496e1f
ytrain, ytest = target[train], target[test]

# ╔═╡ 2f767a38-c0d7-4bb4-9889-935b6687339e
md"Standardizer"

# ╔═╡ b6df771e-e692-46b6-a25b-ef048036052e
sc_ = Standardizer()

# ╔═╡ 07d39f5d-e4f8-4ec5-9575-e75e3fde9aac
md"Load a knn classifier, w/ # neighbors = 3 _(default)_"

# ╔═╡ 61c4bb91-44c5-4fff-96b9-963c246b47c0
@bind K Slider(1:1:10, default=3)

# ╔═╡ 437d656b-52df-4a23-a7c0-bf7c6efaaf8e
begin
	KNN = @load KNNClassifier pkg=NearestNeighborModels
	knn_ = KNN(K=K)
end

# ╔═╡ 08b160a9-d014-44b0-b231-a4e1f277f875
md"You may want to see [NearestNeighborModels.jl](https://github.com/JuliaAI/NearestNeighborModels.jl) and the unwrapped model type [`NearestNeighborModels.KNNClassifier`](@ref)."

# ╔═╡ d48aba21-f61a-43ab-86bc-0c3ac2da6a56
md"Fit a pipeline to the training data"

# ╔═╡ 08dc0c41-8d5c-4544-8558-c2474d4d7a4c
pipe_ = Pipeline(sc_, knn_)

# ╔═╡ 783fb35e-d211-4557-8d4f-75c48a9ff26f
pipe = machine(pipe_, Xtrain, ytrain) |> fit!

# ╔═╡ ea8099b8-a27e-42cf-8be9-c0dcd3877619
md"Let's make some predictions"

# ╔═╡ 914362a9-343e-4bcf-b914-a0d0ce835c36
ŷ = predict_mode(pipe, Xtest)

# ╔═╡ edc5ea59-e231-4456-bbfd-142e18dd2171
md"Confusion Matrix"

# ╔═╡ eaa4a203-8558-4dbb-943a-052be339518d
confusion_matrix(ŷ, ytest)

# ╔═╡ 6d78ba2e-98c1-4fd1-bb3c-d14a311ace46
md"Evaluation Metrics"

# ╔═╡ cdcf4dee-544a-45fd-a784-0d9260be628d
accuracy(ŷ, ytest)

# ╔═╡ 73318658-38ec-49a8-ba02-9d7b29e2e5b9
specificity(ŷ, ytest) # specificity, true negative rate: TN/(TN+FP)

# ╔═╡ 0e569b8c-681a-4d91-bed6-3c925ec22cf5
sensitivity(ŷ, ytest) # sensitivity, true positive rate: TP/(TP+FN)

# ╔═╡ 520e304b-75e8-44ac-8928-84daa9e361af
f1score(ŷ, ytest)

# ╔═╡ 86d91133-26ec-4272-9a1d-e6b1d2fe2b2c
md"We can estimate the performance of `pipe` through the `evaluate!` command."

# ╔═╡ cb828c5d-0111-44a1-8780-f5d73bea404a
evaluate!(pipe, operation=predict)

# ╔═╡ 9d207775-f9b0-4c8f-9315-27198ee04275
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
# ╠═4df16c37-19d8-4cc8-8df5-9ea4b91e986a
# ╠═2d2ad47c-e8c9-4420-a527-fe74ed9cd99f
# ╠═cebea174-69cc-420f-95e4-da9577115d25
# ╠═56680207-6d9f-499d-aa74-9793bff22153
# ╠═48ab2092-42c3-410f-b428-242eb6449c11
# ╠═0a76bdbc-da06-43fb-a7da-7ac51b064eeb
# ╠═56e132c4-4616-4bed-84e1-b3fe806db6c4
# ╠═c08c470e-a0ee-4b7c-8707-583607d94b98
# ╠═bc6142ae-7080-4cd4-b0b0-866e028ccbb8
# ╠═2c95678f-16e5-415f-99f6-6be19e9271d1
# ╠═b4af69af-367c-42a4-809f-9c9fca1f4f6a
# ╠═3a118d0c-3e11-4399-9af7-97c1bd9787de
# ╠═1070caa5-2781-4bfd-b948-8be52a467a35
# ╠═08b855d9-2c72-48bb-9019-291b7fa4bad0
# ╠═408d9f6e-efb4-41c7-b148-b80efe45600c
# ╠═04191fcf-ba34-4956-8eeb-d65b7c8f4a83
# ╠═4df16605-fd55-4111-90b8-46c9cacbc0d7
# ╠═4ebe3b71-bd64-4f69-82fc-28ad0353c529
# ╠═1eb5ebc8-4181-4443-82ad-870efda9efea
# ╠═159184b2-e301-4808-ba01-a9c178496e1f
# ╠═2f767a38-c0d7-4bb4-9889-935b6687339e
# ╠═b6df771e-e692-46b6-a25b-ef048036052e
# ╠═07d39f5d-e4f8-4ec5-9575-e75e3fde9aac
# ╠═d5f343da-9f77-49f3-a2b9-ced4e9664c3b
# ╠═61c4bb91-44c5-4fff-96b9-963c246b47c0
# ╠═437d656b-52df-4a23-a7c0-bf7c6efaaf8e
# ╠═08b160a9-d014-44b0-b231-a4e1f277f875
# ╠═d48aba21-f61a-43ab-86bc-0c3ac2da6a56
# ╠═08dc0c41-8d5c-4544-8558-c2474d4d7a4c
# ╠═783fb35e-d211-4557-8d4f-75c48a9ff26f
# ╠═ea8099b8-a27e-42cf-8be9-c0dcd3877619
# ╠═914362a9-343e-4bcf-b914-a0d0ce835c36
# ╠═edc5ea59-e231-4456-bbfd-142e18dd2171
# ╠═eaa4a203-8558-4dbb-943a-052be339518d
# ╠═6d78ba2e-98c1-4fd1-bb3c-d14a311ace46
# ╠═cdcf4dee-544a-45fd-a784-0d9260be628d
# ╠═73318658-38ec-49a8-ba02-9d7b29e2e5b9
# ╠═0e569b8c-681a-4d91-bed6-3c925ec22cf5
# ╠═520e304b-75e8-44ac-8928-84daa9e361af
# ╠═86d91133-26ec-4272-9a1d-e6b1d2fe2b2c
# ╠═cb828c5d-0111-44a1-8780-f5d73bea404a
# ╟─9d207775-f9b0-4c8f-9315-27198ee04275
