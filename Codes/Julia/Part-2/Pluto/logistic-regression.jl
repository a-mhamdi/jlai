### A Pluto.jl notebook ###
# v0.20.4

#> [frontmatter]
#> title = "Logistic Regression"
#> tags = ["julialang", "logistic-regression", "binary-classification"]
#> date = "2024-12-20"
#> 
#>     [[frontmatter.author]]
#>     name = "A. Mhamdi"
#>     url = "https://a-mhamdi.github.io/jlai/#"

using Markdown
using InteractiveUtils

# ╔═╡ 6e6d37c7-5f08-446a-a7e4-163b496bcfd4
begin
	cd(@__DIR__)
	import Pkg; Pkg.activate("..")
end

# ╔═╡ 7e8e0022-0faa-4efb-b1c3-1441c5945a0e
using CSV, DataFrames

# ╔═╡ b26bcce5-5b75-4789-92c1-666459f8edcb
using MLJ

# ╔═╡ b09f7411-867e-487a-95cd-9ca967c0799d
using Plots; theme(:dracula)

# ╔═╡ a037e28a-71ac-4556-bbe8-6dd888904aae
md"# LOGISTIC REGRESSION"

# ╔═╡ 0b7e3ad5-a41e-43ce-9588-c53d98428bb2
md"
```julia
versioninfo() # -> v\"1.11.2\"
```
"

# ╔═╡ 2f71b33c-e4c3-4d5b-b376-3055742ab879
md"Import librairies"

# ╔═╡ 6f4b7273-a2c0-4629-aa3b-9870a61f84ed
md"Read data from CSV file"

# ╔═╡ a09c4107-156d-4287-9028-e3b20a336177
df = CSV.read("../../Datasets/Social_Network_Ads.csv", DataFrame)

# ╔═╡ b6c05699-fce5-43a0-bb90-53807a3f1856
schema(df)

# ╔═╡ 66227d37-a587-496f-aef1-b1eb08fe524d
coerce!(df, 
        :Age => Continuous,
        :EstimatedSalary => Continuous,
        :Purchased => OrderedFactor)

# ╔═╡ 18d37a78-2ad2-4c10-966d-6b6248cdfc3c
schema(df)

# ╔═╡ e0f7fd3b-e779-450f-b7d9-f8c3f90ec2bf
md"Unpack features & target"

# ╔═╡ 3608be3e-0071-4ce1-bb7e-4e3a517fa464
target, features = unpack(df, ==(:Purchased))

# ╔═╡ 95342bc5-d184-47f0-abc3-cd6e782c4a0a
md"Scatter plot"

# ╔═╡ 882de1cb-7284-4445-9d74-202398cb32a4
scatter(features.Age, features.EstimatedSalary; group=target)

# ╔═╡ ce29653b-7244-4cfa-96e2-6f559fe710e4
md"Split the data into train & test sets"

# ╔═╡ 5831726d-ec7e-4991-9e1d-277e9a8ca812
train, test = partition(eachindex(target), 0.8, rng=123)

# ╔═╡ a407f18c-2d75-4c7c-af84-1fabf9e59920
Xtrain, Xtest = features[train, :], features[test, :]

# ╔═╡ d10275b4-c1db-434c-bd20-cdef0f486585
ytrain, ytest = target[train], target[test]

# ╔═╡ e1d66feb-7369-4bfb-932d-4df36631b97b
md"Standardizer"

# ╔═╡ 5d44b71c-a146-48db-bfd5-1b32c133afae
sc_ = Standardizer()

# ╔═╡ 95aa0a4a-cda5-475b-ac79-91e8e51e8daf
md"Load the `LogisticClassifier` & bind it to `lc_`"

# ╔═╡ 75e26359-fad7-4c25-ba73-5bf6965cb20f
LC = @load LogisticClassifier pkg=MLJLinearModels

# ╔═╡ 0f708d68-571d-4a48-8435-d8a270c6d2af
lc_ = LC()

# ╔═╡ 387b57bf-5001-4eaf-b25c-5af29a92cc18
md"You may want to see [MLJLinearModels.jl](https://github.com/JuliaAI/MLJLinearModels.jl) and the unwrapped model type [`MLJLinearModels.LogisticClassifier`](@ref)."

# ╔═╡ 3311cdf3-b393-41c5-9cfb-6c718d1416c5
md"Construct a pipeline and train it"

# ╔═╡ e491a024-57dd-4e99-bce9-0e9741bd82a1
pipe_ = Pipeline(sc_, lc_)

# ╔═╡ fe1fda3a-c85f-4eba-8efe-6324349f89e7
pipe = machine(pipe_, Xtrain, ytrain) |> fit!

# ╔═╡ ef2f57ef-b932-40e2-95a0-1cc7ee39538b
md"Predict the output for `Xtest`"

# ╔═╡ 65c7bab8-1e22-4e53-820e-4f3a93674d0e
ŷ = predict_mode(pipe, Xtest);

# ╔═╡ 5aecbabc-f756-4a9c-ad92-ded2fc995696
md"Compute the accuracy"

# ╔═╡ 6499e104-e94c-4adf-8d92-69065d1c28a1
acc = mean(ŷ .== ytest);

# ╔═╡ dac9c7e5-1110-4319-8ab6-697e7f18ae38
println("Accuracy is about $(round(100*acc))%")

# ╔═╡ a9c3ccb5-82f4-487b-b8c1-fddc9eb56711
md"Confusion Matrix"

# ╔═╡ 5bad70c6-15d1-4706-89c0-b6b57b9d49d3
confusion_matrix(ŷ, ytest)

# ╔═╡ 0360fd1d-44e1-41a3-9878-d1f4d45f4d30
md"Evaluation Metrics"

# ╔═╡ c62b3174-5e28-4c23-84ed-3320779c2fc1
accuracy(ŷ, ytest)

# ╔═╡ 60fea13f-3084-42c1-9464-b2d35aa2eb8b
specificity(ŷ, ytest) # specificity, true negative rate: TN/(TN+FP)

# ╔═╡ f377936c-7f69-4bd6-9f42-2f18bd4b5aa0
sensitivity(ŷ, ytest) # sensitivity, true positive rate: TP/(TP+FN)

# ╔═╡ d44e7db0-7b71-4710-a6ee-4aebdfaf2eab
f1score(ŷ, ytest)

# ╔═╡ 94bee8c9-e012-446b-a548-4199b21af287
md"We can estimate, more elegantly, the performance of `pipe` through the `evaluate!` command."

# ╔═╡ 16a470ab-9ad5-4a4b-a48c-dfcbf519a02a
evaluate!(pipe, operation=predict)

# ╔═╡ e5d8346c-6354-447d-99bf-b92a353a8a00
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
# ╠═a037e28a-71ac-4556-bbe8-6dd888904aae
# ╠═0b7e3ad5-a41e-43ce-9588-c53d98428bb2
# ╠═6e6d37c7-5f08-446a-a7e4-163b496bcfd4
# ╠═2f71b33c-e4c3-4d5b-b376-3055742ab879
# ╠═7e8e0022-0faa-4efb-b1c3-1441c5945a0e
# ╠═b26bcce5-5b75-4789-92c1-666459f8edcb
# ╠═6f4b7273-a2c0-4629-aa3b-9870a61f84ed
# ╠═a09c4107-156d-4287-9028-e3b20a336177
# ╠═b6c05699-fce5-43a0-bb90-53807a3f1856
# ╠═66227d37-a587-496f-aef1-b1eb08fe524d
# ╠═18d37a78-2ad2-4c10-966d-6b6248cdfc3c
# ╠═e0f7fd3b-e779-450f-b7d9-f8c3f90ec2bf
# ╠═3608be3e-0071-4ce1-bb7e-4e3a517fa464
# ╠═95342bc5-d184-47f0-abc3-cd6e782c4a0a
# ╠═b09f7411-867e-487a-95cd-9ca967c0799d
# ╠═882de1cb-7284-4445-9d74-202398cb32a4
# ╠═ce29653b-7244-4cfa-96e2-6f559fe710e4
# ╠═5831726d-ec7e-4991-9e1d-277e9a8ca812
# ╠═a407f18c-2d75-4c7c-af84-1fabf9e59920
# ╠═d10275b4-c1db-434c-bd20-cdef0f486585
# ╠═e1d66feb-7369-4bfb-932d-4df36631b97b
# ╠═5d44b71c-a146-48db-bfd5-1b32c133afae
# ╠═95aa0a4a-cda5-475b-ac79-91e8e51e8daf
# ╠═75e26359-fad7-4c25-ba73-5bf6965cb20f
# ╠═0f708d68-571d-4a48-8435-d8a270c6d2af
# ╠═387b57bf-5001-4eaf-b25c-5af29a92cc18
# ╠═3311cdf3-b393-41c5-9cfb-6c718d1416c5
# ╠═e491a024-57dd-4e99-bce9-0e9741bd82a1
# ╠═fe1fda3a-c85f-4eba-8efe-6324349f89e7
# ╠═ef2f57ef-b932-40e2-95a0-1cc7ee39538b
# ╠═65c7bab8-1e22-4e53-820e-4f3a93674d0e
# ╠═5aecbabc-f756-4a9c-ad92-ded2fc995696
# ╠═6499e104-e94c-4adf-8d92-69065d1c28a1
# ╠═dac9c7e5-1110-4319-8ab6-697e7f18ae38
# ╠═a9c3ccb5-82f4-487b-b8c1-fddc9eb56711
# ╠═5bad70c6-15d1-4706-89c0-b6b57b9d49d3
# ╠═0360fd1d-44e1-41a3-9878-d1f4d45f4d30
# ╠═c62b3174-5e28-4c23-84ed-3320779c2fc1
# ╠═60fea13f-3084-42c1-9464-b2d35aa2eb8b
# ╠═f377936c-7f69-4bd6-9f42-2f18bd4b5aa0
# ╠═d44e7db0-7b71-4710-a6ee-4aebdfaf2eab
# ╠═94bee8c9-e012-446b-a548-4199b21af287
# ╠═16a470ab-9ad5-4a4b-a48c-dfcbf519a02a
# ╟─e5d8346c-6354-447d-99bf-b92a353a8a00
