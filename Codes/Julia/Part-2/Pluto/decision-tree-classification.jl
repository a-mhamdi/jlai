### A Pluto.jl notebook ###
# v0.20.5

using Markdown
using InteractiveUtils

# ╔═╡ afc0157b-85c6-4282-9900-e88ab9648eca
begin
	cd(@__DIR__)
	import Pkg; Pkg.activate(".."); Pkg.status()
end

# ╔═╡ 94957b8c-2270-4ff8-a1dc-af8837cb7c6b
using CSV, DataFrames, Plots

# ╔═╡ cbeff923-835b-49fd-902a-13dff68177c4
using MLJ

# ╔═╡ 9d8b7fbb-251c-42d1-80ba-83dfb2d64498
md"# DECISION TREE CLASSIFICATION"

# ╔═╡ 29908274-2a01-4863-a1b8-2cb8e3265b5c
versioninfo()

# ╔═╡ 20f37712-ab0e-4724-be4a-44234c6f9605
md"Import librairies"

# ╔═╡ 33f85db7-1947-4d20-a71d-d91ecd80a062
md"Read dataset and assign it `df`"

# ╔═╡ b1e2b307-e015-42f0-91a3-a9341a0dc208
df = CSV.read("../../Datasets/Social_Network_Ads.csv", DataFrame)

# ╔═╡ d582d362-597d-49f3-ad19-17715963c428
md"Unpack data"

# ╔═╡ 2271e38e-7e8f-4d4f-b900-58f981727bc1
features, target = unpack(df,
                          ==(:EstimatedSalary),
                          ==(:Purchased);
                          :EstimatedSalary => Continuous,
                          :Purchased => Multiclass)

# ╔═╡ b6632bbd-3fa2-47d7-b2ec-0694fe06f584
md"Scatter plot"

# ╔═╡ 95e8d49b-6be5-470e-8048-1c83bd18e3f7
scatter(features, target; group=target, legend=false)

# ╔═╡ a9f9ba26-1507-45c9-9767-522feecbcd0a
md"Convert data"

# ╔═╡ 3566eaaf-b2c7-4086-bbcd-25b09bc44e7f
x = Tables.table(features);

# ╔═╡ db3a317c-d68a-421e-a08b-968aadf6e320
y = target;

# ╔═╡ 5b60497e-c726-4bd4-a058-c039348291e6
md"Bind an instance `dtc_` model to training data"

# ╔═╡ e655e73f-9090-4f2f-b8e7-f796f78c3244
begin
	DTC = @load DecisionTreeClassifier pkg=DecisionTree
	dtc_ = DTC(max_depth=5, min_samples_split=3)
	dtc = machine(dtc_, x, y) |> fit!
end

# ╔═╡ 5184c169-2730-4c07-928e-0fba7002bc7e
md"You may want to see [DecisionTree.jl](https://github.com/bensadeghi/DecisionTree.jl) and the unwrapped model type [`MLJDecisionTreeInterface.DecisionTree.DecisionTreeClassifier`](@ref)."

# ╔═╡ f02d2e91-2da7-4369-ab42-91e2164117a5
md"Evaluate model"

# ╔═╡ 7b571619-4ef9-4c3f-8d5b-ff7d38ed3222
evaluate!(dtc)

# ╔═╡ Cell order:
# ╠═9d8b7fbb-251c-42d1-80ba-83dfb2d64498
# ╠═29908274-2a01-4863-a1b8-2cb8e3265b5c
# ╠═afc0157b-85c6-4282-9900-e88ab9648eca
# ╠═20f37712-ab0e-4724-be4a-44234c6f9605
# ╠═94957b8c-2270-4ff8-a1dc-af8837cb7c6b
# ╠═cbeff923-835b-49fd-902a-13dff68177c4
# ╠═33f85db7-1947-4d20-a71d-d91ecd80a062
# ╠═b1e2b307-e015-42f0-91a3-a9341a0dc208
# ╠═d582d362-597d-49f3-ad19-17715963c428
# ╠═2271e38e-7e8f-4d4f-b900-58f981727bc1
# ╠═b6632bbd-3fa2-47d7-b2ec-0694fe06f584
# ╠═95e8d49b-6be5-470e-8048-1c83bd18e3f7
# ╠═a9f9ba26-1507-45c9-9767-522feecbcd0a
# ╠═3566eaaf-b2c7-4086-bbcd-25b09bc44e7f
# ╠═db3a317c-d68a-421e-a08b-968aadf6e320
# ╠═5b60497e-c726-4bd4-a058-c039348291e6
# ╠═e655e73f-9090-4f2f-b8e7-f796f78c3244
# ╠═5184c169-2730-4c07-928e-0fba7002bc7e
# ╠═f02d2e91-2da7-4369-ab42-91e2164117a5
# ╠═7b571619-4ef9-4c3f-8d5b-ff7d38ed3222
