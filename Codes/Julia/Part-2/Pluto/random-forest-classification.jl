### A Pluto.jl notebook ###
# v0.20.5

using Markdown
using InteractiveUtils

# ╔═╡ 96f2e249-825f-4f12-8fa7-fe09f64a3f07
begin
	cd(@__DIR__)
	import Pkg; Pkg.activate(".."); Pkg.status()
end

# ╔═╡ 1b70bc95-7f38-4e46-87aa-dff90cfeec4d
using CSV, DataFrames, Plots

# ╔═╡ cdadeca0-4f30-4799-aadf-7443ad91f2f8
using MLJ

# ╔═╡ da4f6f6b-dc29-491c-a1b7-1fa0564924e6
md"# RANDOM FOREST CLASSIFICATION"

# ╔═╡ c58ada09-ce71-4843-a9b4-81d052a55868
versioninfo() # -> v\"1.11.1\"

# ╔═╡ dab20e11-4d11-4706-923c-097a2bb6c59f
md"Import librairies"

# ╔═╡ d134109f-6cb6-4928-b2d4-82004693358b
md"Read dataset -> `df`"

# ╔═╡ 0739cf94-b052-4c18-a4b7-dc3eebb97492
df = CSV.read("../../Datasets/Social_Network_Ads.csv", DataFrame)

# ╔═╡ 31bbfe7c-df8c-4fc7-97cf-fe4c4deaa2cf
md"Unpack data"

# ╔═╡ 1743730d-2d15-438e-ba4f-f64dc6019f72
features, target = unpack(df,
                          ==(:EstimatedSalary),
                          ==(:Purchased);
                          :EstimatedSalary => Continuous,
                          :Purchased => Multiclass)

# ╔═╡ cfbe8a30-6bb7-48c1-bdf5-65069a8ff493
md"Scatter plot"

# ╔═╡ 424b70d8-c643-4c6a-971a-5cacc6217158
scatter(features, target; group=target, legend=false)

# ╔═╡ 50275a9f-56c2-46cc-b5d9-502317b2571a
md"Convert data to tabular format"

# ╔═╡ b2989780-3c7e-43d3-953a-c1ae2cfc5b56
x = Tables.table(features);

# ╔═╡ 593e87ba-7ac3-4056-9a76-8f8467951e97
y = target;

# ╔═╡ 4aa488da-4ce5-401f-8c10-39984971c632
md"Bind an instance `rfc_` model to training data"

# ╔═╡ 0433480f-1ba7-43c0-92d9-2d9c5bc11728
RFC = @load RandomForestClassifier pkg=DecisionTree

# ╔═╡ 06126d90-efc8-4fba-986b-6fa4a8201c53
rfc_ = RFC(max_depth=5, min_samples_split=3)

# ╔═╡ 53ea62b6-161e-4908-9370-40c688fa5e37
rfc = machine(rfc_, x, y) |> fit!

# ╔═╡ a6009d17-ae7c-4cba-926f-d4d6fc1bccd6
md"You may want to see [DecisionTree.jl](https://github.com/bensadeghi/DecisionTree.jl) and the unwrapped model type [`MLJDecisionTreeInterface.DecisionTree.RandomForestClassifier`](@ref)."

# ╔═╡ 130d535e-0cd2-409b-878d-36e293d1399e
md"Evaluate the model"

# ╔═╡ a1626d6e-a368-469f-a5c6-fbaf05876caa
evaluate!(rfc)

# ╔═╡ Cell order:
# ╠═da4f6f6b-dc29-491c-a1b7-1fa0564924e6
# ╠═c58ada09-ce71-4843-a9b4-81d052a55868
# ╠═96f2e249-825f-4f12-8fa7-fe09f64a3f07
# ╠═dab20e11-4d11-4706-923c-097a2bb6c59f
# ╠═1b70bc95-7f38-4e46-87aa-dff90cfeec4d
# ╠═cdadeca0-4f30-4799-aadf-7443ad91f2f8
# ╠═d134109f-6cb6-4928-b2d4-82004693358b
# ╠═0739cf94-b052-4c18-a4b7-dc3eebb97492
# ╠═31bbfe7c-df8c-4fc7-97cf-fe4c4deaa2cf
# ╠═1743730d-2d15-438e-ba4f-f64dc6019f72
# ╠═cfbe8a30-6bb7-48c1-bdf5-65069a8ff493
# ╠═424b70d8-c643-4c6a-971a-5cacc6217158
# ╠═50275a9f-56c2-46cc-b5d9-502317b2571a
# ╠═b2989780-3c7e-43d3-953a-c1ae2cfc5b56
# ╠═593e87ba-7ac3-4056-9a76-8f8467951e97
# ╠═4aa488da-4ce5-401f-8c10-39984971c632
# ╠═0433480f-1ba7-43c0-92d9-2d9c5bc11728
# ╠═06126d90-efc8-4fba-986b-6fa4a8201c53
# ╠═53ea62b6-161e-4908-9370-40c688fa5e37
# ╠═a6009d17-ae7c-4cba-926f-d4d6fc1bccd6
# ╠═130d535e-0cd2-409b-878d-36e293d1399e
# ╠═a1626d6e-a368-469f-a5c6-fbaf05876caa
