### A Pluto.jl notebook ###
# v0.20.5

using Markdown
using InteractiveUtils

# ╔═╡ ec000cc5-f59b-457b-8591-551abed91d45
begin
	cd(@__DIR__)
	import Pkg; Pkg.activate(".."); Pkg.status()
end

# ╔═╡ 8ea138ce-43df-41c2-8b95-f7e48c6b0aa6
using DataFrames, MLJ

# ╔═╡ 00b6a6ed-072f-4757-8c0a-30f55488d058
using Plots

# ╔═╡ f79c06d0-1335-11f0-3cba-bf258b014084
md"# DBSCAN"

# ╔═╡ 8a3b2601-765b-453e-86cf-61ae2f547f5f
versioninfo()

# ╔═╡ 0dd10708-0582-4230-8929-bfa5013f2543
md"Generate 100 synthetic data points close to two concentric circles"

# ╔═╡ b57038fa-b2bf-4b01-bdff-ade8b08100c5
X, y = make_circles(100; noise=0.05, factor=0.3)

# ╔═╡ bc10fca9-15a2-42fd-85f0-9e7219621ced
scatter(X.x1, X.x2, group=y)

# ╔═╡ 57258b17-aec6-4e97-abc0-6f5c551932b4
DBSCAN = @load DBSCAN pkg=Clustering

# ╔═╡ 764d211d-9b57-478c-a71b-69cf496d65ce
dbscan = DBSCAN(radius=0.1, min_cluster_size=10)

# ╔═╡ 338dc37c-6b07-4fc6-9de0-24f82e4938a9
mach_dbscan = machine(dbscan)

# ╔═╡ f0362c4c-5ed4-4af5-bc4f-7e30c287afd0
md"Compute and output cluster assignments for observations in `X`"

# ╔═╡ f025f1b5-11c2-4b0d-a69c-3bb9362f46d2
ŷ = predict(mach_dbscan, X)

# ╔═╡ 5acdba2c-64f2-409c-9698-8af1473e4d73
md"Get DBSCAN point types"

# ╔═╡ e14b9f49-348a-4bfe-8449-412d3142e74d
report(mach_dbscan).point_types

# ╔═╡ e81b46dc-8796-46ba-9648-fb529c858d04
report(mach_dbscan).nclusters

# ╔═╡ 12261b66-2541-4892-96f9-f8c70adbcea9
md"Compare cluster labels with actual labels"

# ╔═╡ 1ba14446-9e03-4b5e-b7b3-e1d2be0c9ba0
compare = zip(ŷ, y) |> collect;

# ╔═╡ 1023d391-0560-4081-ae29-755512165653
compare[1:10]

# ╔═╡ a30e5b00-ea53-43a0-a647-df779d08e0b8
md"Visualize clusters, noise in red"

# ╔═╡ e53398e4-f06a-4c7c-b340-5beebe0cde48
colors = map(ŷ) do label
	label == 0 ? :red : 
	label == 1 ? :blue :
	label == 2 ? :green :
	:black
end

# ╔═╡ c1042632-3acc-49a1-a265-984a7c32663c
scatter(X.x1, X.x2, color=colors)

# ╔═╡ ce37cefb-c7c0-4c29-892f-a7ca44db94a6
scatter(X.x1, X.x2, group=ŷ)

# ╔═╡ Cell order:
# ╠═f79c06d0-1335-11f0-3cba-bf258b014084
# ╠═8a3b2601-765b-453e-86cf-61ae2f547f5f
# ╠═ec000cc5-f59b-457b-8591-551abed91d45
# ╠═8ea138ce-43df-41c2-8b95-f7e48c6b0aa6
# ╠═0dd10708-0582-4230-8929-bfa5013f2543
# ╠═b57038fa-b2bf-4b01-bdff-ade8b08100c5
# ╠═00b6a6ed-072f-4757-8c0a-30f55488d058
# ╠═bc10fca9-15a2-42fd-85f0-9e7219621ced
# ╠═57258b17-aec6-4e97-abc0-6f5c551932b4
# ╠═764d211d-9b57-478c-a71b-69cf496d65ce
# ╠═338dc37c-6b07-4fc6-9de0-24f82e4938a9
# ╠═f0362c4c-5ed4-4af5-bc4f-7e30c287afd0
# ╠═f025f1b5-11c2-4b0d-a69c-3bb9362f46d2
# ╠═5acdba2c-64f2-409c-9698-8af1473e4d73
# ╠═e14b9f49-348a-4bfe-8449-412d3142e74d
# ╠═e81b46dc-8796-46ba-9648-fb529c858d04
# ╠═12261b66-2541-4892-96f9-f8c70adbcea9
# ╠═1ba14446-9e03-4b5e-b7b3-e1d2be0c9ba0
# ╠═1023d391-0560-4081-ae29-755512165653
# ╠═a30e5b00-ea53-43a0-a647-df779d08e0b8
# ╠═e53398e4-f06a-4c7c-b340-5beebe0cde48
# ╠═c1042632-3acc-49a1-a265-984a7c32663c
# ╠═ce37cefb-c7c0-4c29-892f-a7ca44db94a6
