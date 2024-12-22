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

# ╔═╡ c0445977-65af-4368-9e64-91ee2db27a53
begin
	cd(@__DIR__)
	import Pkg; Pkg.activate("..")
end

# ╔═╡ dfd07be0-e12e-41ed-837a-ff2b58a2db01
using CSV, DataFrames

# ╔═╡ ff150ddf-c94a-4766-a64e-6d2cab463107
using Plots; theme(:dracula)

# ╔═╡ 779a23b0-6bdf-4d31-95c5-1029e584645c
using MLJ

# ╔═╡ 5abfca26-fb09-4a46-bd6c-9741aaaf115d
using PlutoUI

# ╔═╡ 838a220d-c8f5-42aa-a3f4-a33ffe00900d
md"# KMEANS"

# ╔═╡ 89420f6d-b069-4b06-8f0f-cca5a00accad
md"
```julia
versioninfo() # -> v\"1.11.2\"
```
"

# ╔═╡ 6861c462-8aac-4a60-943f-99f07d964bd8
md"It is a clustering algorithm that is used to partition an unlabeled dataset into a specified number of clusters."

# ╔═╡ 759d2896-b974-4b78-b541-3053186e01bc
md"Import librairies"

# ╔═╡ 3e32277b-3235-4f3d-865a-908dc634c999
md"Load data from CSV file"

# ╔═╡ c078e499-cea3-4943-be11-0f6f1d7f9107
df = CSV.read("../../Datasets/Mall_Customers.csv", DataFrame);

# ╔═╡ 1cef0274-f768-47cf-90d3-873fa7b99288
schema(df)

# ╔═╡ b670323d-26ed-4282-8346-124593176430
first(df, 5)

# ╔═╡ cbd57995-7c1d-44eb-abed-31d42178e028
md"Features"

# ╔═╡ 294d7dab-eb60-4d8b-985d-f7738f2dc64e
income, ss = df[!, 4], df[!, 5];

# ╔═╡ 72e864c9-16a3-4734-895d-9e54a46670b6
X = hcat(ss, income);

# ╔═╡ 30e994f0-9baa-4a1e-84da-bfa0511a9faf
typeof(X)

# ╔═╡ f0c4a3db-db4f-4db9-861c-ed7e51a720cd
md"Take a look @ data"

# ╔═╡ d15fd27c-2812-431b-8268-552d8dec1c60
scatter(income, ss, legend=false)

# ╔═╡ 76c2d64a-8db5-4064-8a4b-24ec7d2f7263
md"Load & instantiate `KMeans` object"

# ╔═╡ 3adfbdd2-da2b-4159-8966-6aadc5bbe94b
@bind k Slider(1:1:10, default=5)

# ╔═╡ 19577401-6308-41de-bfdc-2080e52f479c
begin
	KMeans = @load KMeans pkg=Clustering
	kmeans_ = KMeans(k=k)
end

# ╔═╡ 4e57671b-c945-4fcb-ae32-d11281c2f68d
md"You may want to see [Clustering.jl](https://github.com/JuliaStats/Clustering.jl) and the unwrapped model type [`Clustering.KMeans`](@ref)."

# ╔═╡ e2d4d1c2-41aa-4a62-9563-f1626a012d78
md"Train & regroup into clusters"

# ╔═╡ 6c868647-0031-458f-99f8-693610082d4d
kmeans = machine(kmeans_, X) |> fit!

# ╔═╡ 2645cd19-c10d-4854-a9a6-44648409cc0e
md"Clusters & centroids"

# ╔═╡ 0a6aaea7-a736-429a-90f9-6488394516d5
centroids = fitted_params(kmeans).centers

# ╔═╡ 39023ee8-3b13-46db-a0f9-7f3a252e3191
md"Extract clusters values"

# ╔═╡ 15d78ebd-7d96-4078-b47b-ad60c938e31c
y = report(kmeans).assignments

# ╔═╡ 4ace1040-67c4-4d22-b0fb-3626d672f9aa
md"Scatter plots"

# ╔═╡ 33420260-2033-4e31-a804-bfe4a55be9f4
scatter(ss, income, marker_z=y, color=:winter, legend=false)

# ╔═╡ 63df9627-b143-4778-8c78-adc048cdfb91
scatter!(centroids[1,:], centroids[2,:], color=:red, labels=['1', '2', '3', '4', '5'])

# ╔═╡ 47d02dac-fb2c-4b4f-a683-0f5ff33ac8d1
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
# ╠═838a220d-c8f5-42aa-a3f4-a33ffe00900d
# ╠═89420f6d-b069-4b06-8f0f-cca5a00accad
# ╠═6861c462-8aac-4a60-943f-99f07d964bd8
# ╠═c0445977-65af-4368-9e64-91ee2db27a53
# ╠═759d2896-b974-4b78-b541-3053186e01bc
# ╠═dfd07be0-e12e-41ed-837a-ff2b58a2db01
# ╠═ff150ddf-c94a-4766-a64e-6d2cab463107
# ╠═779a23b0-6bdf-4d31-95c5-1029e584645c
# ╠═3e32277b-3235-4f3d-865a-908dc634c999
# ╠═c078e499-cea3-4943-be11-0f6f1d7f9107
# ╠═1cef0274-f768-47cf-90d3-873fa7b99288
# ╠═b670323d-26ed-4282-8346-124593176430
# ╠═cbd57995-7c1d-44eb-abed-31d42178e028
# ╠═294d7dab-eb60-4d8b-985d-f7738f2dc64e
# ╠═72e864c9-16a3-4734-895d-9e54a46670b6
# ╠═30e994f0-9baa-4a1e-84da-bfa0511a9faf
# ╠═f0c4a3db-db4f-4db9-861c-ed7e51a720cd
# ╠═d15fd27c-2812-431b-8268-552d8dec1c60
# ╠═76c2d64a-8db5-4064-8a4b-24ec7d2f7263
# ╠═5abfca26-fb09-4a46-bd6c-9741aaaf115d
# ╠═3adfbdd2-da2b-4159-8966-6aadc5bbe94b
# ╠═19577401-6308-41de-bfdc-2080e52f479c
# ╠═4e57671b-c945-4fcb-ae32-d11281c2f68d
# ╠═e2d4d1c2-41aa-4a62-9563-f1626a012d78
# ╠═6c868647-0031-458f-99f8-693610082d4d
# ╠═2645cd19-c10d-4854-a9a6-44648409cc0e
# ╠═0a6aaea7-a736-429a-90f9-6488394516d5
# ╠═39023ee8-3b13-46db-a0f9-7f3a252e3191
# ╠═15d78ebd-7d96-4078-b47b-ad60c938e31c
# ╠═4ace1040-67c4-4d22-b0fb-3626d672f9aa
# ╠═33420260-2033-4e31-a804-bfe4a55be9f4
# ╠═63df9627-b143-4778-8c78-adc048cdfb91
# ╟─47d02dac-fb2c-4b4f-a683-0f5ff33ac8d1
