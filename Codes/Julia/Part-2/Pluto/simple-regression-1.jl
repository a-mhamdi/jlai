### A Pluto.jl notebook ###
# v0.20.4

#> [frontmatter]
#> title = "Simple Linear Regression"
#> date = "2024-12-20"
#> 
#>     [[frontmatter.author]]
#>     name = "A. Mhamdi"
#>     url = "https://a-mhamdi.github.io/jlai/#"

using Markdown
using InteractiveUtils

# ╔═╡ b3e9d378-0277-4219-b8e6-ff9caef39725
begin
	cd(@__DIR__)
	import Pkg; Pkg.activate("..")
end

# ╔═╡ f9c34383-95c7-4e4b-a5e6-aaf661197b30
using CSV, DataFrames

# ╔═╡ f406ddf3-1a92-45e5-b471-81b837e30001
using MLJ

# ╔═╡ b62abbfe-b623-4ba3-b2be-758699f20ad8
using Plots; theme(:dracula)

# ╔═╡ 7d95bf75-ebf9-49a8-813c-17bf24ad2c1d
md"# SIMPLE LINEAR REGRESSION _(SALARY vs. YEARS of EXPERIENCE)_"

# ╔═╡ 97c1a7c2-fb03-4463-aa4e-a16bccc94597
md"
```julia
versioninfo() # -> v\"1.11.2\"
```
"

# ╔═╡ a10b0619-4da9-4a2f-b75e-002baa5d1a92
md"Import librairies"

# ╔═╡ 5a418a7e-5779-4697-93db-c22c2b60ecfe
md"Read data using `.csv` file. Convert it to `DataFrame` object"

# ╔═╡ c30699f3-9baa-4ae8-8273-3e2654186732
df = CSV.read("../../Datasets/Salary_Data.csv", DataFrame)

# ╔═╡ f2ce0f72-c4b4-4eca-b866-c41e1e8244fa
md"Unpacking features & target"

# ╔═╡ 2932f0ed-7f29-4f31-bfb9-ac818a3a00dd
x = df.YearsExperience

# ╔═╡ 4740408b-c8f6-48ce-a304-1586a8212fe8
y = df.Salary

# ╔═╡ 13a6aa1c-5ee6-44ae-b118-996383aac0b3
md"Scatter Plot of `Salary` vs. `YearsExperience`"

# ╔═╡ f7703119-ec4f-4ae6-b744-a6cf69436423
scatter(x, y, label=:none, title="Salary vs. YearsExperience")

# ╔═╡ c8c30c1a-204c-47bd-8655-51c87b48d035
md"Preparing the split"

# ╔═╡ dad9ec09-cdbb-457e-a171-4b77747d591d
train, test = partition(eachindex(y), 0.8, shuffle=true, rng=123)

# ╔═╡ 2e4a0a65-086c-44fd-a50e-6ac2b83d35eb
xtrain, xtest = x[train], x[test]

# ╔═╡ 566e3a7f-71ed-4da8-b02f-afdf47aefeed
ytrain, ytest = y[train], y[test]

# ╔═╡ 569eefd5-4768-4c70-a3a3-7d40ff409846
md"Load & instantiate the linear regression object"

# ╔═╡ 3c22b67f-c460-475e-8446-851046f013d1
LR = @load LinearRegressor pkg=MLJLinearModels

# ╔═╡ 57f5fcb9-a023-4538-9696-e88da2ad7744
lr_ = LR()

# ╔═╡ 4bc66bbc-9517-411c-a47e-7724a329776c
md"You may want to see [MLJLinearModels.jl](https://github.com/JuliaAI/MLJLinearModels.jl) and the unwrapped model type [`MLJLinearModels.LinearRegressor`](@ref)."

# ╔═╡ 54c4944f-1044-4f66-a829-e853510e4820
md"Train & fit"

# ╔═╡ da390157-cbad-4441-8b0d-91f734c0cba3
lr = machine(lr_, Tables.table(xtrain), ytrain) |> fit!

# ╔═╡ 80587eb9-dcbc-4232-aa4b-f00de68c6289
md"Fitted parameters"

# ╔═╡ db13e2db-4354-48db-a4b3-7b78cf55ecee
fitted_params(lr)

# ╔═╡ fce60864-4e27-499e-ad02-c6017c1f5ab0
md"Prediction"

# ╔═╡ 1b009cad-21bc-490d-998d-19fdb665398b
yhat = predict(lr, Tables.table(xtest))

# ╔═╡ 29bdd18a-68d4-4e55-89af-d03400bb36aa
md"Error measurement"

# ╔═╡ 292c1ce0-662d-4979-8b7a-123b5ec06bbc
println("Error is $(sum( (yhat .- ytest).^2 ) ./ length(ytest) )")

# ╔═╡ d91aa4dd-0cdf-4bbc-978c-c5e79cfa9b1d
begin
	scatter(xtest, ytest, label=:none)
	scatter!(xtest, yhat, label=:none)
end

# ╔═╡ 3af7120e-86ff-4cd8-879e-bfd6390c0302
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
# ╠═7d95bf75-ebf9-49a8-813c-17bf24ad2c1d
# ╠═97c1a7c2-fb03-4463-aa4e-a16bccc94597
# ╠═b3e9d378-0277-4219-b8e6-ff9caef39725
# ╠═a10b0619-4da9-4a2f-b75e-002baa5d1a92
# ╠═f9c34383-95c7-4e4b-a5e6-aaf661197b30
# ╠═f406ddf3-1a92-45e5-b471-81b837e30001
# ╠═5a418a7e-5779-4697-93db-c22c2b60ecfe
# ╠═c30699f3-9baa-4ae8-8273-3e2654186732
# ╠═f2ce0f72-c4b4-4eca-b866-c41e1e8244fa
# ╠═2932f0ed-7f29-4f31-bfb9-ac818a3a00dd
# ╠═4740408b-c8f6-48ce-a304-1586a8212fe8
# ╠═13a6aa1c-5ee6-44ae-b118-996383aac0b3
# ╠═b62abbfe-b623-4ba3-b2be-758699f20ad8
# ╠═f7703119-ec4f-4ae6-b744-a6cf69436423
# ╠═c8c30c1a-204c-47bd-8655-51c87b48d035
# ╠═dad9ec09-cdbb-457e-a171-4b77747d591d
# ╠═2e4a0a65-086c-44fd-a50e-6ac2b83d35eb
# ╠═566e3a7f-71ed-4da8-b02f-afdf47aefeed
# ╠═569eefd5-4768-4c70-a3a3-7d40ff409846
# ╠═3c22b67f-c460-475e-8446-851046f013d1
# ╠═57f5fcb9-a023-4538-9696-e88da2ad7744
# ╠═4bc66bbc-9517-411c-a47e-7724a329776c
# ╠═54c4944f-1044-4f66-a829-e853510e4820
# ╠═da390157-cbad-4441-8b0d-91f734c0cba3
# ╠═80587eb9-dcbc-4232-aa4b-f00de68c6289
# ╠═db13e2db-4354-48db-a4b3-7b78cf55ecee
# ╠═fce60864-4e27-499e-ad02-c6017c1f5ab0
# ╠═1b009cad-21bc-490d-998d-19fdb665398b
# ╠═29bdd18a-68d4-4e55-89af-d03400bb36aa
# ╠═292c1ce0-662d-4979-8b7a-123b5ec06bbc
# ╠═d91aa4dd-0cdf-4bbc-978c-c5e79cfa9b1d
# ╟─3af7120e-86ff-4cd8-879e-bfd6390c0302
