### A Pluto.jl notebook ###
# v0.20.4

#> [frontmatter]
#> title = "ML Data Preprocessing Workflows"
#> date = "2024-12-21"
#> tags = ["ml", "julialang", "standardization", "data-split"]
#> 
#>     [[frontmatter.author]]
#>     name = "A. Mhamdi"
#>     url = "https://a-mhamdi.github.io/jlai/"

using Markdown
using InteractiveUtils

# ╔═╡ e75026f1-f5e4-48a1-b9c4-d65a19dfdd9d
begin
	cd(@__DIR__)
	import Pkg; Pkg.activate("..")
end

# ╔═╡ 60c276fa-3605-4cc0-9805-ca331b5892a5
using CSV, DataFrames

# ╔═╡ f8a8da51-ae01-476a-865e-3acd40038966
using MLJ

# ╔═╡ aa831e6c-66db-4884-90a4-3184bd540781
md"# COMMON DATA PREPROCESSING `WORKFLOWS`"

# ╔═╡ 4ff29318-5d25-4909-adc6-3a2a8bc578b8
md"
```julia
versioninfo() # -> v\"1.11.2\"
```
"

# ╔═╡ 6493ef22-17b1-4851-b537-b34496e3bc69
md"Import librairies"

# ╔═╡ d4eaf94b-1893-4b65-9c24-8a102dd7b0cb
md"Import data from CSV file"

# ╔═╡ 7a44f465-c922-4eb8-a8fb-b42f39c5571c
df = CSV.read("../../Datasets/Data.csv", DataFrame)

# ╔═╡ 69eafcf8-0a9b-4eee-9416-1654686140a5
describe(df)

# ╔═╡ 28eb8e6f-67ee-4bb0-9b7c-8aa9673e4f9b
nrow(df), ncol(df)

# ╔═╡ 80120c69-e14d-4b10-b71f-e61eeaba86ef
schema(df)

# ╔═╡ 246738aa-83cd-44ac-8e77-59c9c91661d2
md"Scientific type coercion"

# ╔═╡ de4130b1-5730-44e1-813f-8c3539b700ea
df_coerced = coerce(df,
    :Country => Multiclass,
    :Age => Continuous,
    :Salary => Continuous,
    :Purchased => Multiclass);

# ╔═╡ 3d350b68-6de4-4d64-9040-7e24ff6c831d
schema(df_coerced)

# ╔═╡ e4795cf3-e5be-403d-a899-0609f862df05
md"Missing values imputation"

# ╔═╡ f2e650cf-98af-4038-9d30-53103d279197
imputer = FillImputer()

# ╔═╡ e4038ff0-8841-407d-b73a-a569d838b15c
mach = machine(imputer, df_coerced) |> fit!

# ╔═╡ fb5b6cae-46c1-4806-95d6-6feec6ea05ec
df_imputed = MLJ.transform(mach, df_coerced);

# ╔═╡ 5e449816-8278-4acd-b1bc-ba56385618f5
schema(df_imputed)

#= CAN BE WRITTEN THIS WAY
df_imputed = machine(imputer, df_coerced) |> fit! |> MLJ.transform
=#

# ╔═╡ 466b7405-6e69-4244-a5d3-bf33f02bcf28
md"Features & target selection"

# ╔═╡ d1ac47ef-1372-4937-8912-8b00e892b66e
X_imputed = select(df_imputed,
    :Country, # :Country__France, :Country__Germany, :Country__Spain, # levels(df.Country)
    :Age,
    :Salary)

# ╔═╡ 5abb2589-fb45-4ade-966d-44a77b9d2ee4
y_imputed = select(df_imputed, :Purchased)

# ╔═╡ ff5abaf3-533f-4249-bc1c-7233e1bdcdcf
md"Feature encoding"

# ╔═╡ efa2bbf1-3b2a-467a-a446-8697a417404a
encoder_X = ContinuousEncoder()

# ╔═╡ af4eed5f-0476-47d4-9a93-498b90e5a2f5
encoder_y = ContinuousEncoder(drop_last=true)

#=
mach_X = machine(encoder_X, X_imputed) |> fit!
mach_y = machine(encoder_y, y_imputed) |> fit!
X = MLJ.transform(mach_X, X_imputed);
y = MLJ.transform(mach_y, y_imputed);
=#

# ╔═╡ 4af41c48-caa5-4261-b1cd-8c0a05d4acad
X = machine(encoder_X, X_imputed) |> fit! |> MLJ.transform

# ╔═╡ d0f8e9f9-820f-45f3-a4dd-d3f120ce13ed
y = machine(encoder_y, y_imputed) |> fit! |> MLJ.transform

# ╔═╡ c34fa114-c937-457f-ba26-aa1b06c3d8d9
schema(X)

# ╔═╡ 5bcbafad-acd0-4058-8ac2-b0ccdce6ad87
schema(y)

# ╔═╡ 717fc689-db96-4f0c-aa06-5ea47c123a9a
md"Split data to train & test sets"

# ╔═╡ d09b013d-ae68-424a-9f9f-cac6885f7953
(Xtrain, Xtest), (ytrain, ytest) = partition((X, y), .8, rng=123, multi=true);

# ╔═╡ b5a83a2e-059e-49e4-bfd5-21cc14d3ac4b
md"Standardizer"

# ╔═╡ 604a5a3f-a569-4816-8f97-2d945a48a02b
sc_ = Standardizer()

# ╔═╡ 7f9722d9-ab0c-4a6b-9a81-4b22bf587e23
sc_age = machine(sc_, Xtrain.Age) |> fit!

# ╔═╡ 0688251e-bc57-4b6a-aa9a-259a51351fc8
Xtrain.Age = MLJ.transform(sc_age, Xtrain.Age)

# ╔═╡ 265d068a-18e7-4e32-a4fa-cb341f13dde2
Xtest.Age = MLJ.transform(sc_age, Xtest.Age)

# ╔═╡ ccee508e-ec60-46c8-88b8-b292a66bd6e7
sc_salary = machine(sc_, Xtrain.Salary) |> fit!

# ╔═╡ e1859e5d-1d83-4461-86c1-8f032f09205e
Xtrain.Salary = MLJ.transform(sc_salary, Xtrain.Salary)

# ╔═╡ 092355f9-6b62-4062-8687-0d4bb1e9c235
Xtest.Salary = MLJ.transform(sc_salary, Xtest.Salary) 

# ╔═╡ 09a2cb38-a2ac-4d40-8d01-7ede680510a2
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
# ╠═aa831e6c-66db-4884-90a4-3184bd540781
# ╠═4ff29318-5d25-4909-adc6-3a2a8bc578b8
# ╠═e75026f1-f5e4-48a1-b9c4-d65a19dfdd9d
# ╠═6493ef22-17b1-4851-b537-b34496e3bc69
# ╠═60c276fa-3605-4cc0-9805-ca331b5892a5
# ╠═f8a8da51-ae01-476a-865e-3acd40038966
# ╠═d4eaf94b-1893-4b65-9c24-8a102dd7b0cb
# ╠═7a44f465-c922-4eb8-a8fb-b42f39c5571c
# ╠═69eafcf8-0a9b-4eee-9416-1654686140a5
# ╠═28eb8e6f-67ee-4bb0-9b7c-8aa9673e4f9b
# ╠═80120c69-e14d-4b10-b71f-e61eeaba86ef
# ╠═246738aa-83cd-44ac-8e77-59c9c91661d2
# ╠═de4130b1-5730-44e1-813f-8c3539b700ea
# ╠═3d350b68-6de4-4d64-9040-7e24ff6c831d
# ╠═e4795cf3-e5be-403d-a899-0609f862df05
# ╠═f2e650cf-98af-4038-9d30-53103d279197
# ╠═e4038ff0-8841-407d-b73a-a569d838b15c
# ╠═fb5b6cae-46c1-4806-95d6-6feec6ea05ec
# ╠═5e449816-8278-4acd-b1bc-ba56385618f5
# ╠═466b7405-6e69-4244-a5d3-bf33f02bcf28
# ╠═d1ac47ef-1372-4937-8912-8b00e892b66e
# ╠═5abb2589-fb45-4ade-966d-44a77b9d2ee4
# ╠═ff5abaf3-533f-4249-bc1c-7233e1bdcdcf
# ╠═efa2bbf1-3b2a-467a-a446-8697a417404a
# ╠═af4eed5f-0476-47d4-9a93-498b90e5a2f5
# ╠═4af41c48-caa5-4261-b1cd-8c0a05d4acad
# ╠═d0f8e9f9-820f-45f3-a4dd-d3f120ce13ed
# ╠═c34fa114-c937-457f-ba26-aa1b06c3d8d9
# ╠═5bcbafad-acd0-4058-8ac2-b0ccdce6ad87
# ╠═717fc689-db96-4f0c-aa06-5ea47c123a9a
# ╠═d09b013d-ae68-424a-9f9f-cac6885f7953
# ╠═b5a83a2e-059e-49e4-bfd5-21cc14d3ac4b
# ╠═604a5a3f-a569-4816-8f97-2d945a48a02b
# ╠═7f9722d9-ab0c-4a6b-9a81-4b22bf587e23
# ╠═0688251e-bc57-4b6a-aa9a-259a51351fc8
# ╠═265d068a-18e7-4e32-a4fa-cb341f13dde2
# ╠═ccee508e-ec60-46c8-88b8-b292a66bd6e7
# ╠═e1859e5d-1d83-4461-86c1-8f032f09205e
# ╠═092355f9-6b62-4062-8687-0d4bb1e9c235
# ╟─09a2cb38-a2ac-4d40-8d01-7ede680510a2
