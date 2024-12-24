### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ 1d7288e7-b999-46ba-8610-a24719fb05e0
begin
	cd(@__DIR__)
	import Pkg; Pkg.activate(".."); Pkg.status()
end

# ╔═╡ ed5ffc0a-66ef-477d-ba51-eeaaeca3db37
using CSV, DataFrames

# ╔═╡ ddd77de3-29fb-473e-9f8c-c11a8b6c5c6f
using MLJ

# ╔═╡ a3f4cabc-5860-49b2-8b53-75ad581bb422
using Plots; theme(:dracula)

# ╔═╡ 259f3256-e07b-4e3b-bdda-f61b1b6dd808
md"# SIMPLE LINEAR REGRESSION _(WEIGHT vs. HEIGHT)_"

# ╔═╡ 8b0d9203-17ab-4004-8123-031f86fcb097
md"
```julia
versioninfo() # -> v\"1.11.2\"
```
"

# ╔═╡ 808febd8-2525-444f-91a7-4d0eb1d45118
md"Import librairies"

# ╔═╡ fcbb3b3d-31af-490e-870e-c29c3707c621
md"Load the dataset"

# ╔═╡ cc81e0fb-a58b-43c9-9b15-b811c038ab46
df = CSV.read("../../Datasets/Weight_Height.csv", DataFrame)

# ╔═╡ 17471ec7-2c89-4cef-bd7c-7c4473b2abe4
md"Unpacking features & target"

# ╔═╡ 49dcc06d-07eb-4a8e-98ae-08df0fd636a2
md"Scatter Plot of `Weight` vs. `Height`"

# ╔═╡ d5b17dfc-574b-47cb-9f63-0b02e634b4ea
scatter(df.Height, df.Weight, label=:none, title="Weight vs. Height")

# ╔═╡ 03497fa5-e7fc-4c91-8fab-5fe95384bc5b
md"Split the data"

# ╔═╡ 839a2fbe-6093-471c-ab32-8cc0e6aec6d0
begin
	xtrain, xtest = partition(df.Height, 0.8, shuffle=true, rng=123)
	ytrain, ytest = partition(df.Weight, 0.8, shuffle=true, rng=123) # `rng` must be the same
end

# ╔═╡ 669f1fee-171c-4a4a-ad8b-14bbcb3083b5
md"Load & instantiate the linear regression Object"

# ╔═╡ 03478520-8df8-432a-9e12-a773ad885824
LR = @load LinearRegressor pkg=MLJLinearModels

# ╔═╡ 8fda62c0-2302-4b36-ba02-e8f2dbb36e18
lr_ = LR()

# ╔═╡ e1b688d7-4175-4455-925d-3f411c1b6b34
md"Train & fit"

# ╔═╡ 1dd69b09-d329-4c80-b8b8-2eed020e292e
lr = machine(lr_, Tables.table(xtrain), ytrain) |> fit!

# ╔═╡ 6dc95c84-270b-4559-b192-fb151df1b806
md"Fitted parameters"

# ╔═╡ 41ac1a22-1774-43f4-9d39-7fe8d73f8860
fitted_params(lr)

# ╔═╡ a319b68c-a3f3-49be-97f5-efdc3489bdde
md"Prediction"

# ╔═╡ 06ea44ac-e568-4d63-bf5c-5ab006205dcd
yhat = predict(lr, Tables.table(xtest))

# ╔═╡ 3113b1d1-ce05-4074-99ac-3dcc859ae408
md"Metric"

# ╔═╡ 500c799e-1749-4803-b6b6-94df60741fb6
println("Error is $(sum( (yhat .- ytest).^2 ) ./ length(ytest) )")

# ╔═╡ 783cf935-9051-4113-9b8a-8aa18042e977
scatter(xtest, ytest, label=:none)

# ╔═╡ 130a1a0b-7f1d-44ef-8c6f-f5deef194ea6
scatter!(xtest, yhat, label=:none)

# ╔═╡ Cell order:
# ╠═259f3256-e07b-4e3b-bdda-f61b1b6dd808
# ╠═8b0d9203-17ab-4004-8123-031f86fcb097
# ╠═1d7288e7-b999-46ba-8610-a24719fb05e0
# ╠═808febd8-2525-444f-91a7-4d0eb1d45118
# ╠═ed5ffc0a-66ef-477d-ba51-eeaaeca3db37
# ╠═ddd77de3-29fb-473e-9f8c-c11a8b6c5c6f
# ╠═fcbb3b3d-31af-490e-870e-c29c3707c621
# ╠═cc81e0fb-a58b-43c9-9b15-b811c038ab46
# ╠═17471ec7-2c89-4cef-bd7c-7c4473b2abe4
# ╠═49dcc06d-07eb-4a8e-98ae-08df0fd636a2
# ╠═a3f4cabc-5860-49b2-8b53-75ad581bb422
# ╠═d5b17dfc-574b-47cb-9f63-0b02e634b4ea
# ╠═03497fa5-e7fc-4c91-8fab-5fe95384bc5b
# ╠═839a2fbe-6093-471c-ab32-8cc0e6aec6d0
# ╠═669f1fee-171c-4a4a-ad8b-14bbcb3083b5
# ╠═03478520-8df8-432a-9e12-a773ad885824
# ╠═8fda62c0-2302-4b36-ba02-e8f2dbb36e18
# ╠═e1b688d7-4175-4455-925d-3f411c1b6b34
# ╠═1dd69b09-d329-4c80-b8b8-2eed020e292e
# ╠═6dc95c84-270b-4559-b192-fb151df1b806
# ╠═41ac1a22-1774-43f4-9d39-7fe8d73f8860
# ╠═a319b68c-a3f3-49be-97f5-efdc3489bdde
# ╠═06ea44ac-e568-4d63-bf5c-5ab006205dcd
# ╠═3113b1d1-ce05-4074-99ac-3dcc859ae408
# ╠═500c799e-1749-4803-b6b6-94df60741fb6
# ╠═783cf935-9051-4113-9b8a-8aa18042e977
# ╠═130a1a0b-7f1d-44ef-8c6f-f5deef194ea6
