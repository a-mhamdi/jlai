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

# ╔═╡ 012037ab-71b5-4435-a083-6560d60c8255
begin
	cd(@__DIR__)
	import Pkg; Pkg.activate("..")
end

# ╔═╡ 29fb345c-2085-45d8-b71a-1b71bca8d140
using CSV, DataFrames

# ╔═╡ 9377bdbe-9233-4716-a5e4-091b41284e2e
using Plots; theme(:dracula)

# ╔═╡ ec50ec67-c710-41f9-b93c-14d21a642bac
using MLJ

# ╔═╡ c1754fdb-f84d-401d-ac46-8e6ef2010112
using PlutoUI

# ╔═╡ 7f6a4804-ac54-4b25-83da-eca14319e7ab
md"# POLYNOMIAL LINEAR REGRESSION"

# ╔═╡ b0548d1d-8d27-4638-8d7f-999d81935219
md"
```julia
versioninfo() # -> v\"1.11.2\"
```
"

# ╔═╡ 07339eda-32b4-4e53-bd64-9a180e7c5892
md"Import librairies"

# ╔═╡ 79d051f3-98df-4f95-b94c-a9b4be17724d
md"Read data from file"

# ╔═╡ c1b809ea-14c2-4537-8763-82676a304fd9
df = CSV.read("../../Datasets/Position_Salaries.csv", DataFrame)

# ╔═╡ e6825df5-bc1a-4447-b342-705ae2606b47
schema(df)

# ╔═╡ 54b007e0-f3dd-47f1-b912-cc1e364262fd
md"Unpack data"

# ╔═╡ cdf5f466-4ebb-46d6-b4b7-d5da7c9ccace
x = select(df, :Level)

# ╔═╡ 9e0405d6-90f3-4009-aa55-66dceddff0ae
y = df.Salary

# ╔═╡ dcfd9309-9a01-4541-8fdd-6e21ae0d0180
@bind p Slider(1:1:10, default=4)

# ╔═╡ 7a170267-254b-4284-8455-eb144db79ef5
begin
	X_ = Matrix{Float32}(undef, length(x.Level), p)
	for i in 1:p
		X_[:, i] = x.Level.^i
	end
	X = Tables.table(X_);
end

# ╔═╡ 19bfa988-2581-4273-9584-cd2da025f163
md"Partition of data"

# ╔═╡ 9137d03f-ffe7-4ca3-920c-73403fd90e68
train, test = partition(eachindex(y), 0.8, shuffle=true, rng=123)

# ╔═╡ 525472a8-3792-41a3-8182-284790e0349d
md"Add extra features & scaling"

# ╔═╡ cda0b49b-738d-483b-9389-998c2602a143
sc_ = Standardizer()

# ╔═╡ b11e1b01-b65b-46bb-8c2a-dbd76768d38f
begin
	sc = machine(sc_, X)
	fit!(sc, rows=train)
end

# ╔═╡ 514973ee-f17c-4201-9216-61f676ea9d67
begin
    Xtrain = MLJ.transform(sc, rows=train)
    Xtest = MLJ.transform(sc, rows=test)
end

# ╔═╡ a2e514f6-840d-4ab5-bf32-d1c0a3a3a812
md"Load linear regression model"

# ╔═╡ 8d13f9ce-fb15-46bd-9396-46322982b5ad
LR = @load LinearRegressor pkg=MLJLinearModels

# ╔═╡ 7667a893-eb07-4bd9-b300-cc7ebaa46b3e
md"Train & fit the regression model"

# ╔═╡ 00a21d32-4c3e-4138-99f3-30f341f34b47
scitype(y)

# ╔═╡ 238d3b65-5d2a-4d99-840b-e339f58d37cd
begin
	lr = machine(LR(), X, y)
	fit!(lr, rows=train)
end

# ╔═╡ 57023c24-7192-46a2-8aeb-020e22a6baa8
params = fitted_params(lr)

# ╔═╡ add2d034-d0cd-40b4-b566-53ffb1e4e892
md"Predict & measure the error"

# ╔═╡ dd4bdb74-db52-4f83-87f0-35f91e7b84fd
yhat = predict(lr, rows=train)

# ╔═╡ 58bc41c1-f0b1-4a5c-aa69-b5392fc1de74
println("Error is $(sum(( yhat.- y[train] ).^2) ./ length(y[train]) )")

# ╔═╡ 8abfe033-41aa-40da-a78d-1734529eabf9
md"Plot & compare"

# ╔═╡ cde54b7f-5d41-4cde-9381-ac97f9d0ab07
begin
	scatter(x.Level[train], y[train], label="y(t)")
	scatter!(x.Level[train], yhat, label="ŷ(t)")
end

# ╔═╡ a64d10e5-9b0e-4583-b29a-b4d44c634796
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
# ╠═7f6a4804-ac54-4b25-83da-eca14319e7ab
# ╠═b0548d1d-8d27-4638-8d7f-999d81935219
# ╠═012037ab-71b5-4435-a083-6560d60c8255
# ╠═07339eda-32b4-4e53-bd64-9a180e7c5892
# ╠═29fb345c-2085-45d8-b71a-1b71bca8d140
# ╠═9377bdbe-9233-4716-a5e4-091b41284e2e
# ╠═ec50ec67-c710-41f9-b93c-14d21a642bac
# ╠═79d051f3-98df-4f95-b94c-a9b4be17724d
# ╠═c1b809ea-14c2-4537-8763-82676a304fd9
# ╠═e6825df5-bc1a-4447-b342-705ae2606b47
# ╠═54b007e0-f3dd-47f1-b912-cc1e364262fd
# ╠═cdf5f466-4ebb-46d6-b4b7-d5da7c9ccace
# ╠═9e0405d6-90f3-4009-aa55-66dceddff0ae
# ╠═c1754fdb-f84d-401d-ac46-8e6ef2010112
# ╠═dcfd9309-9a01-4541-8fdd-6e21ae0d0180
# ╠═7a170267-254b-4284-8455-eb144db79ef5
# ╠═19bfa988-2581-4273-9584-cd2da025f163
# ╠═9137d03f-ffe7-4ca3-920c-73403fd90e68
# ╠═525472a8-3792-41a3-8182-284790e0349d
# ╠═cda0b49b-738d-483b-9389-998c2602a143
# ╠═b11e1b01-b65b-46bb-8c2a-dbd76768d38f
# ╠═514973ee-f17c-4201-9216-61f676ea9d67
# ╠═a2e514f6-840d-4ab5-bf32-d1c0a3a3a812
# ╠═8d13f9ce-fb15-46bd-9396-46322982b5ad
# ╠═7667a893-eb07-4bd9-b300-cc7ebaa46b3e
# ╠═00a21d32-4c3e-4138-99f3-30f341f34b47
# ╠═238d3b65-5d2a-4d99-840b-e339f58d37cd
# ╠═57023c24-7192-46a2-8aeb-020e22a6baa8
# ╠═add2d034-d0cd-40b4-b566-53ffb1e4e892
# ╠═dd4bdb74-db52-4f83-87f0-35f91e7b84fd
# ╠═58bc41c1-f0b1-4a5c-aa69-b5392fc1de74
# ╠═8abfe033-41aa-40da-a78d-1734529eabf9
# ╠═cde54b7f-5d41-4cde-9381-ac97f9d0ab07
# ╟─a64d10e5-9b0e-4583-b29a-b4d44c634796
