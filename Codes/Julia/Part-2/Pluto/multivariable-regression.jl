### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ 47d72653-b058-43bd-8add-255c6a030da5
begin
	cd(@__DIR__)
	import Pkg; Pkg.activate("..")
end

# ╔═╡ ca65a07e-e2b9-4fdd-a775-e47af2cbfbd4
using CSV, DataFrames

# ╔═╡ 75250206-e124-4a09-bee6-f9c2bb82c4df
using MLJ

# ╔═╡ 0695ea0e-6206-453b-8df3-7561957fddd7
md"# MULTIVARIABLE LINEAR REGRESSION"

# ╔═╡ d9d6bcc4-5983-4355-bad2-678f21703572
md"
```julia
versioninfo() # -> v\"1.11.2\"
```
"

# ╔═╡ 9ca4d9d1-e452-4c8c-b1bc-29e6610cb9fc
md"Import librairies"

# ╔═╡ a1c28b92-cf43-40db-b83d-656b26603122
md"Load data from CSV file"

# ╔═╡ f8384ad1-ae48-4f85-8f22-0b318cfa45c3
df = CSV.read("../../Datasets/50_Startups.csv", DataFrame)

# ╔═╡ 8b5f5614-0b0e-4d54-a275-07b16575f5c4
schema(df)

# ╔═╡ 346f0093-570c-4f46-adeb-5fcfbe0d19d4
md"Design the features"

# ╔═╡ 345686e8-4add-42f5-93bc-0ce77fb9c57e
colnames = ["rd", "admin", "spend", "state"]

# ╔═╡ bdede73f-360e-4b1e-8a93-ec38c13723d0
begin
	X_ = df[!, 1:4]
	rename!(X_, Symbol.(colnames))
	coerce!(X_, :state => Multiclass)
end

# ╔═╡ 04da874b-d924-4d67-8760-72cabde38125
md"Encoding the state column"

# ╔═╡ 63d81c49-0992-435c-acf7-f9c8a1cb16e9
ce = ContinuousEncoder()

# ╔═╡ bf1a9860-2a35-4981-af88-b4c2e0ca0f9c
X = machine(ce, X_) |> fit! |> MLJ.transform

# ╔═╡ da56acf8-7b3d-4c8f-ac98-0f1d21eb4915
md"Extract target vector"

# ╔═╡ 1b83d407-e318-4017-9646-1ff76bec4db0
y = df.Profit

# ╔═╡ 6052b3da-de93-48b6-a71a-656a85b758cb
md"Preparing for the split"

# ╔═╡ 2e4bd0b2-71b9-41c8-a045-5af3c96edaeb
train, test = partition(eachindex(y), 0.8, shuffle=true, rng=123)

# ╔═╡ c3ce413f-19ac-4b62-a911-7139a74f907e
Xtrain, Xtest = X[train, :], X[test, :]

# ╔═╡ e1bf455b-fe9b-46ab-a0e2-e3bd048bb17c
ytrain, ytest = y[train], y[test]

# ╔═╡ d41153c1-ef30-4900-9562-0ac62f706bad
md"Load & instantiate the linear regression model"

# ╔═╡ 3a4a42cd-0282-431d-a198-df7cfcc201de
LR = @load LinearRegressor pkg=MLJLinearModels

# ╔═╡ 4050d857-f995-45cd-81a0-27caf068bcfb
lr_ = LR()

# ╔═╡ 52667011-9f20-4ec6-a403-a74c2b8c6e1c
md"You may want to see [MLJLinearModels.jl](https://github.com/JuliaAI/MLJLinearModels.jl) and the unwrapped model type [`MLJLinearModels.LinearRegressor`](@ref)."

# ╔═╡ 37ad859f-c76c-4e39-9beb-61f1aae0c6b6
md"Train & fit"

# ╔═╡ 9c5c389a-3874-4c2c-bf02-8d29f6e2b490
lr = machine(lr_, Xtrain, ytrain) |> fit!

# ╔═╡ 8a06056e-1714-4659-a13e-02ad15bfba3f
println("Params of fitted model are $(fitted_params(lr))")

# ╔═╡ dcc5f361-7de7-4710-885d-7df359770af4
md"Prediction"

# ╔═╡ a75dbe19-d208-48ed-a254-1bf5c948457c
yhat_lr = predict(lr, Xtest)

# ╔═╡ 1dc3df14-b7cc-46d0-b4b5-b94dea650685
md"Results & metrics"

# ╔═╡ 04efaf69-3679-429d-8cca-efb6814891db
println("Error is $(sum((yhat_lr .- ytest).^2) ./ length(ytest))")

# ╔═╡ 6a982384-e68d-4f20-b576-1dde3bb8e7c2
md"Using `MLJ` Builtin Methods For Evaluation"

# ╔═╡ 7c3391d7-0899-435c-8878-42888182242d
MLJ.evaluate!(lr, measure=[l1, l2, rms])

### RIDGE REGRESSOR

# ╔═╡ 2f604799-9de6-4e10-aa33-f16e73defb6c
md"Load Ridge Regressor"

# ╔═╡ 5414674f-3620-428e-bdac-59fa56031a8c
RIDGE = @load RidgeRegressor pkg=MLJLinearModels

# ╔═╡ 63c8822d-9541-4b08-af56-38be05263e9c
ridge_= RIDGE()

# ╔═╡ c581b220-356c-42f0-8f42-bb28f5f4b64d
md"Train & fit the model"

# ╔═╡ ad10bfd2-415e-4d14-9429-4b9d8fc47ef4
ridge = machine(ridge_, Xtrain, ytrain) |> fit!

# ╔═╡ 738a9c94-b0ce-4f99-9f05-752bb53fd0ad
md"Evalute the model"

# ╔═╡ e324e3d7-40db-4409-afef-c45f0a3a6f9c
yhat_ridge = predict(ridge, Xtest)

# ╔═╡ c07b6017-2563-44db-a1a0-02b8dea760f7
println("Error is $(sum((yhat_ridge .- ytest).^2) ./ length(ytest))")

# ╔═╡ 25f91dd4-05d4-4edb-b7d3-7d24de3c01e1
md"### LASSO REGRESSOR"

# ╔═╡ c8b46cd5-e69d-4d82-af3a-8b9833e52d1c
md"Load Lasso Regressor"

# ╔═╡ c4f8aeda-74f2-4da1-ac4a-42b9751cb1c8
LASSO = @load LassoRegressor pkg=MLJLinearModels

# ╔═╡ b62223b8-e4cc-4c18-84bd-426e75563246
lasso_= LASSO()

# ╔═╡ a97ce729-fdff-4fbe-84cc-92ebb71de0c6
md"Train & fit the model"

# ╔═╡ f7952fd2-5a69-48a1-b7d2-be013fc18c27
lasso = machine(lasso_, Xtrain, ytrain) |> fit!

# ╔═╡ de935aef-f8a4-427e-af4c-00804446ae9a
md"Evalute the model"

# ╔═╡ 5f625bd4-b1dd-46e8-a040-487226588388
yhat_lasso = predict(lasso, Xtest)

# ╔═╡ 42c3615d-1483-4350-966f-e50fe5ca4df6
println("Error is $(sum((yhat_lasso .- ytest).^2) ./ length(ytest))")

# ╔═╡ 7d2612c2-0785-43b4-b165-28f014b25cd8
md"### ELASTIC NET REGRESSOR"

# ╔═╡ d6161806-603d-404b-a9f3-a1fb536f5d7e
md"Load Elastic Net Regressor"

# ╔═╡ b504f502-90c2-4191-a5ff-2a819b5c02df
EN = @load ElasticNetRegressor pkg=MLJLinearModels

# ╔═╡ b2ac0786-9c0c-49d2-a6ea-0d7455e0134f
en_= EN(lambda=.2)

# ╔═╡ 33864a65-f46c-49a3-8e04-47ac1ae1cdb7
md"Train & fit the model"

# ╔═╡ 1fbb4ec9-c947-45e6-824c-c3a6e015b174
en = machine(en_, Xtrain, ytrain) |> fit!

# ╔═╡ 133540b0-464e-469d-90d2-5e44c804bdda
md"Evalute the model"

# ╔═╡ c49e6e0e-ea7e-458f-8cc4-91fed0c52d87
yhat_en = predict(en, Xtest)

# ╔═╡ b1e3685c-659c-42b4-ab5d-591f3337eb71
println("Error is $(sum((yhat_en .- ytest).^2) ./ length(ytest))")

# ╔═╡ cd69639b-17a0-4dcb-8a21-ca66291d82ba
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
# ╠═0695ea0e-6206-453b-8df3-7561957fddd7
# ╠═d9d6bcc4-5983-4355-bad2-678f21703572
# ╠═47d72653-b058-43bd-8add-255c6a030da5
# ╠═9ca4d9d1-e452-4c8c-b1bc-29e6610cb9fc
# ╠═ca65a07e-e2b9-4fdd-a775-e47af2cbfbd4
# ╠═75250206-e124-4a09-bee6-f9c2bb82c4df
# ╠═a1c28b92-cf43-40db-b83d-656b26603122
# ╠═f8384ad1-ae48-4f85-8f22-0b318cfa45c3
# ╠═8b5f5614-0b0e-4d54-a275-07b16575f5c4
# ╠═346f0093-570c-4f46-adeb-5fcfbe0d19d4
# ╠═345686e8-4add-42f5-93bc-0ce77fb9c57e
# ╠═bdede73f-360e-4b1e-8a93-ec38c13723d0
# ╠═04da874b-d924-4d67-8760-72cabde38125
# ╠═63d81c49-0992-435c-acf7-f9c8a1cb16e9
# ╠═bf1a9860-2a35-4981-af88-b4c2e0ca0f9c
# ╠═da56acf8-7b3d-4c8f-ac98-0f1d21eb4915
# ╠═1b83d407-e318-4017-9646-1ff76bec4db0
# ╠═6052b3da-de93-48b6-a71a-656a85b758cb
# ╠═2e4bd0b2-71b9-41c8-a045-5af3c96edaeb
# ╠═c3ce413f-19ac-4b62-a911-7139a74f907e
# ╠═e1bf455b-fe9b-46ab-a0e2-e3bd048bb17c
# ╠═d41153c1-ef30-4900-9562-0ac62f706bad
# ╠═3a4a42cd-0282-431d-a198-df7cfcc201de
# ╠═4050d857-f995-45cd-81a0-27caf068bcfb
# ╠═52667011-9f20-4ec6-a403-a74c2b8c6e1c
# ╠═37ad859f-c76c-4e39-9beb-61f1aae0c6b6
# ╠═9c5c389a-3874-4c2c-bf02-8d29f6e2b490
# ╠═8a06056e-1714-4659-a13e-02ad15bfba3f
# ╠═dcc5f361-7de7-4710-885d-7df359770af4
# ╠═a75dbe19-d208-48ed-a254-1bf5c948457c
# ╠═1dc3df14-b7cc-46d0-b4b5-b94dea650685
# ╠═04efaf69-3679-429d-8cca-efb6814891db
# ╠═6a982384-e68d-4f20-b576-1dde3bb8e7c2
# ╠═7c3391d7-0899-435c-8878-42888182242d
# ╠═2f604799-9de6-4e10-aa33-f16e73defb6c
# ╠═5414674f-3620-428e-bdac-59fa56031a8c
# ╠═63c8822d-9541-4b08-af56-38be05263e9c
# ╠═c581b220-356c-42f0-8f42-bb28f5f4b64d
# ╠═ad10bfd2-415e-4d14-9429-4b9d8fc47ef4
# ╠═738a9c94-b0ce-4f99-9f05-752bb53fd0ad
# ╠═e324e3d7-40db-4409-afef-c45f0a3a6f9c
# ╠═c07b6017-2563-44db-a1a0-02b8dea760f7
# ╠═25f91dd4-05d4-4edb-b7d3-7d24de3c01e1
# ╠═c8b46cd5-e69d-4d82-af3a-8b9833e52d1c
# ╠═c4f8aeda-74f2-4da1-ac4a-42b9751cb1c8
# ╠═b62223b8-e4cc-4c18-84bd-426e75563246
# ╠═a97ce729-fdff-4fbe-84cc-92ebb71de0c6
# ╠═f7952fd2-5a69-48a1-b7d2-be013fc18c27
# ╠═de935aef-f8a4-427e-af4c-00804446ae9a
# ╠═5f625bd4-b1dd-46e8-a040-487226588388
# ╠═42c3615d-1483-4350-966f-e50fe5ca4df6
# ╠═7d2612c2-0785-43b4-b165-28f014b25cd8
# ╠═d6161806-603d-404b-a9f3-a1fb536f5d7e
# ╠═b504f502-90c2-4191-a5ff-2a819b5c02df
# ╠═b2ac0786-9c0c-49d2-a6ea-0d7455e0134f
# ╠═33864a65-f46c-49a3-8e04-47ac1ae1cdb7
# ╠═1fbb4ec9-c947-45e6-824c-c3a6e015b174
# ╠═133540b0-464e-469d-90d2-5e44c804bdda
# ╠═c49e6e0e-ea7e-458f-8cc4-91fed0c52d87
# ╠═b1e3685c-659c-42b4-ab5d-591f3337eb71
# ╟─cd69639b-17a0-4dcb-8a21-ca66291d82ba
