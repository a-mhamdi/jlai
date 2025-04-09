### A Pluto.jl notebook ###
# v0.20.5

using Markdown
using InteractiveUtils

# ╔═╡ 3b39b794-0ff4-11f0-30d1-dbda9bb6d886
begin
	cd(@__DIR__)
	import Pkg; Pkg.activate(".."); Pkg.status()
end

# ╔═╡ b74dfdab-272a-4f3f-ba18-2407313c09b1
begin
	using MLJ
	using Plots
end

# ╔═╡ 5f77fa36-9b5b-4fd4-8da5-248e407649c9
md"# HYPERPARAMETERS TUNING" 

# ╔═╡ 21f6079a-d82e-47e8-98a6-8e2bd5c3b59d
versioninfo()

# ╔═╡ 051021bb-8ef8-4c55-b5a6-30763e73cf3a
begin
	X, y_ = @load_iris
	schema(X)
end

# ╔═╡ e242efbb-e1a8-474a-8104-7c8f129dd9cf
sc = Standardizer()

# ╔═╡ dfd15c72-32c9-48fe-8d4d-686d77faa3ea
machine(sc, X) |> x -> fit!(x) |> MLJ.transform

# ╔═╡ 230b9073-af0d-432e-ba05-080bc8af44fa
y = coerce(y_, Multiclass)

# ╔═╡ 75a8e8d9-8063-4687-82f6-a47bcc0a962e
begin
	RF_Model = @load RandomForestClassifier pkg=DecisionTree
	model = RF_Model()
end

# ╔═╡ 16b6c7b8-10e8-4c5e-bee8-71f61d59a550
begin
	r1 = range(model, :min_purity_increase, lower=0.001, upper=1.0, scale=:log)
	r2 = range(model, :n_trees, lower=50, upper=200)
end

# ╔═╡ ce0f3148-46f7-4b3f-91b9-a24300edb120
md"## Tuning using a GRID SEARCH"

# ╔═╡ 87de11bb-c057-4637-9962-8fdde09ddeb2
self_tuning_grid = TunedModel(model=model, tuning=Grid(goal=30), resampling=CV(nfolds=5, shuffle=true), range=[r1, r2], measure=accuracy)

# ╔═╡ a8773144-f2ad-42a0-a51c-ae532a53aac4
mach_grid = machine(self_tuning_grid, X, y_)

# ╔═╡ 8cfa60bb-dcb4-4468-8a82-46151f1bd5d5
fit!(mach_grid, verbosity=0)

# ╔═╡ 889433d6-57ce-4a28-8e96-98a043017e0b
plot(mach_grid)

# ╔═╡ bad61ae0-ef8a-45cc-8b3d-34a75ad99c1c
entry_grid = report(mach_grid).best_history_entry # [1].min_purity_increase

# ╔═╡ 7fe91fd3-581f-4f5b-b043-b8f3c6dd9368
md"## Tuning using a RANDOM SEARCH"

# ╔═╡ 9c6dd75f-20bc-436f-bd70-24653fdba265
self_tuning_random = TunedModel(model=model, tuning=RandomSearch(), resampling=CV(nfolds=5, shuffle=true), range=[r1, r2], measure=accuracy)

# ╔═╡ 6358a949-01a2-4e56-a559-11d030e502ee
mach_random = machine(self_tuning_random, X, y)

# ╔═╡ de350df8-71e9-490f-bceb-594e21762ec1
fit!(mach_random, verbosity=0)

# ╔═╡ 302f19a2-b420-443f-8c62-dc16c90bc891
plot(mach_random)

# ╔═╡ 00416e26-3a5b-4100-b71d-9dc513a54e4a
# ╠═╡ disabled = true
#=╠═╡
entry_random = report(mach_random).best_history_entry # [1].min_purity_increase
  ╠═╡ =#

# ╔═╡ Cell order:
# ╠═5f77fa36-9b5b-4fd4-8da5-248e407649c9
# ╠═21f6079a-d82e-47e8-98a6-8e2bd5c3b59d
# ╠═3b39b794-0ff4-11f0-30d1-dbda9bb6d886
# ╠═b74dfdab-272a-4f3f-ba18-2407313c09b1
# ╠═051021bb-8ef8-4c55-b5a6-30763e73cf3a
# ╠═e242efbb-e1a8-474a-8104-7c8f129dd9cf
# ╠═dfd15c72-32c9-48fe-8d4d-686d77faa3ea
# ╠═230b9073-af0d-432e-ba05-080bc8af44fa
# ╠═75a8e8d9-8063-4687-82f6-a47bcc0a962e
# ╠═16b6c7b8-10e8-4c5e-bee8-71f61d59a550
# ╠═ce0f3148-46f7-4b3f-91b9-a24300edb120
# ╠═87de11bb-c057-4637-9962-8fdde09ddeb2
# ╠═a8773144-f2ad-42a0-a51c-ae532a53aac4
# ╠═8cfa60bb-dcb4-4468-8a82-46151f1bd5d5
# ╠═889433d6-57ce-4a28-8e96-98a043017e0b
# ╠═bad61ae0-ef8a-45cc-8b3d-34a75ad99c1c
# ╠═7fe91fd3-581f-4f5b-b043-b8f3c6dd9368
# ╠═9c6dd75f-20bc-436f-bd70-24653fdba265
# ╠═6358a949-01a2-4e56-a559-11d030e502ee
# ╠═de350df8-71e9-490f-bceb-594e21762ec1
# ╠═302f19a2-b420-443f-8c62-dc16c90bc891
# ╠═00416e26-3a5b-4100-b71d-9dc513a54e4a
