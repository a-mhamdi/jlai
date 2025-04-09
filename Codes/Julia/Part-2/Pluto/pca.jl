### A Pluto.jl notebook ###
# v0.20.5

using Markdown
using InteractiveUtils

# ╔═╡ 043b5b8f-c323-4659-95c3-8bcf9afa393a
begin
	cd(@__DIR__)
	import Pkg; Pkg.activate(".."); Pkg.status()
end

# ╔═╡ c50493d4-6af4-4c2a-8a79-ce9ac8862aca
using DataFrames, MLJ


# ╔═╡ 31b81a0a-1a42-45a8-93aa-c9b13beacb69
using Plots

# ╔═╡ 2be5d7cc-1336-11f0-2a07-e91d67d87458
md"# PRINCIPAL COMPONENT ANALYSIS"

# ╔═╡ 860463b5-b02d-4e1f-8ae6-43517df7a0aa
versioninfo()

# ╔═╡ 988f3571-e493-4bf4-a828-61505f67d220
data, species = @load_iris

# ╔═╡ 68426efd-7494-41e4-a82e-b7939f843000
begin
	X = DataFrame(data)
	first(X, 5)
end

# ╔═╡ 12b2398c-d998-45b6-a59d-edf62f04059c
begin
	y = coerce(species, OrderedFactor)
	first(y, 5)
end

# ╔═╡ 5a9a9945-6464-4418-9b00-73b1218b56d8
PCA = @load PCA pkg="MultivariateStats"

# ╔═╡ 9db5c4b7-f7fa-407a-a26c-c607b78df3a4
pca = PCA(; maxoutdim=2)

# ╔═╡ de8e3d0a-a7bc-4b54-96a4-63505b477745
mach_pca = machine(pca, X) |> fit!

# ╔═╡ c997b5c1-d2aa-4213-83ae-63dee9b5d9ce
mach_pca.report[:fit].principalvars

# ╔═╡ a1de1ea5-7059-4d19-a62e-556ac0187090
begin
	components = MLJ.transform(mach_pca, X)
	first(components, 5)
end

# ╔═╡ 681d6262-f414-4580-a453-bd2a498b4f0d
Plots.scatter(components.x1, components.x2, group=y, color_palette=["red", "green", "blue"])

# ╔═╡ Cell order:
# ╠═2be5d7cc-1336-11f0-2a07-e91d67d87458
# ╠═860463b5-b02d-4e1f-8ae6-43517df7a0aa
# ╠═043b5b8f-c323-4659-95c3-8bcf9afa393a
# ╠═c50493d4-6af4-4c2a-8a79-ce9ac8862aca
# ╠═988f3571-e493-4bf4-a828-61505f67d220
# ╠═68426efd-7494-41e4-a82e-b7939f843000
# ╠═12b2398c-d998-45b6-a59d-edf62f04059c
# ╠═5a9a9945-6464-4418-9b00-73b1218b56d8
# ╠═9db5c4b7-f7fa-407a-a26c-c607b78df3a4
# ╠═de8e3d0a-a7bc-4b54-96a4-63505b477745
# ╠═c997b5c1-d2aa-4213-83ae-63dee9b5d9ce
# ╠═a1de1ea5-7059-4d19-a62e-556ac0187090
# ╠═31b81a0a-1a42-45a8-93aa-c9b13beacb69
# ╠═681d6262-f414-4580-a453-bd2a498b4f0d
