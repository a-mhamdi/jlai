### A Pluto.jl notebook ###
# v0.20.4

#> [frontmatter]
#> title = "Tipper Problem"
#> tags = ["Fuzzy", "julialang"]
#> date = "2024-12-21"
#> description = "Fuzzy Inference System"
#> 
#>     [[frontmatter.author]]
#>     name = "A. Mhamdi"
#>     url = "https://a-mhamdi.github.io/jlai/"

using Markdown
using InteractiveUtils

# ╔═╡ 5882d747-06ed-46bd-8611-0cf9fd7980ce
begin
	cd(@__DIR__)
	import Pkg
	Pkg.activate("..")
	Pkg.status()
end

# ╔═╡ 65d0a1cf-b0b7-4971-8cc9-9a0fe785477c
using Fuzzy

# ╔═╡ 0b178d02-d55b-4ee2-81df-dfb568bd1a31
using Plots; theme(:dracula)

# ╔═╡ 264261bf-4e21-4510-a106-4854cbb30c1c
md"# TIPPER PROBLEM"

# ╔═╡ ec6eaa3e-9c69-4dc1-bc86-ae44b1977d93
md"
```julia
versioninfo() -> v\"1.11.2\"
```
"

# ╔═╡ 3c4a5ad6-0e02-4030-a0a2-8c93bc1da120
md"Import required librairies"

# ╔═╡ eb64fc7d-0143-4084-9ed6-9e71247f3dd9
md"`score` denotes the horizontal axis"

# ╔═╡ 98c49f34-3f89-48a3-a38f-f41478fbb80f
score = range(0, 10, length=100)

# ╔═╡ b35cddb4-c45f-4ab7-860b-abd02f07dd4d
md"`food` is the 1st fuzzy input"

# ╔═╡ 3280a4bb-3671-42fc-80ad-e5bd19b8ad03
food = Dict(
	    "Rancid" => TrapezoidalMF(0, 0, 2, 4),
	    "Delicious" => TrapezoidalMF(6, 8, 10, 10)
	    )

# ╔═╡ 2f4b8391-d58d-4d5d-a931-35300ecbd930
food_chart = chart_prepare(food, score)

# ╔═╡ efd91d74-2a90-49c6-8271-9feadcc683d4
md"`service` is the 2nd fuzzy input"

# ╔═╡ 251c041f-dd91-4c12-bb1e-428e7339d6a1
service = Dict(
	    "Poor" => TrapezoidalMF(0, 0, 2, 4),
	    "Good" => TrapezoidalMF(3, 4, 6, 7),
	    "Excellent" => TrapezoidalMF(6, 8, 10, 10)
	    )

# ╔═╡ 0d8d8db1-74ff-419d-84ba-a5115d9e3f01
service_chart = chart_prepare(service, score)

# ╔═╡ f46a3069-5862-49f4-9137-3c730f776deb
md"`tip` is the fuzzy output"

# ╔═╡ 9c8daf9d-e2d2-439a-a683-b7ce0dca78ba
tip = Dict(
	   "Cheap" => TrapezoidalMF(0, 0, 1, 3),
	   "Average" => TrapezoidalMF(2, 4, 6, 8),
	   "Generous" => TrapezoidalMF(7, 9, 10, 10)
	    )

# ╔═╡ e393fcc1-8a0c-4ba5-8cdc-8a8767d30543
tip_chart = chart_prepare(tip, score)

# ╔═╡ 55703f0a-20e8-45cd-9ee2-40c985143cf1
md"Design the rules set"

# ╔═╡ b53f04e3-74bb-4d94-8c21-ed615b194d07
rule_1 = Rule(["Rancid", "Poor"], "Cheap", "MAX")

# ╔═╡ 213e434d-e1bf-4398-8ba8-38a5ec77c674
rule_2 = Rule(["", "Good"], "Average", "MAX")

# ╔═╡ 65b816d8-5fe6-42dd-bf62-e042c2cc36ef
rule_3 = Rule(["Delicious", "Excellent"], "Generous", "MAX")

# ╔═╡ e6871271-fd07-4937-af28-a60740d9f3f6
md"`rules` aggregates all individual rules"

# ╔═╡ d608bece-7f66-4085-89f6-b37786d2351e
rules = [rule_1, rule_2, rule_3]

# ╔═╡ bc00fc5b-f919-425e-b89e-cfd73ce5bc8a
md"Plot the fuzzy membership variables"
#= GRAPHS =#

# ╔═╡ 9baa4c7f-88b7-4342-9ff8-c7dd775937ce
p1 = plot(score, food_chart["values"], ylabel="Food", label=food_chart["names"], legend=:bottomright)

# ╔═╡ 94772693-2397-4768-8a7f-6fe2abf12cbb
p2 = plot(score, service_chart["values"], ylabel="Service", label=service_chart["names"], legend=:bottomright)

# ╔═╡ a7c6646d-4541-4619-8562-82aaf2e10cd5
p3 = plot(score, tip_chart["values"], xlabel="Score", ylabel="Tip", label=tip_chart["names"], legend=:bottomright)

# ╔═╡ 283bd241-a880-406a-9094-390a33496031
graphs = plot(p1, p2, p3, layout=(3, 1), lw=2)
# savefig(graphs, "mf-graphs.pdf")

# ╔═╡ d46fa358-3f45-4958-a540-cf8b8337bb7b
md"**FUZZY INFERENCE SYSTEM: MAMDANI**"

# ╔═╡ dfe56972-c831-4e73-b19d-dda42bbb45c8
fis = FISMamdani([food, service], tip, rules)

# ╔═╡ 411ae0e5-4c42-4e4e-a9dd-99ed072ed85a
eval_fis(fis, [9., 8.])

# ╔═╡ Cell order:
# ╠═264261bf-4e21-4510-a106-4854cbb30c1c
# ╠═ec6eaa3e-9c69-4dc1-bc86-ae44b1977d93
# ╠═5882d747-06ed-46bd-8611-0cf9fd7980ce
# ╠═3c4a5ad6-0e02-4030-a0a2-8c93bc1da120
# ╠═65d0a1cf-b0b7-4971-8cc9-9a0fe785477c
# ╠═0b178d02-d55b-4ee2-81df-dfb568bd1a31
# ╠═eb64fc7d-0143-4084-9ed6-9e71247f3dd9
# ╠═98c49f34-3f89-48a3-a38f-f41478fbb80f
# ╠═b35cddb4-c45f-4ab7-860b-abd02f07dd4d
# ╠═3280a4bb-3671-42fc-80ad-e5bd19b8ad03
# ╠═2f4b8391-d58d-4d5d-a931-35300ecbd930
# ╠═efd91d74-2a90-49c6-8271-9feadcc683d4
# ╠═251c041f-dd91-4c12-bb1e-428e7339d6a1
# ╠═0d8d8db1-74ff-419d-84ba-a5115d9e3f01
# ╠═f46a3069-5862-49f4-9137-3c730f776deb
# ╠═9c8daf9d-e2d2-439a-a683-b7ce0dca78ba
# ╠═e393fcc1-8a0c-4ba5-8cdc-8a8767d30543
# ╠═55703f0a-20e8-45cd-9ee2-40c985143cf1
# ╠═b53f04e3-74bb-4d94-8c21-ed615b194d07
# ╠═213e434d-e1bf-4398-8ba8-38a5ec77c674
# ╠═65b816d8-5fe6-42dd-bf62-e042c2cc36ef
# ╠═e6871271-fd07-4937-af28-a60740d9f3f6
# ╠═d608bece-7f66-4085-89f6-b37786d2351e
# ╠═bc00fc5b-f919-425e-b89e-cfd73ce5bc8a
# ╠═9baa4c7f-88b7-4342-9ff8-c7dd775937ce
# ╠═94772693-2397-4768-8a7f-6fe2abf12cbb
# ╠═a7c6646d-4541-4619-8562-82aaf2e10cd5
# ╠═283bd241-a880-406a-9094-390a33496031
# ╠═d46fa358-3f45-4958-a540-cf8b8337bb7b
# ╠═dfe56972-c831-4e73-b19d-dda42bbb45c8
# ╠═411ae0e5-4c42-4e4e-a9dd-99ed072ed85a
