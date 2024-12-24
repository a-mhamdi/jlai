### A Pluto.jl notebook ###
# v0.20.4

#> [frontmatter]
#> title = "Selection Process"
#> tags = ["fuzzy", "julialang"]
#> date = "2024-12-21"
#> description = "Fuzzy Decision Maker"
#> 
#>     [[frontmatter.author]]
#>     name = "A. Mhamdi"
#>     url = "https://a-mhamdi.github.io/jlai/"

using Markdown
using InteractiveUtils

# ╔═╡ 7600b114-8fa6-4e96-ac87-7b6dde6a8e2e
begin
	cd(@__DIR__)
	import Pkg
	Pkg.activate("..")
	Pkg.status()
end

# ╔═╡ d7f0bb85-56f6-485f-910a-765e793988e0
using Plots; theme(:dracula)

# ╔═╡ e682be72-2190-44ed-887f-4eae11eb1b10
using Fuzzy

# ╔═╡ 65c4f4fb-cfa6-42b2-b74d-a160e3ea920f
md"# SELECTION PROCESS"

# ╔═╡ 76c90b07-36d5-4ab8-95f3-1cbb02d5477d
md"
```julia
versioninfo() -> v\"1.11.2\"
```
"

# ╔═╡ 8845c516-c350-4e06-b2f1-40f8348653e5
md"Let's begin by importing the `Fuzzy` module. `Plots` is used later to draw the membership functions."

# ╔═╡ 35d99522-69ce-4611-b152-da54afc0c9cd
md"""
## Input
We denote later by `input` all the plausible values of concern in each particular situation. `input` is often referred to as the universe of discourse or universal set $(u)$.
"""

# ╔═╡ 904238c6-e7d5-44f5-a45c-27b66f07333a
input = range(0, 10, length = 1000);

# ╔═╡ 7bf02868-708f-4984-ab36-084053ff28d8
md"""
## Application
The first criterion to be used in our case is `application`. This latter represents the score given for a submitted application. We thought of using four membership functions to describe the status of any particular submission:
1. Weak
1. Moderate
1. Good
1. Strong
"""

# ╔═╡ 50085d54-4ab5-460e-86b0-aea18357db7c
application = Dict(
	"Weak" => TrapezoidalMF(0, 0, 2, 4),
	"Moderate" => TrapezoidalMF(2, 4, 5, 7),
	"Good" => TrapezoidalMF(4, 6, 7, 9),
	"Strong" => TrapezoidalMF(7, 9, 10, 10)
)

# ╔═╡ de5ccc8f-eefe-40e9-9088-b5562717e19a
md"In order to better understand the fuzzyfication process, let's plot the chart describing `application`."

# ╔═╡ 59da6bdb-e3b2-45ae-ab4d-208f8befe881
data_application = chart_prepare(application, input)

# ╔═╡ 54733a99-d234-47c4-89e5-e0697a8b9cbf
plot(
	input, data_application["values"], 
	label=data_application["names"], 
	legend=:bottomleft
)

# ╔═╡ 08ecfee2-daea-4261-b432-f0f5d599edd3
md"""
## Interview
The variable `interview` describes the score given to an apllicant after passing the interview test.
"""

# ╔═╡ 1fc2fb82-0397-4598-81e1-6bc0866b205c
interview = Dict(
	"A" => TriangularMF(0, 0, 2), 
	"B" => TriangularMF(1, 4, 6),
	"C" => TriangularMF(5, 8, 10),
	"D" => TriangularMF(9, 10, 10)
)

# ╔═╡ 53c55a53-6f51-4ca5-b6ad-7d6c962db7e8
data_interview = chart_prepare(interview, input)

# ╔═╡ 0fed8281-da55-4f66-b27f-f757e6f6d7f0
plot(
	input, data_interview["values"], 
	label=data_interview["names"], 
	legend=:bottomright
)

# ╔═╡ eb3ec344-7dc6-4b94-925a-ff588c7de389
md"It is time now to design the variable `criteria` which aggregates both `application` and `interview`."

# ╔═╡ 52b6012d-41fe-414c-a654-7ac40015508a
criteria = [application, interview]

# ╔═╡ 97e41114-bb9e-4f69-94ec-a46bb5b9b872
md"""
## Decision
As for the output, we designate by `decision` the final status of any given application.
"""

# ╔═╡ 81373eac-09e7-4f37-90bb-160476a454e1
decision = Dict(
	"Rejected" => TrapezoidalMF(0, 0, 2, 7),
	"Accepted" => TrapezoidalMF(3, 8, 10, 10)
)

# ╔═╡ 7c8d6a08-7f6c-4bcc-bbca-1054c808f2e2
data_decision = chart_prepare(decision, input)

# ╔═╡ cc90f766-3586-47bf-9116-b46fbdc1c6d8
plot(
	input, data_decision["values"], 
	label=data_decision["names"], 
	legend=:inside
)

# ╔═╡ 8c15c09a-a4d8-4697-ae97-6752c3505d75
md"""
## Set of Rules
"""

# ╔═╡ 458f28a7-edb9-425b-9067-7c78ff8bb6cf
begin
	rule_w1 = Rule(["Weak", "A"], "Rejected") 
	rule_w2 = Rule(["Weak", "B"], "Rejected")
	rule_w3 = Rule(["Weak", "C"], "Rejected")
	rule_w4 = Rule(["Weak", "D"], "Accepted")
end

# ╔═╡ 21e330c9-d024-43cb-b77f-da78ab235245
begin
	rule_m1 = Rule(["Moderate", "A"], "Rejected") 
	rule_m2 = Rule(["Moderate", "B"], "Rejected")
	rule_m3 = Rule(["Moderate", "C"], "Accepted")
	rule_m4 = Rule(["Moderate", "D"], "Accepted")
end

# ╔═╡ c371f588-2697-4c7f-a978-4f7391ce5d17
begin
	rule_g1 = Rule(["Good", "A"], "Rejected") 
	rule_g2 = Rule(["Good", "B"], "Accepted")
	rule_g3 = Rule(["Good", "C"], "Accepted")
	rule_g4 = Rule(["Good", "D"], "Accepted")
end

# ╔═╡ 52c9294a-ab36-4aa5-a030-4e1d4225f460
begin
	rule_s1 = Rule(["Strong", "A"], "Accepted") 
	rule_s2 = Rule(["Strong", "B"], "Accepted")
	rule_s3 = Rule(["Strong", "C"], "Accepted")
	rule_s4 = Rule(["Strong", "D"], "Accepted")
end

# ╔═╡ 1e2da10c-7e5a-407b-a50b-9a0b6d7d185a
rules = [
	rule_w1, rule_w2, rule_w3, rule_w4,
	rule_m1, rule_m2, rule_m3, rule_m4,
	rule_g1, rule_g2, rule_g3, rule_g4,
	rule_s1, rule_s2, rule_s3, rule_s4
]

# ╔═╡ 44dd0c81-f3b5-482a-b7d6-b2133db2f085
md"""
## Fuzzy Inference System
"""

# ╔═╡ 3b07c9ab-33c3-4714-8481-013ed86f0954
fis = FISMamdani(criteria, decision, rules)

# ╔═╡ 9089e155-dbba-490c-9ddf-5f3a76270cbe
md"Let's make some predictions"

# ╔═╡ bbc2a4c3-7095-437f-b59d-c99bfb46759e
test_in = [9., 5.]

# ╔═╡ 3be1e889-8d70-4620-be13-0c0b3b59ee2e
eval_fis(fis, test_in)

# ╔═╡ Cell order:
# ╠═65c4f4fb-cfa6-42b2-b74d-a160e3ea920f
# ╠═76c90b07-36d5-4ab8-95f3-1cbb02d5477d
# ╠═7600b114-8fa6-4e96-ac87-7b6dde6a8e2e
# ╠═8845c516-c350-4e06-b2f1-40f8348653e5
# ╠═d7f0bb85-56f6-485f-910a-765e793988e0
# ╠═e682be72-2190-44ed-887f-4eae11eb1b10
# ╠═35d99522-69ce-4611-b152-da54afc0c9cd
# ╠═904238c6-e7d5-44f5-a45c-27b66f07333a
# ╠═7bf02868-708f-4984-ab36-084053ff28d8
# ╠═50085d54-4ab5-460e-86b0-aea18357db7c
# ╠═de5ccc8f-eefe-40e9-9088-b5562717e19a
# ╠═59da6bdb-e3b2-45ae-ab4d-208f8befe881
# ╠═54733a99-d234-47c4-89e5-e0697a8b9cbf
# ╠═08ecfee2-daea-4261-b432-f0f5d599edd3
# ╠═1fc2fb82-0397-4598-81e1-6bc0866b205c
# ╠═53c55a53-6f51-4ca5-b6ad-7d6c962db7e8
# ╠═0fed8281-da55-4f66-b27f-f757e6f6d7f0
# ╠═eb3ec344-7dc6-4b94-925a-ff588c7de389
# ╠═52b6012d-41fe-414c-a654-7ac40015508a
# ╠═97e41114-bb9e-4f69-94ec-a46bb5b9b872
# ╠═81373eac-09e7-4f37-90bb-160476a454e1
# ╠═7c8d6a08-7f6c-4bcc-bbca-1054c808f2e2
# ╠═cc90f766-3586-47bf-9116-b46fbdc1c6d8
# ╠═8c15c09a-a4d8-4697-ae97-6752c3505d75
# ╠═458f28a7-edb9-425b-9067-7c78ff8bb6cf
# ╠═21e330c9-d024-43cb-b77f-da78ab235245
# ╠═c371f588-2697-4c7f-a978-4f7391ce5d17
# ╠═52c9294a-ab36-4aa5-a030-4e1d4225f460
# ╠═1e2da10c-7e5a-407b-a50b-9a0b6d7d185a
# ╠═44dd0c81-f3b5-482a-b7d6-b2133db2f085
# ╠═3b07c9ab-33c3-4714-8481-013ed86f0954
# ╠═9089e155-dbba-490c-9ddf-5f3a76270cbe
# ╠═bbc2a4c3-7095-437f-b59d-c99bfb46759e
# ╠═3be1e889-8d70-4620-be13-0c0b3b59ee2e
