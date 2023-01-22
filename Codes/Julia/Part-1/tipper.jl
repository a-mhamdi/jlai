####################
#= TIPPER PROBLEM =#
####################

using Markdown

md"Import required librairies"
using Fuzzy
using Plots
md"`score` denotes the horizontal axis"
score = range(0, 10, length=100)
md"`food` is the 1st fuzzy input"
food = Dict(
	    "Rancid" => TrapezoidalMF(0, 0, 2, 4),
	    "Delicious" => TrapezoidalMF(6, 8, 10, 10)
	    )
food_chart = chart_prepare(food, score)
md"`service` is the 2nd fuzzy input"
service = Dict(
	    "Poor" => TrapezoidalMF(0, 0, 2, 4),
	    "Good" => TrapezoidalMF(3, 4, 6, 7),
	    "Excellent" => TrapezoidalMF(6, 8, 10, 10)
	    )
service_chart = chart_prepare(service, score)
md"`tip` is the fuzzy output"
tip = Dict(
	   "Cheap" => TrapezoidalMF(0, 0, 1, 3),
	   "Average" => TrapezoidalMF(2, 4, 6, 8),
	   "Generous" => TrapezoidalMF(7, 9, 10, 10)
	    )
tip_chart = chart_prepare(tip, score)
md"Design the rules set"
rule_1 = Rule(["Rancid", "Poor"], "Cheap", "MAX")
rule_2 = Rule(["", "Good"], "Average", "MAX")
rule_3 = Rule(["Delicious", "Excellent"], "Generous", "MAX")
md"`rules` aggregates all individual rules"
rules = [rule_1, rule_2, rule_3]
md"Plot the fuzzy membership variables"
#= GRAPHS =#
p1 = plot(score, food_chart["values"], ylabel="Food", label=food_chart["names"], legend=:bottomright)
p2 = plot(score, service_chart["values"], ylabel="Service", label=service_chart["names"], legend=:bottomright)
p3 = plot(score, tip_chart["values"], xlabel="Score", ylabel="Tip", label=tip_chart["names"], legend=:bottomright)
graphs = plot(p1, p2, p3, layout=(3, 1), lw=2)
# savefig(graphs, "../../Docs/mf-graphs.pdf")
md"## FUZZY INFERENCE SYSTEM: MAMDANI"
fis = FISMamdani([food, service], tip, rules)
eval_fis(fis, [9., 8.])
