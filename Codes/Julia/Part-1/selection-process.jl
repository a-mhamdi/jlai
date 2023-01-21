#######################
#= SELECTION PROCESS =#
#######################

using Markdown

md"Let's begin by importing the `Fuzzy` module. `Plots` is used later to draw the membership functions."
using Plots
using Fuzzy

md"""
## Input
We denote later by `input` all the plausible values of concern in each particular situation. `input` is often referred to as the universe of discourse or universal set $(u)$.
"""
input = range(0, 10, length = 1000);

md"""
## Application
The first criterion to be used in our case is `application`. This latter represents the score given for a submitted application. We thought of using four membership functions to describe the status of any particular submission:
1. Weak
1. Moderate
1. Good
1. Strong
"""
application = Dict(
	"Weak" => TrapezoidalMF(0, 0, 2, 4),
	"Moderate" => TrapezoidalMF(2, 4, 5, 7),
	"Good" => TrapezoidalMF(4, 6, 7, 9),
	"Strong" => TrapezoidalMF(7, 9, 10, 10)
)

md"In order to better understand the fuzzyfication process, let's plot the chart describing `application`."
data_application = chart_prepare(application, input)
plot(
	input, data_application["values"], 
	label=data_application["names"], 
	legend=:bottomleft
)

md"""
## Interview
The variable `interview` describes the score given to an apllicant after passing the interview test.
"""
interview = Dict(
	"A" => TriangularMF(0, 0, 2), 
	"B" => TriangularMF(1, 4, 6),
	"C" => TriangularMF(5, 8, 10),
	"D" => TriangularMF(9, 10, 10)
)
data_interview = chart_prepare(interview, input)
plot(
	input, data_interview["values"], 
	label=data_interview["names"], 
	legend=:bottomright
)

md"It is time now to design the variable `criteria` which aggregates both `application` and `interview`."
criteria = [application, interview]

md"""
## Decision
As for the output, we designate by `decision` the final status of any given application.
"""
decision = Dict(
	"Rejected" => TrapezoidalMF(0, 0, 2, 7),
	"Accepted" => TrapezoidalMF(3, 8, 10, 10)
)
data_decision = chart_prepare(decision, input)
plot(
	input, data_decision["values"], 
	label=data_decision["names"], 
	legend=:inside
)

md"""
## Set of Rules
"""

begin
	rule_w1 = Rule(["Weak", "A"], "Rejected") 
	rule_w2 = Rule(["Weak", "B"], "Rejected")
	rule_w3 = Rule(["Weak", "C"], "Rejected")
	rule_w4 = Rule(["Weak", "D"], "Accepted")
end

begin
	rule_m1 = Rule(["Moderate", "A"], "Rejected") 
	rule_m2 = Rule(["Moderate", "B"], "Rejected")
	rule_m3 = Rule(["Moderate", "C"], "Accepted")
	rule_m4 = Rule(["Moderate", "D"], "Accepted")
end

begin
	rule_g1 = Rule(["Good", "A"], "Rejected") 
	rule_g2 = Rule(["Good", "B"], "Accepted")
	rule_g3 = Rule(["Good", "C"], "Accepted")
	rule_g4 = Rule(["Good", "D"], "Accepted")
end

begin
	rule_s1 = Rule(["Strong", "A"], "Accepted") 
	rule_s2 = Rule(["Strong", "B"], "Accepted")
	rule_s3 = Rule(["Strong", "C"], "Accepted")
	rule_s4 = Rule(["Strong", "D"], "Accepted")
end

rules = [
	rule_w1, rule_w2, rule_w3, rule_w4,
	rule_m1, rule_m2, rule_m3, rule_m4,
	rule_g1, rule_g2, rule_g3, rule_g4,
	rule_s1, rule_s2, rule_s3, rule_s4
]

md"""
## Fuzzy Inference System
"""

fis = FISMamdani(criteria, decision, rules)

md"Let's make some predictions"

test_in = [9., 5.]
eval_fis(fis, test_in)
