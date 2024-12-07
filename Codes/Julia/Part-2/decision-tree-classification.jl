##################################
#= DECISION TREE CLASSIFICATION =#
##################################
# `versioninfo()` -> 1.11.1

using Markdown

md"Import librairies"
using CSV, DataFrames, Plots
using MLJ

md"Read dataset and assign it `df`"
df = CSV.read("../Datasets/Social_Network_Ads.csv", DataFrame)

md"Unpack data"
features, target = unpack(df,
                          ==(:EstimatedSalary),
                          ==(:Purchased);
                          :EstimatedSalary => Continuous,
                          :Purchased => Multiclass)

md"Scatter plot"
scatter(features, target; group=target, legend=false)

md"Convert data"
x = Tables.table(features);
y = target;

md"Bind an instance `dtc_` model to training data"
DTC = @load DecisionTreeClassifier pkg=DecisionTree
dtc_ = DTC(max_depth=5, min_samples_split=3)
dtc = machine(dtc_, x, y) |> fit!

md"You may want to see [DecisionTree.jl](https://github.com/bensadeghi/DecisionTree.jl) and the unwrapped model type [`MLJDecisionTreeInterface.DecisionTree.DecisionTreeClassifier`](@ref)."

md"Evaluate model"
evaluate!(dtc)
