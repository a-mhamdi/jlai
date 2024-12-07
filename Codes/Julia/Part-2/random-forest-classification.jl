##################################
#= RANDOM FOREST CLASSIFICATION =#
##################################
# `versioninfo()` -> 1.11.1

using Markdown

md"Import librairies"
using CSV, DataFrames, Plots
using MLJ

md"Read dataset -> `df`"
df = CSV.read("../Datasets/Social_Network_Ads.csv", DataFrame)

md"Unpack data"
features, target = unpack(df,
                          ==(:EstimatedSalary),
                          ==(:Purchased);
                          :EstimatedSalary => Continuous,
                          :Purchased => Multiclass)

md"Scatter plot"
scatter(features, target; group=target, legend=false)

md"Convert data to tabular format"
x = Tables.table(features);
y = target;

md"Bind an instance `rfc_` model to training data"
RFC = @load RandomForestClassifier pkg=DecisionTree
rfc_ = RFC(max_depth=5, min_samples_split=3)
rfc = machine(rfc_, x, y) |> fit!

md"You may want to see [DecisionTree.jl](https://github.com/bensadeghi/DecisionTree.jl) and the unwrapped model type [`MLJDecisionTreeInterface.DecisionTree.RandomForestClassifier`](@ref)."

md"Evaluate the model"
evaluate!(rfc)
