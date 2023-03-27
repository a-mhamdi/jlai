##################################
#= RANDOM FOREST CLASSIFICATION =#
##################################

using Markdown

md"Import Librairies"
using CSV, DataFrames
using Plots; # unicodeplots()
using MLJ

md"Read Dataset => `df`"
df = CSV.read("../Datasets/Social_Network_Ads.csv", DataFrame)

md"Unpack Data"
features, target = unpack(df,
                          ==(:EstimatedSalary),
                          ==(:Purchased);
                          :EstimatedSalary => Continuous,
                          :Purchased => Multiclass)

md"Scatter Plot"
scatter(features, target; group=target, legend=false)

md"Convert Data"
x = Tables.table(features);
y = target;

md"Bind An Instance `rfc_` Model To Training Data"
RFC = @load RandomForestClassifier pkg=DecisionTree
rfc_ = RFC(max_depth=5, min_samples_split=3)
rfc = machine(rfc_, x, y) |> fit!

md"You may want to see [DecisionTree.jl](https://github.com/bensadeghi/DecisionTree.jl) and the unwrapped model type [`MLJDecisionTreeInterface.DecisionTree.RandomForestClassifier`](@ref)."

md"Evaluate Model"
evaluate!(rfc)
