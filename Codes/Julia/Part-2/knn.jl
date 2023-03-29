#########################
#= k-Nearest Neighbors =#
#########################

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

md"Find Nearest Neighbors"
KNN = @load KNNClassifier pkg=NearestNeighborModels
knn_ = KNN(K=3)
knn = machine(knn_, x, y) |> fit!

md"You may want to see [NearestNeighborModels.jl](https://github.com/JuliaAI/NearestNeighborModels.jl) and the unwrapped model type [`NearestNeighborModels.KNNClassifier`](@ref)."

md"Evaluate Model"
evaluate!(knn)
