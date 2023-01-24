#########################
#= k-Nearest Neighbors =#
#########################

using Markdown

md"Import Librairies"
using CSV, DataFrames
using Plots; # unicodeplots()
using MLJ

md"Read Dataset => `df`"
df = CSV.read("../../Datasets/Social_Network_Ads.csv", DataFrame)

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
knn = KNN(K=3)
mach = machine(knn, x, y) |> fit!
md"Evaluate Model"
evaluate!(mach)
