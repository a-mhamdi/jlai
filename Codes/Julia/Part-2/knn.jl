#= k-Nearest Neighbors =#

## Import Librairies
using CSV, DataFrames
using Plots; # unicodeplots()
using MLJ

## Read Dataset => `df`
df = CSV.read("../../Datasets/Social_Network_Ads.csv", DataFrame)

## Unpack Data
features, target = unpack(df,
                          ==(:EstimatedSalary),
                          ==(:Purchased);
                          :EstimatedSalary => Continuous,
                          :Purchased => Multiclass)

## Scatter Plot
scatter(features, target; group=target, legend=false)

## Convert Data
x = Tables.table(features);
y = target;

## Find Nearest Neighbors
KNN = @load KNNClassifier pkg=NearestNeighborModels
knn = KNN(K=3)
mach = machine(knn, x, y) |> fit!

## Evaluate Model
evaluate!(mach)
