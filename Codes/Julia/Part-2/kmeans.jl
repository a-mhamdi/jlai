############
#= KMEANS =#
############
# `versioninfo()` -> 1.11.1

using Markdown

md"It is a clustering algorithm that is used to partition an unlabeled dataset into a specified number of clusters."

md"Import librairies"
using CSV, DataFrames
using Plots; # unicodeplots()
using MLJ

md"Load data from CSV file"
df = CSV.read("../Datasets/Mall_Customers.csv", DataFrame);
schema(df)
first(df, 5)

md"Features"
income, ss = df[!, 4], df[!, 5];
X = hcat(ss, income);
typeof(X)

md"Take a look @ data"
scatter(income, ss, legend=false)

md"Load & instantiate `KMeans` object"
KMeans = @load KMeans pkg=Clustering
kmeans_ = KMeans(k=5)

md"You may want to see [Clustering.jl](https://github.com/JuliaStats/Clustering.jl) and the unwrapped model type [`Clustering.KMeans`](@ref)."

md"Train & regroup into clusters"
kmeans = machine(kmeans_, X) |> fit!

md"Clusters & centroids"
centroids = fitted_params(kmeans).centers

md"Extract clusters values"
y = report(kmeans).assignments

md"Scatter plots"
scatter(ss, income, marker_z=y, color=:winter, legend=false)
scatter!(centroids[1,:], centroids[2,:], color=:red, labels=['1', '2', '3', '4', '5'])
