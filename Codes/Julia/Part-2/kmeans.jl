############
#= KMEANS =#
############

using Markdown

md"It is a clustering algorithm that is used to partition an unlabeled dataset into a specified number of clusters."

md"Import Librairies"
using CSV, DataFrames
using Plots; # unicodeplots()
using MLJ

md"Load Data From CSV File"
df = CSV.read("../Datasets/Mall_Customers.csv", DataFrame);
schema(df)
first(df, 5)

md"Features"
income, ss = df[!, 4], df[!, 5];
X = hcat(ss, income);
typeof(X)

md"Take a Loot @ Data"
scatter(income, ss, legend=false)

md"Load & Instantiate `KMeans` Object"
KMeans = @load KMeans pkg=Clustering
kmeans_ = KMeans(k=5)

md"Train & Regroup Into Clusters"
kmeans = machine(kmeans_, table(X)) |> fit!

md"Clusters & Centroids"
clusters = predict(kmeans);
centroids = permutedims(kmeans.fitresult[1]);

md"Extract Clusters Values"
using CategoricalArrays
y = levelcode.(clusters);

md"Scatter Plots"
scatter(ss, income, marker_z=y, color=:winter, legend=false)
scatter!(centroids[:,1], centroids[:,2], color=:red, labels=["1" "2" "3" "4" "5"])
