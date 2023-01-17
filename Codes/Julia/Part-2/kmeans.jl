#= KMEANS =#

#= It is a clustering algorithm that is used to partition an unlabeled dataset into a specified number of clusters. =#

## Import Librairies
using CSV, DataFrames
using Plots; # unicodeplots()
using MLJ

## Load Data From CSV File
df = CSV.read("../../Datasets/Mall_Customers.csv", DataFrame);
schema(df)
first(df, 5)

## Features
income, ss = df[!, 4], df[!, 5];
X = hcat(ss, income);
typeof(X)

## Take a Loot @ Data
scatter(income, ss, legend=false)

## Load & Instantiate `KMeans` Object
KMeans = @load KMeans pkg=Clustering
kmeans = KMeans(k=5)

## Train & Regroup Into Clusters
mach = machine(kmeans, table(X)) |> fit!

## Clusters & Centroids
clusters = predict(mach);
centroids = permutedims(mach.fitresult[1]);

## Extract Clusters Values
using CategoricalArrays
y = levelcode.(clusters);

## Scatter Plots
scatter(ss, income, marker_z=y, color=:winter, legend=false)
scatter!(centroids[:,1], centroids[:,2], color=:red, labels=["1" "2" "3" "4" "5"])
