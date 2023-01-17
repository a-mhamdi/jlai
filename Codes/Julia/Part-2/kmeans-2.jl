#= kmeans =#

## Import Librairies
using CSV, DataFrames

## Load The Dataset From CSV File
df = CSV.read("./datasets/Mall_Customers.csv", DataFrame);

## Take A Look @ Data
first(df, 5)
income = df[!, 4];
ss = df[!, 5];

## Plots PKG
using Plots
scatter(income, ss, legend=false)

## Clustering PKG
using Clustering

## Features Construction
X = hcat(ss, income);
typeof(X)
hat_clusters = kmeans(X', 5; display=:iter)

## Scatter Plot
scatter(ss, income, marker_z=hat_clusters.assignments,
    color=:winter,
    legend=false)
    
scatter!(hat_clusters.centers[1,:]', hat_clusters.centers[2,:]', 
    color=:black, 
    labels=["#1" "#2" "#3" "#4" "#5"],
    legend=true)
