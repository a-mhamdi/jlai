#= k-NEAREST NEIGHBORS =#

using CSV, DataFrames

## Load Data
df = CSV.read("./datasets/Social_Network_Ads.csv", DataFrame)
x = Float64.(df[!, 2]);
y = df[!, end];

println(typeof(x), size(x))
l = size(x)[1]

# Scatter Plot Of Data
using Plots; # unicodeplots() 
g1 = scatter(x, y; c=y, legend=false); 

using NearestNeighbors
# KDTree(data, metric; leafsize, reorder)
tree = KDTree(x')
# Initialize k for k-NN
k = 3

tst = rand(1:l, Int(.2*l)) 
# Find Nearest Neighbors Using k-NN & k-d Tree
idxs, dists = knn(tree, x[tst], k, true)
