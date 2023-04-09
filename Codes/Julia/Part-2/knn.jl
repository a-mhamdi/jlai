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

md"Split The Data Into Train & Test Sets"
train, test = partition(eachindex(target), 0.8, rng=123)
xtrain, xtest = table(features[train, :]), table(features[test, :])
ytrain, ytest = target[train], target[test]

md"Find Nearest Neighbors"
KNN = @load KNNClassifier pkg=NearestNeighborModels
knn_ = KNN(K=3)
knn = machine(knn_, xtrain, ytrain) |> fit!

md"You may want to see [NearestNeighborModels.jl](https://github.com/JuliaAI/NearestNeighborModels.jl) and the unwrapped model type [`NearestNeighborModels.KNNClassifier`](@ref)."

md"Make predictions on `xtest`"
yhat = predict_mode(knn, xtest)

md"Confusion Matrix"
confusion_matrix(yhat, ytest)

md"Evaluation the model's performances"
accuracy(yhat, ytest)
precision(yhat, ytest)
recall(yhat, ytest)
f1score(yhat, ytest)

md"Estimate the performance of `knn`"
evaluate!(knn)