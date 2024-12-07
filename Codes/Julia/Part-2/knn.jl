#########################
#= k-NEAREST NEIGHBORS =#
#########################
# `versioninfo()` -> 1.11.1

using Markdown

md"Import librairies"
using CSV, DataFrames
using MLJ

md"Read data from CSV file"
df = CSV.read("../Datasets/Social_Network_Ads.csv", DataFrame)
schema(df)

coerce!(df, 
        :Age => Continuous,
        :EstimatedSalary => Continuous,
        :Purchased => OrderedFactor)
schema(df)

md"Unpack features & target"
target, features = unpack(df, ==(:Purchased))

md"Scatter plot"
using Plots
scatter(features.Age, features.EstimatedSalary; group=target)

md"Split the data into train & test sets"
train, test = partition(eachindex(target), 0.8, rng=123)
Xtrain, Xtest = features[train, :], features[test, :]
ytrain, ytest = target[train], target[test]

md"Standardizer"
sc_ = Standardizer()

md"Load a knn classifier, w/ # neighbors = 3"
KNN = @load KNNClassifier pkg=NearestNeighborModels
knn_ = KNN(K=3)

md"You may want to see [NearestNeighborModels.jl](https://github.com/JuliaAI/NearestNeighborModels.jl) and the unwrapped model type [`NearestNeighborModels.KNNClassifier`](@ref)."

md"Fit a pipeline to the training data"
pipe_ = Pipeline(sc_, knn_)
pipe = machine(pipe_, Xtrain, ytrain) |> fit!

md"Let's make some predictions"
ŷ = predict_mode(pipe, Xtest)

md"Confusion Matrix"
confusion_matrix(ŷ, ytest)

md"Evaluation Metrics"
accuracy(ŷ, ytest)
specificity(ŷ, ytest) # specificity, true negative rate: TN/(TN+FP)
sensitivity(ŷ, ytest) # sensitivity, true positive rate: TP/(TP+FN)
f1score(ŷ, ytest)

md"We can estimate the performance of `pipe` through the `evaluate!` command."
evaluate!(pipe, operation=predict)
