###############################################
#= SUPPORT VECTOR MACHINE FOR CLASSIFICATION =#
###############################################
# `versioninfo()` -> 1.11.1

using Markdown

md"This an example of how we implemented an `SVM` in _Julia_ for classification task using the `LIBSVM` package interfacing with `MLJ` module."

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

md"Import SVC and bind it to SVM"
SVM = @load SVC pkg=LIBSVM
svm_ = SVM() # If you want to change the `kernel`, please install `LIBSVM` pkg.

md"Fit a pipeline to the training data"
pipe_ = Pipeline(sc_, svm_)
pipe = machine(pipe_, Xtrain, ytrain) |> fit!

md"Let's make some predictions"
ŷ = predict(pipe, Xtest)

md"Confusion Matrix"
confusion_matrix(ŷ, ytest)

md"Evaluation Metrics"
accuracy(ŷ, ytest)
specificity(ŷ, ytest) # specificity, true negative rate: TN/(TN+FP)
sensitivity(ŷ, ytest) # sensitivity, true positive rate: TP/(TP+FN)
f1score(ŷ, ytest)

md"We can estimate the performance of `pipe` through the `evaluate!` command."
evaluate!(pipe, operation=predict)
