###############################################
#= SUPPORT VECTOR MACHINE FOR CLASSIFICATION =#
###############################################

using Markdown
md"This an example of how we implemented an `SVM` in _Julia_ for classification task using the `LIBSVM` package interfacing with `MLJ` module."

using CSV, DataFrames
using MLJ

md"Load data"
df = CSV.read("../Datasets/Social_Network_Ads.csv", DataFrame)
schema(df)

md"Unpacking data"
features, target = unpack(df,
                            ==(:EstimatedSalary),           # `x` is the :EstimatedSalary Column
                            ==(:Purchased);                 # `y` is the :Purchased Column
                            :EstimatedSalary => Continuous, # Updating Scitypes
                            :Purchased => Multiclass)
    
md"Split the data into train & test sets"
train, test = partition(eachindex(target), 0.8, rng=123)
xtrain, xtest = table(features[train, :]), table(features[test, :])
ytrain, ytest = target[train], target[test]

md"Import SVC and bind it to SVM"
SVM = @load SVC pkg=LIBSVM
svm_ = SVM()

md"Train the classifier on the training data"
svm = machine(svm_, xtrain, ytrain) |> fit!

md"Use the trained classifier to make predictions on the test data"
yhat = predict(svm, xtest)

md"Confusion Matrix"
confusion_matrix(yhat, ytest)

md"Evaluation the model's performances"
accuracy(yhat, ytest)
precision(yhat, ytest)
recall(yhat, ytest)
f1score(yhat, ytest)

md"Estimate the performance of `svm`"
evaluate!(svm)