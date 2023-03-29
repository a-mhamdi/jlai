###############################################
#= SUPPORT VECTOR MACHINE FOR CLASSIFICATION =#
###############################################

using Markdown
md"Here is an example of how we might implement an `SVM` in _Julia_ for classification tasks using the `LIBSVM` package interfacing with `MLJ` module."

using CSV, DataFrames
using MLJ

md"Load Data"
df = CSV.read("../Datasets/Social_Network_Ads.csv", DataFrame)
schema(df)

md"Unpacking Data"
x, y = unpack(df,
    ==(:EstimatedSalary),           # `x` is the :EstimatedSalary Column
    ==(:Purchased);                 # `y` is the :Purchased Column
    :EstimatedSalary => Continuous, # Updating Scitypes
    :Purchased => Multiclass)
    
md"Splitting Data into Train and Test"
train, test = partition(eachindex(y), 0.8, shuffle=true, rng=123)

md"Import SVC and bind it to SVM"
SVM = @load SVC pkg=LIBSVM
svm_ = SVM()

md"Train the classifier on the training data"
svm = machine(svm_, Tables.table(x[train]), y[train]) |> fit!

md"Use the trained classifier to make predictions on the test data"
y_hat = predict(svm, Tables.table(X[test]))

md"Evaluate the model's performance"
accuracy = mean(y_hat .== y[test]);
println("Accuracy is about $(round(100*accuracy))%")
