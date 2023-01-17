#= SUPPORT VECTOR MACHINE FOR CLASSIFICATION =#

#= Here is an example of how we might implement an `SVM`` in _Julia_ for classification tasks using the `LIBSVM` package interfacing with `MLJ` module.
=#

using CSV, DataFrames
using MLJ

## Load Data
df = CSV.read("../../Datasets/Social_Network_Ads.csv", DataFrame)
schema(df)

## Unpacking Data
x, y = unpack(df,
    ==(:EstimatedSalary),           # `x` is the :EstimatedSalary Column
    ==(:Purchased);                 # `y` is the :Purchased Column
    :EstimatedSalary => Continuous, # Correcting Wrong Scitypes
    :Purchased => Multiclass)

## Splitting Data into Train and Test
train, test = partition(eachindex(y), 0.8, shuffle=true, rng=123)

## Import SVC and bind it to SVM
SVM = @load SVC pkg=LIBSVM
clf = SVM()

## Train the classifier on the training data
mach = machine(clf, Tables.table(x[train]), y[train]) |> fit!

## Use the trained classifier to make predictions on the test data
y_hat = predict(mach, Tables.table(X[test]))

## Evaluate the model's performance
accuracy = mean(y_hat .== y[test]);
println("Accuracy is about $(round(100*accuracy))%")

#=
Note that this is just one way to implement an `SVM`` in _Julia_, and there are many other packages and approaches we can use. In this example, we used the `LIBSVM`` package, which provides a convenient interface for working with `SVM`s in _Julia_.
=#
