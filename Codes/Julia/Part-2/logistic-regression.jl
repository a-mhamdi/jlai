#= LOGISTIC REGRESSION =#

## Import Librairies
using CSV, DataFrames
using MLJ

## Read Data From CSV File
df = CSV.read("../../Datasets/Social_Network_Ads.csv", DataFrame)
schema(df)

## Unpack Features & Target
y, features = unpack(df,
    ==(:Purchased),
    !=(:Age);
    :EstimatedSalary => Continuous,
    :Purchased => Multiclass)

## Split The Data Into Train & Test Sets
train, test = partition(eachindex(y), 0.8, rng=123)
Xtrain, Xtest = Tables.table(features[train, :]), Tables.table(features[test, :])
ytrain, ytest = y[train], y[test]

## Standardizer
sc = Standardizer()
mach = machine(sc, Xtrain) |> fit!
Xtrain = MLJ.transform(mach, Xtrain);
Xtest = MLJ.transform(mach, Xtest);

## Load The `LogisticClassifier` & Bind It To `lc`
LC = @load LogisticClassifier pkg=MLJLinearModels
lc = LC()

## Train The Logistic Classifier
mach = machine(lc, Xtrain, ytrain) |> fit!

## Predict The `Xtest`
yhat = predict_mode(mach, Xtest);

## Accuracy
acc = mean( yhat .== ytest);
println("Accuracy is about $(round(100*acc))%")

## Evaluation
evaluate!(mach)
