#########################
#= LOGISTIC REGRESSION =#
#########################

using Markdown

md"Import Librairies"
using CSV, DataFrames
using MLJ

md"Read Data From CSV File"
df = CSV.read("../../Datasets/Social_Network_Ads.csv", DataFrame)
schema(df)

md"Unpack Features & Target"
y, features = unpack(df,
    ==(:Purchased),
    !=(:Age);
    :EstimatedSalary => Continuous,
    :Purchased => Multiclass)

md"Split The Data Into Train & Test Sets"
train, test = partition(eachindex(y), 0.8, rng=123)
Xtrain, Xtest = Tables.table(features[train, :]), Tables.table(features[test, :])
ytrain, ytest = y[train], y[test]
md"Standardizer"
sc = Standardizer()
mach = machine(sc, Xtrain) |> fit!
Xtrain = MLJ.transform(mach, Xtrain);
Xtest = MLJ.transform(mach, Xtest);
md"Load The `LogisticClassifier` & Bind It To `lc`"
LC = @load LogisticClassifier pkg=MLJLinearModels
lc = LC()
md"Train The Logistic Classifier"
mach = machine(lc, Xtrain, ytrain) |> fit!
md"Predict The `Xtest`"
yhat = predict_mode(mach, Xtest);
md"Accuracy"
acc = mean( yhat .== ytest);
println("Accuracy is about $(round(100*acc))%")
md"Evaluation"
evaluate!(mach)
