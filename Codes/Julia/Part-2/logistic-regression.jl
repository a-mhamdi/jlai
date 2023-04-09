#########################
#= LOGISTIC REGRESSION =#
#########################

using Markdown

md"Import Librairies"
using CSV, DataFrames
using MLJ

md"Read Data From CSV File"
df = CSV.read("../Datasets/Social_Network_Ads.csv", DataFrame)
schema(df)

md"Unpack Features & Target"
target, features = unpack(df,
                            ==(:Purchased),
                            !=(:Age);
                            :EstimatedSalary => Continuous,
                            :Purchased => Multiclass)

md"Split The Data Into Train & Test Sets"
train, test = partition(eachindex(target), 0.8, rng=123)
xtrain, xtest = table(features[train, :]), table(features[test, :])
ytrain, ytest = target[train], target[test]

md"Standardizer"
sc_ = Standardizer()
sc = machine(sc_, xtrain) |> fit!
Xtrain = MLJ.transform(sc, xtrain);
Xtest = MLJ.transform(sc, xtest);

md"Load The `LogisticClassifier` & Bind It To `lc`"
LC = @load LogisticClassifier pkg=MLJLinearModels
lc_ = LC()

md"You may want to see [MLJLinearModels.jl](https://github.com/JuliaAI/MLJLinearModels.jl) and the unwrapped model type [`MLJLinearModels.LogisticClassifier`](@ref)."

md"Train The Logistic Classifier"
lc = machine(lc_, xtrain, ytrain) |> fit!

md"Predict The `xtest`"
yhat = predict_mode(lc, xtest);

md"Accuracy"
acc = mean( yhat .== ytest);
println("Accuracy is about $(round(100*acc))%")

md"Confusion Matrix"
confusion_matrix(yhat, ytest)

md"Evaluation Metrics"
accuracy(yhat, ytest)
precision(yhat, ytest)
recall(yhat, ytest)
f1score(yhat, ytest)

md"Estimate the performance of `lc`"
evaluate!(lc)