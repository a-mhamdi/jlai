#########################
#= LOGISTIC REGRESSION =#
#########################

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
        :Purchased => Multiclass)
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

md"Load the `LogisticClassifier` & bind it to `lc_`"
LC = @load LogisticClassifier pkg=MLJLinearModels
lc_ = LC()

md"You may want to see [MLJLinearModels.jl](https://github.com/JuliaAI/MLJLinearModels.jl) and the unwrapped model type [`MLJLinearModels.LogisticClassifier`](@ref)."

md"Construct a pipeline and train it"
pipe_ = Pipeline(sc_, lc_)
pipe = machine(pipe_, Xtrain, ytrain) |> fit!

md"Predict the output for `Xtest`"
ŷ = predict_mode(pipe, Xtest);

md"Compute the accuracy"
acc = mean(ŷ .== ytest);
println("Accuracy is about $(round(100*acc))%")

md"Confusion Matrix"
confusion_matrix(ŷ, ytest)

md"Evaluation Metrics"
accuracy(ŷ, ytest)
precision(ŷ, ytest)
recall(ŷ, ytest)
f1score(ŷ, ytest)

md"We can estimate, more elegantly, the performance of `pipe` through the `evaluate!` command."
evaluate!(pipe, operation=predict_mode, measures=[accuracy, precision, recall, f1score])