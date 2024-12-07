#######################
#= LINEAR REGRESSION =#
#######################
#  WEIGHT vs. HEIGHT  #

# `versioninfo()` -> 1.11.1

using Markdown

md"Import librairies"
using CSV, DataFrames
using MLJ

md"Load the dataset"
df = CSV.read("../Datasets/Weight_Height.csv", DataFrame)

md"Unpacking features & target"
x = df.Height
y = df.Weight

md"Scatter Plot of `Weight` vs. `Height`"
using Plots
scatter(x, y, label=:none, title="Weight vs. Height")

md"Split the data"
train, test = partition(eachindex(y), 0.8, shuffle=true, rng=123)
xtrain, xtest = x[train], x[test]
ytrain, ytest = y[train], y[test]

md"Load & instantiate the linear regression Object"
LR = @load LinearRegressor pkg=MLJLinearModels
lr_ = LR()

md"Train & fit"
lr = machine(lr_, Tables.table(xtrain), ytrain) |> fit!

md"Fitted parameters"
fitted_params(lr)

md"Prediction"
yhat = predict(lr, Tables.table(xtest))

md"Metric"
println("Error is $(sum( (yhat .- ytest).^2 ) ./ length(ytest) )")

scatter(xtest, ytest, label=:none)
scatter!(xtest, yhat, label=:none)
