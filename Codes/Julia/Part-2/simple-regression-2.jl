#######################
#= LINEAR REGRESSION =#
#######################
#  WEIGHT vs. HEIGHT  #

using Markdown

md"Import librairies"
using CSV, DataFrames
using MLJ

md"Load the dataset"
df = CSV.read("../Datasets/Weight_Height.csv", DataFrame)

md"Unpacking Features & Target"
x = df.Height
y = df.Weight

md"Preparing The Split"
train, test = partition(eachindex(y), 0.8, shuffle=true, rng=123)
xtrain, xtest = x[train], x[test]
ytrain, ytest = y[train], y[test]

md"Load & Instantiate The Linear Regressor Object"
LR = @load LinearRegressor pkg=MLJLinearModels
lr_ = LR()

md"Train & Fit"
lr = machine(lr_, Tables.table(xtrain), ytrain) |> fit!

md"Prediction"
yhat = predict(lr, Tables.table(xtest))

md"Results & Metrics"
fitted_params(lr)
println("Error is $(sum( (yhat .- ytest).^2 ))")