#= MULTIPLE LINEAR REGRESSION =#

## Import Librairies
using CSV, DataFrames
using MLJ

## Load Data From CSV File
df = CSV.read("../../Datasets/50_Startups.csv", DataFrame)
schema(df)

## Design The Features
X = df[!, 1:4]
colnames = ["rd", "admin", "spend", "state"]
rename!(X, Symbol.(colnames))
coerce!(X, :state => Multiclass)

## Encoding The State Column
ce = ContinuousEncoder()
X = machine(ce, X) |> fit! |> MLJ.transform

## Extract Target Vector
y = df.Profit

## Preparing For The Split
train, test = partition(eachindex(y), 0.8, shuffle=true, rng=123)
Xtrain, Xtest = X[train, :], X[test, :]
ytrain, ytest = y[train], y[test]

## Load & Instantiate The Linear Regressor Class
LR = @load LinearRegressor pkg=MLJLinearModels
lr = LR()

## Train & Fit
mach = machine(lr, Xtrain, ytrain) |> fit!
println("Params of fitted model are $(fitted_params(mach))")

## Prediction
yhat = predict(mach, Xtest)

## Results & Metrics
println("Error is $(sum( (yhat .- ytest).^2 ))")
