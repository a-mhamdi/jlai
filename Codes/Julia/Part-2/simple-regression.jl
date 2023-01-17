#= SIMPLE LINEAR REGRESSION =#

## Import Librairies
using CSV, DataFrames
using MLJ

## Read Data Using .csv File. Convert It To DataFrame Object
df = CSV.read("../../Datasets/Salary_Data.csv", DataFrame)

## Unpacking Features & Target
x = df.YearsExperience
y = df.Salary 

## Preparing The Split
train, test = partition(eachindex(y), 0.8, shuffle=true, rng=123)
xtrain, xtest = x[train], x[test]
ytrain, ytest = y[train], y[test]

## Load & Instantiate The Linear Regressor Object
LR = @load LinearRegressor pkg=MLJLinearModels
lr = LR()

## Train & Fit
mach = machine(lr, Tables.table(xtrain), ytrain) |> fit!
fit!(mach)

## Prediction
yhat = predict(mach, Tables.table(xtest))

## Results & Metrics
fitted_params(mach)
println("Error is $(sum( (yhat .- ytest).^2 ))")
