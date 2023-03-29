##############################
#= SIMPLE LINEAR REGRESSION =#
##############################
# SALARY vs. YEARSEXPERIENCE #

using Markdown

md"Import Librairies"
using CSV, DataFrames
using MLJ

md"Read Data Using .csv File. Convert It To DataFrame Object"
df = CSV.read("../Datasets/Salary_Data.csv", DataFrame)

md"Unpacking Features & Target"
x = df.YearsExperience
y = df.Salary

md"Scatter Plot of `Salary` vs. `YearsExperience`"
using Plots
scatter(x, y, label=:none, title="Salary vs. YearsExperience")

md"Preparing The Split"
train, test = partition(eachindex(y), 0.8, shuffle=true, rng=123)
xtrain, xtest = x[train], x[test]
ytrain, ytest = y[train], y[test]

md"Load & Instantiate The Linear Regressor Object"
LR = @load LinearRegressor pkg=MLJLinearModels
lr_ = LR()

md"You may want to see [MLJLinearModels.jl](https://github.com/JuliaAI/MLJLinearModels.jl) and the unwrapped model type [`MLJLinearModels.LinearRegressor`](@ref)."

md"Train & Fit"
lr = machine(lr_, Tables.table(xtrain), ytrain) |> fit!

md"Fitted Parameters"
fitted_params(lr)

md"Prediction"
yhat = predict(lr, Tables.table(xtest))

md"Error Measurement"
println("Error is $(sum( (yhat .- ytest).^2 ) ./ length(ytest) )")

scatter(xtest, ytest, label=:none)
scatter!(xtest, yhat, label=:none)