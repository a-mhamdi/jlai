##############################
#= SIMPLE LINEAR REGRESSION =#
##############################
# SALARY vs. YEARSEXPERIENCE #

using Markdown

md"Import librairies"
using CSV, DataFrames
using MLJ

md"Read data using .csv File. Convert it to `DataFrame` object"
df = CSV.read("../Datasets/Salary_Data.csv", DataFrame)

md"Unpacking features & target"
x = df.YearsExperience
y = df.Salary

md"Scatter Plot of `Salary` vs. `YearsExperience`"
using Plots
scatter(x, y, label=:none, title="Salary vs. YearsExperience")

md"Preparing the split"
train, test = partition(eachindex(y), 0.8, shuffle=true, rng=123)
xtrain, xtest = x[train], x[test]
ytrain, ytest = y[train], y[test]

md"Load & instantiate the linear regression object"
LR = @load LinearRegressor pkg=MLJLinearModels
lr_ = LR()

md"You may want to see [MLJLinearModels.jl](https://github.com/JuliaAI/MLJLinearModels.jl) and the unwrapped model type [`MLJLinearModels.LinearRegressor`](@ref)."

md"Train & fit"
lr = machine(lr_, Tables.table(xtrain), ytrain) |> fit!

md"Fitted parameters"
fitted_params(lr)

md"Prediction"
yhat = predict(lr, Tables.table(xtest))

md"Error measurement"
println("Error is $(sum( (yhat .- ytest).^2 ) ./ length(ytest) )")

scatter(xtest, ytest, label=:none)
scatter!(xtest, yhat, label=:none)
