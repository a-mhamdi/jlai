##################################
#= POLYNOMIAL LINEAR REGRESSION =#
##################################
# `versioninfo()` -> 1.11.1

using Markdown

md"Import librairies"
using CSV, DataFrames
using Plots
using MLJ

md"Read data from file"
df = CSV.read("../Datasets/Position_Salaries.csv", DataFrame)
schema(df)

md"Unpack data"
x = select(df, :Level)
n = 4
X = zeros(size(x)[1], n) # length(x.Level)
y = df.Salary

md"Partition of data"
train, test = partition(eachindex(y), 0.8, shuffle=true, rng=123)
Xtrain, Xtest = X[train, :], X[test, :]
ytrain, ytest = y[train], y[test]

md"Load linear regression model"
LR = @load LinearRegressor pkg=MLJLinearModels
lr_ = LR()

md"Add extra features & scaling"
sc_ = Standardizer()
for i in 1:n
    sc = machine(sc_, x.Level[train].^i) |> fit!
    Xtrain[:, i] = MLJ.transform(sc, x.Level[train].^i)
    Xtest[:, i] = MLJ.transform(sc, x.Level[test].^i)
end

md"Train & fit the regression model"
lr = machine(lr_, table(Xtrain), ytrain) |> fit!
params = fitted_params(lr)

md"Predict & measure the error"
yhat = predict(lr, table(Xtest))
println("Error is $(sum(( yhat.- ytest ).^2) ./ length(ytest) )")

md"Plot & compare"
using LaTeXStrings
plot(x.Level[test], ytest, label=L"y(t)")
plot!(x.Level[test], yhat, label=L"\hat{y}(t)")
