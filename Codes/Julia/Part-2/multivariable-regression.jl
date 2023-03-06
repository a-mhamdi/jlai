#####################################
#= MULTIVARIABLE LINEAR REGRESSION =#
#####################################

using Markdown

md"Import Librairies"
using CSV, DataFrames
using MLJ

md"Load Data From CSV File"
df = CSV.read("../Datasets/50_Startups.csv", DataFrame)
schema(df)

md"Design The Features"
X = df[!, 1:4]
colnames = ["rd", "admin", "spend", "state"]
rename!(X, Symbol.(colnames))
coerce!(X, :state => Multiclass)

md"Encoding The State Column"
ce = ContinuousEncoder()
X = machine(ce, X) |> fit! |> MLJ.transform

md"Extract Target Vector"
y = df.Profit

md"Preparing For The Split"
train, test = partition(eachindex(y), 0.8, shuffle=true, rng=123)
Xtrain, Xtest = X[train, :], X[test, :]
ytrain, ytest = y[train], y[test]

md"Load & Instantiate The Linear Regressor Class"
LR = @load LinearRegressor pkg=MLJLinearModels
lr_ = LR()

md"Train & Fit"
lr = machine(lr_, Xtrain, ytrain) |> fit!
println("Params of fitted model are $(fitted_params(lr))")

md"Prediction"
yhat = predict(lr, Xtest)

md"Results & Metrics"
println("Error is $(sum((yhat .- ytest).^2) ./ length(ytest))")
