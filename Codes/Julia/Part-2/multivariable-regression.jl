#####################################
#= MULTIVARIABLE LINEAR REGRESSION =#
#####################################

using Markdown

md"Import librairies"
using CSV, DataFrames
using MLJ

md"Load data from CSV file"
df = CSV.read("../Datasets/50_Startups.csv", DataFrame)
schema(df)

md"Design the features"
X = df[!, 1:4]
colnames = ["rd", "admin", "spend", "state"]
rename!(X, Symbol.(colnames))
coerce!(X, :state => Multiclass)

md"Encoding the state column"
ce = ContinuousEncoder()
X = machine(ce, X) |> fit! |> MLJ.transform

md"Extract target vector"
y = df.Profit

md"Preparing for the split"
train, test = partition(eachindex(y), 0.8, shuffle=true, rng=123)
Xtrain, Xtest = X[train, :], X[test, :]
ytrain, ytest = y[train], y[test]

md"Load & instantiate the linear regression model"
LR = @load LinearRegressor pkg=MLJLinearModels
lr_ = LR()

md"You may want to see [MLJLinearModels.jl](https://github.com/JuliaAI/MLJLinearModels.jl) and the unwrapped model type [`MLJLinearModels.LinearRegressor`](@ref)."

md"Train & fit"
lr = machine(lr_, Xtrain, ytrain) |> fit!
println("Params of fitted model are $(fitted_params(lr))")

md"Prediction"
yhat_lr = predict(lr, Xtest)

md"Results & metrics"
println("Error is $(sum((yhat_lr .- ytest).^2) ./ length(ytest))")

md"Using `MLJ` Builtin Methods For Evaluation"
MLJ.evaluate!(lr, measure=[l1, l2, rms])

### RIDGE REGRESSOR
md"Load Ridge Regressor"
RIDGE = @load RidgeRegressor pkg=MLJLinearModels
ridge_= RIDGE()
md"Train & fit the model"
ridge = machine(ridge_, Xtrain, ytrain) |> fit!
md"Evalute the model"
yhat_ridge = predict(ridge, Xtest)
println("Error is $(sum((yhat_ridge .- ytest).^2) ./ length(ytest))")

### LASSO REGRESSOR
md"Load Lasso Regressor"
LASSO = @load LassoRegressor pkg=MLJLinearModels
lasso_= LASSO()
md"Train & fit the model"
lasso = machine(lasso_, Xtrain, ytrain) |> fit!
md"Evalute the model"
yhat_lasso = predict(lasso, Xtest)
println("Error is $(sum((yhat_lasso .- ytest).^2) ./ length(ytest))")

### ELASTIC NET REGRESSOR
md"Load Elastic Net Regressor"
EN = @load ElasticNetRegressor pkg=MLJLinearModels
en_= EN(lambda=.2)
md"Train & fit the model"
en = machine(en_, Xtrain, ytrain) |> fit!
md"Evalute the model"
yhat_en = predict(en, Xtest)
println("Error is $(sum((yhat_en .- ytest).^2) ./ length(ytest))")
