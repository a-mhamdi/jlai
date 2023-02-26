###########################################
#= COMMON DATA PREPROCESSING `WORKFLOWS` =#
###########################################

using Markdown

md"Import Librairies"
using CSV, DataFrames
using MLJ

md"Import Data From CSV File"
df = CSV.read("../Datasets/Data.csv", DataFrame)
describe(df)
nrow(df), ncol(df)
schema(df)

md"Scientific Type Coercion"
df_coerced = coerce(df,
    :Country => Multiclass,
    :Age => Continuous,
    :Salary => Continuous,
    :Purchased => Multiclass);
schema(df_coerced)

md"Missing Values Imputation"
imputer = FillImputer()
mach = machine(imputer, df_coerced) |> fit!
df_imputed = MLJ.transform(mach, df_coerced);
schema(df_imputed)

#= CAN BE WRITTEN THIS WAY
df_imputed = machine(imputer, df_coerced) |> fit! |> MLJ.transform
=#

md"Features & Target Selection"
X_imputed = select(df_imputed,
    :Country, # __France, :Country__Germany, :Country__Spain, # levels(df.Country)
    :Age,
    :Salary)
y_imputed = select(df_imputed, :Purchased)

md"Feature Encoding"
encoder_X = ContinuousEncoder()
encoder_y = ContinuousEncoder(drop_last=true)

#=
mach_X = machine(encoder_X, X_imputed) |> fit!
mach_y = machine(encoder_y, y_imputed) |> fit!
X = MLJ.transform(mach_X, X_imputed);
y = MLJ.transform(mach_y, y_imputed);
=#

X = machine(encoder_X, X_imputed) |> fit! |> MLJ.transform
y = machine(encoder_y, y_imputed) |> fit! |> MLJ.transform
schema(X)
schema(y)

md"Split Data To Train & Test Sets"
(Xtrain, Xtest), (ytrain, ytest) = partition((X, y), .8, rng=123, multi=true);

md"Standardizer"
sc_ = Standardizer()

sc_age = machine(sc_, Xtrain.Age) |> fit! 
Xtrain.Age = MLJ.transform(sc_age, Xtrain.Age) 
Xtest.Age = MLJ.transform(sc_age, Xtest.Age) 

sc_salary = machine(sc_, Xtrain.Salary) |> fit! 
Xtrain.Salary = MLJ.transform(sc_salary, Xtrain.Salary) 
Xtest.Salary = MLJ.transform(sc_salary, Xtest.Salary) 

#=
vscodedisplay(Xtrain), vscodedisplay(Xtest)
vscodedisplay(ytrain), vscodedisplay(ytest)
=#
