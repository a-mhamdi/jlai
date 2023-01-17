#= COMMON DATA PREPROCESSING `WORKFLOWS` =#

## Import Librairies
using CSV, DataFrames
using MLJ

## Import Data From CSV File
df = CSV.read("../../Datasets/Data.csv", DataFrame)
describe(df)
nrow(df), ncol(df)
schema(df)

## Scientific Type Coercion
df_coerced = coerce(df,
    :Country => Multiclass,
    :Age => Continuous,
    :Salary => Continuous,
    :Purchased => Multiclass);
schema(df_coerced)

## Missing Values Imputation
imputer = FillImputer()
mach = machine(imputer, df_coerced) |> fit!
df_imputed = MLJ.transform(mach, df_coerced);
schema(df_imputed)

#= CAN BE WRITTEN THIS WAY
df_imputed = machine(imputer, df_coerced) |> fit! |> MLJ.transform
=#

## Features & Target Selection
X_imputed = select(df_imputed,
    :Country,#__France, :Country__Germany, :Country__Spain, # levels(df.Country)
    :Age,
    :Salary)
y_imputed = select(df_imputed, :Purchased)

## Feature Encoding
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

## Split Data To Train & Test Sets
(Xtrain, Xtest), (ytrain, ytest) = partition((X, y), .8, rng=123, multi=true);

## Standardizer
sc = Standardizer()
mach_age = machine(sc, Xtrain.Age) |> fit! 
Xtrain.Age = MLJ.transform(mach_age, Xtrain.Age) 
Xtest.Age = MLJ.transform(mach_age, Xtest.Age) 
mach_salary = machine(sc, Xtrain.Salary) |> fit! 
Xtrain.Salary = MLJ.transform(mach_salary, Xtrain.Salary) 
Xtest.Salary = MLJ.transform(mach_salary, Xtest.Salary) 

#=
vscodedisplay(Xtrain), vscodedisplay(Xtest)
vscodedisplay(ytrain), vscodedisplay(ytest)
=#
