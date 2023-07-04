#####################################################
#= BINARY CLASSIFIER USING ANN _(CHURN MODELLING)_ =#
#####################################################

using Markdown

md"Import the required librairies"
using CSV, DataFrames
using MLJ
using Flux

md"Hyperparameters tuning"
η, epochs, batchsize = .001, 1_000, 32
md"Load data for csv file"
df = CSV.read("./Datasets/Churn_Modelling.csv", DataFrame)
md"Choose the target vector `y`"
ydf = select(df, :Exited)
y = ydf.Exited
md"Specify the features matrix `X`"
Xdf = select(df, Not([:RowNumber, :CustomerId, :Surname, :Exited]))
coerce!(Xdf,
    :Geography => Multiclass,
    :Gender => Multiclass
)
md"Onehotencoding of multiclass variables"
ce = ContinuousEncoder()
Xdf = machine(ce, Xdf) |> fit! |> MLJ.transform
md"Features scaling"
sc = Standardizer()
Xdf = machine(sc, Xdf) |> fit! |> MLJ.transform
md"Extract only the values for `X`, i.e, rm the headers."
n, m = size(Xdf)
X = Array{Float64, 2}(undef, (n, m));
for i in 1:m
    X[:, i] = Xdf[!, i];
end
md"Design the architecture of the classifier, denoted hereafter by `clf`"
clf = Chain(
            Dense( 13 => 8, relu),
            Dense( 8  => 8, relu),
            Dense( 8  => 4, relu),
            Dense( 4  => 1, σ)
            )
md"Permute dims: ROW => features and COL => observation"
X = permutedims(X)
y = permutedims(y)
md"Optimizers and data loader"
optim_state = Flux.setup(Adam(η), clf);
loader = Flux.Data.DataLoader((X, y); batchsize=batchsize, shuffle=true);
vec_loss = []
md"**Training phase**"
using ProgressMeter
@showprogress for _ in 1:epochs
    for (X, y) in loader
        loss, grads = Flux.withgradient(clf) do mdl
            ŷ = mdl(X);
            Flux.logitbinarycrossentropy(ŷ, y);
        end
        Flux.update!(optim_state, clf, grads[1]); # Upd `W` and `b`
        push!(vec_loss, loss); # Log `loss` to the vector `vec_loss`
    end
end
md"Plot the loss vector `vec_loss`"
using Plots
plot(vec_loss, label="Loss")
extrema(vec_loss)
md"Some metrics"
ŷ = clf(X);
ŷ = (ŷ .≥ .5);
# y = (y .≥ .5);
md"Basic way to compute the accuracy"
accuracy = sum(mean( ŷ .== y))
md"Confusion Matrix"
cm = MLJ.ConfusionMatrix()(ŷ, y)
TP, TN, FP, FN = cm[2, 2], cm[1, 1], cm[2, 1], cm[1, 2];
accuracy_ = (TP+TN)/(TP+TN+FP+FN) # MLJ.accuracy(cm)
precision_ = TP/(TP+FP) # MLJ.precision(cm)
recall_ = TP/(TP+FN) # MLJ.recall(cm)
f1score_ = 2/(1/precision_ + 1/recall_) # MLJ.f1score(cm)
