###########################################
# IMPLEMENTING GRADIENT DESCENT ALGORITHM #
###########################################

using Markdown
md"**NORMAL EQUATION**"

md"Features matrix"
X = [1 0; 1 25; 1 50; 1 75; 1 100]

md"Target vector"
y = [14, 38, 54, 76, 95]

md"Estimated θ"
θ = (X' * X) \ X' * y

md"**GRADIENT DESCENT**"
using Random
using Plots

md"*Stocastic Gradient Descent (SGD)*"
α, n, θ = 0.0003, 5, [10; .5]
J = []
for _ in 1:1_000 # RUN IT MULTIPLE TIMES TO CONVERGE
    for k in shuffle(1:5)
		cost = ( y[k] - X[k, :]' * θ )^2
		push!(J, cost);
		θ += α * (y[k] - X[k, :]' * θ) * X[k, :]
		println("θ is $(θ)")
    end
end

plot(J)

md"*Batch Gradient Descent*"
α, n, θ = 0.0003, 5, [10; .5]
J = []
for _ in 1:1_000
	ϵ = (y-X*θ)[:, 1]
	cost = (2*n)\ ϵ'*ϵ
	push!(J, cost)
	θ += n\α * sum((y - X * θ) .* X, dims=1)'
	println("θ is $(θ)")
end

plot(J)

md"TO DO: Add Regularization Term To Cost Function"