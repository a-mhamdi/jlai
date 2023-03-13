###########################################
# IMPLEMENTING GRADIENT DESCENT ALGORITHM #
###########################################

using Markdown
md"*NORMAL EQUATION*"

md"Matrix of features"
X = [1,0; 1,25; 1,50; 1,75; 1,100]

md"Target vector"
y = [14, 38, 54, 76, 95]

md"Estimated \theta"
\theta = (X'*X)\X'*y 

md"*GRADIENT DESCENT*"
md"Stocastic Gradient Descent"

\alpha = .3
\theta = [1; .5]
n = 5
for i in 1:5
	\theta = \theta + \alpha*( y[i] - X[i, :]' * \theta ) * X[i, :] 
	println("\theta is $(\theta))"
end

md"Batch Gradient Descent"
for _ in 1:1000
	\theta = \theta + \alpha/n*sum( ( y - X * \theta ) .* X , dims=1)
	J = sum( (y - X * \theta ) .^ 2 ) ./ (2*n)	 
end
