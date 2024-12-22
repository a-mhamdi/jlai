### A Pluto.jl notebook ###
# v0.20.4

#> [frontmatter]
#> title = "Binary Classifier using ANN"
#> tags = ["ann", "flux", "julialang"]
#> date = "2024-12-21"
#> description = "Customer Churn Modelling"
#> 
#>     [[frontmatter.author]]
#>     name = "A. Mhamdi"
#>     url = "https://a-mhamdi.github.io/jlai/"

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 122e1c87-b4c3-4ca8-882c-a9d352ff19d9
begin
	cd(@__DIR__)
	using Pkg
	Pkg.activate("..")
end

# ╔═╡ 3caf8f15-5852-4108-9636-74988155197d
using CSV, DataFrames

# ╔═╡ f7f0914f-c875-4fae-9c68-06d7d860c7f1
using MLJ

# ╔═╡ 1cfb96eb-9b3a-4c17-af86-1cd8f92c45cb
using Flux

# ╔═╡ d0a6e9c5-1a39-4edc-bb67-56b95d045052
using PlutoUI

# ╔═╡ e813d51a-1dbc-497d-a4b6-1f396fd59489
using ProgressMeter

# ╔═╡ 4aacfa4d-22ef-4360-9f1b-5aa9712a7aa1
using Plots; theme(:dracula)

# ╔═╡ 891eda37-95d5-46dd-a642-79e68e7bb322
md"# BINARY CLASSIFIER USING ANN _(CHURN MODELLING)_"

# ╔═╡ 4f009769-c8f7-448d-83e5-ec54ff54b999
md"
```julia
versioninfo() -> v\"1.11.2\"
```
"

# ╔═╡ 5b6cb1fd-b25b-4803-9a90-56fd4d9167e3
md"Import the required librairies"

# ╔═╡ dfc23226-03a8-485a-a43b-aa88e235f3e7
md"Hyperparameters tuning"

# ╔═╡ 486ac35e-5a6e-4f7d-8bfd-3d00782eeffb
md"Load data from csv file"

# ╔═╡ 0d137d2c-db4f-4ec1-8290-9f621f956ee4
df = CSV.read("../../Datasets/Churn_Modelling.csv", DataFrame)

# ╔═╡ 37665a62-4c7b-4cb3-a353-3a436de3d8a8
md"Choose the target vector `y`"

# ╔═╡ 601b31ca-be6f-4b7d-bbb3-72c5dc4760f9
begin
	ydf = select(df, :Exited)
	# coerce!(ydf, :Exited => OrderedFactor)
end

# ╔═╡ 27c7a2c0-ebd7-41bb-8a89-8a5d2c8d1300
y_ = ydf.Exited

# ╔═╡ 7a52ca66-f742-4a24-9aa9-3f9e4c27d50f
md"Specify the features matrix `X`"

# ╔═╡ 2e8e8d45-1880-44f3-ad32-190d8c75d653
begin
	Xdf = select(df, Not([:RowNumber, :CustomerId, :Surname, :Exited]))
	coerce!(Xdf,
    :Geography => Multiclass,
    :Gender => Multiclass
	)
end

# ╔═╡ f1af0ba2-2a0f-45dd-9495-2ca9aab8a40f
md"Onehotencoding of multiclass variables"

# ╔═╡ e9a76936-545c-4683-b29f-c32ff255eb8c
ce = ContinuousEncoder(drop_last=true)

# ╔═╡ 9e109da1-41bf-4b8e-8c65-2d67d65e55fa
Xdf_ = machine(ce, Xdf) |> fit! |> MLJ.transform

# ╔═╡ 78fde993-c8db-4aad-a9a5-048957cc9366
md"Features scaling"

# ╔═╡ a928768c-83c5-46ff-9dc2-f8c0e9ce6828
sc = Standardizer()

# ╔═╡ 31025fb1-261d-4ac4-91c7-b8e931fb05cd
Xdf_m = machine(sc, Xdf_) |> fit! |> MLJ.transform

# ╔═╡ 626457c2-c4a2-438d-9d2b-bdb9ada0f429
md"Extract only the values for `X`, i.e, rm the headers."

# ╔═╡ 9797655e-803b-42ee-b393-67be988876ff
n, m = size(Xdf_m)

# ╔═╡ d01d463e-0bb8-47c1-8279-b7925f83c5e8
begin
	X_ = Array{Float32, 2}(undef, (n, m));
	for i in 1:m
	    X_[:, i] = Xdf_m[!, i];
	end
end

# ╔═╡ 57128bbf-704e-4879-94ce-acd478afe8b4
md"Design the architecture of the classifier, denoted hereafter by `clf`"

# ╔═╡ e2e6cc50-d02c-4bc9-8095-479207efa7bb
mdl = Chain(
            Dense( 11 => 8, relu ),
            Dense(  8 => 8, relu ),
            Dense(  8 => 8, relu ),
            Dense(  8 => 1 )
            )

# ╔═╡ 9f737b1c-fe5a-4ec9-a47e-a05c44c15a35
md"Permute dims: ROW => features and COL => observation"

# ╔═╡ b0bdb22e-8196-4046-ad6a-317d1ef2d879
X = permutedims(X_)

# ╔═╡ 567a79d9-a0a3-4142-864e-207d3f011895
y = permutedims(y_)

# ╔═╡ e2df135a-8ce2-462a-8a25-402de6c5cd26
md"Optimizers and data loader"

# ╔═╡ 55ecb9c4-8339-4690-84da-25407fdf4219
@bind η Slider(.0001:0.01:.1, default=.001)

# ╔═╡ 2dddfc11-9907-4e5a-b949-ef1f4e27dfac
opt = Flux.Adam(η);

# ╔═╡ 090e9957-7f97-49f7-89b5-efe39b8f94f3
state = Flux.setup(opt, mdl);

# ╔═╡ 19752951-1314-4df4-98d2-407d4a8c6e72
@bind batchsize Slider(2:8:128, default=32)

# ╔═╡ 1b2c6fe5-ec88-4a54-88d0-f3bde759d8da
loader = Flux.DataLoader((X, y); batchsize=batchsize, shuffle=true);

# ╔═╡ 348b2e02-a9c4-4ac7-b34a-641a218ed69d
md"**Training phase**"

# ╔═╡ f83759e2-eed2-46cc-95fc-c974e6af1f04
@bind epochs Slider(1:8:128, default=64)

# ╔═╡ 1d0f0991-6709-4d1b-b316-3a4cf52cfdb7
begin
	vec_loss = []
	@showprogress for epoch in 1:epochs
	    for (mb_X, mb_y) in loader
	        loss, grads = Flux.withgradient(mdl) do m
	            mb_ŷ = m(mb_X);
				Flux.logitbinarycrossentropy(mb_ŷ, mb_y);
	        end
	        Flux.update!(state, mdl, grads[1]); # Upd `W` and `b`
	        push!(vec_loss, loss); # Log `loss` to the vector `vec_loss`
	    end
	end
end

# ╔═╡ 81f472c9-d4ee-41d3-8c26-bcb9a2d20a35
md"Plot the loss vector `vec_loss`"

# ╔═╡ e1931c59-6355-49da-95bc-4bcb761f5554
plot(vec_loss, label="Loss")

# ╔═╡ 6f5b2f2a-109d-46d1-8b29-0271e86d3a1e
extrema(vec_loss)

# ╔═╡ f567f2ec-225c-4ab2-a9d7-e0833b972868
md"**Some metrics**"

# ╔═╡ 710d5efb-3c20-4655-a3cd-2490a80133c1
ŷ_ = mdl(X) |> σ;

# ╔═╡ 1da45b0e-c430-4f35-8a6a-2a6a2bf0afc6
ŷ = (ŷ_ .≥ .5);

# ╔═╡ c8d291ea-8505-4304-9bb5-6177f01a5c43
md"_Basic way to compute the accuracy_"

# ╔═╡ 086e0922-608a-4449-86e4-e86b32d20f41
accuracy = mean( ŷ .== y )

# ╔═╡ b71d3fc7-29be-49f9-994a-7371d03e81fc
md"_Confusion Matrix_"

# ╔═╡ de87da99-5d60-4b60-b1bd-c89d18ba6bbd
displayed_cm = MLJ.ConfusionMatrix(levels=[0, 1])(ŷ, y)

# ╔═╡ b578d4cb-eca6-4098-91c9-148ab6dbf52e
cm = ConfusionMatrices.matrix(displayed_cm)

# ╔═╡ 6cff22b5-1e50-4d8b-ab3c-1b92ba1063c6
md"_Classification metrics_"

# ╔═╡ a9ba36ff-0321-4cb9-a7af-0cbef12554ce
TP, TN, FP, FN = cm[2, 2], cm[1, 1], cm[2, 1], cm[1, 2];

# ╔═╡ e3487d37-edb3-46a2-909f-4e294626bae1
md"_Accuracy_"

# ╔═╡ 72c1d033-42a9-4ad4-814b-4d5c1d0d375b
accuracy_ = (TP+TN)/(TP+TN+FP+FN)

# ╔═╡ 0bb0694a-d4e2-4c15-a24d-d073a9bc514c
MLJ.accuracy(ŷ, y)

# ╔═╡ 0fb71566-73ee-4096-98b1-bfb7917f494d
md"_True Negative Rate_"

# ╔═╡ ba2efbb6-0029-4fee-b216-d537bd32b60e
true_negative_rate_ = TN/(TN+FP)

# ╔═╡ f45cc9db-03aa-46e4-be87-1af83a3b958d
MLJ.true_negative_rate(ŷ, y)

# ╔═╡ 862a1e4c-0e68-468d-9149-94158926285a
md"_True Positive Rate_"

# ╔═╡ 5053c71a-9236-4723-8dd3-d2fe2180c357
true_positive_rate_ = TP/(TP+FN)

# ╔═╡ 8883aedb-7e8e-482d-9225-5ba3fd6794b1
MLJ.true_positive_rate(ŷ, y)

# ╔═╡ 25b29473-507a-4d7f-9a5e-8267e7acbdb6
md"_f1-score_"

# ╔═╡ f46fcd55-6630-4ac3-b86b-316db204bbdf
begin
	precision_ = TP/(TP+FP)
	recall_ = TP/(TP+FN)
	f1score_ = 2/(1/precision_ + 1/recall_)
end

# ╔═╡ 9748255a-b13f-4cfe-b9a7-400a15ead1f5
MLJ.f1score(ŷ, y)

# ╔═╡ 90100f84-e006-4915-9e82-c849c3d91351
html"""
<style>
	main {
		margin: 0 auto;
		max-width: 2000px;
    	padding-left: max(160px, 10%);
    	padding-right: max(160px, 10%);
	}
</style>
"""

# ╔═╡ Cell order:
# ╠═891eda37-95d5-46dd-a642-79e68e7bb322
# ╠═4f009769-c8f7-448d-83e5-ec54ff54b999
# ╠═122e1c87-b4c3-4ca8-882c-a9d352ff19d9
# ╠═5b6cb1fd-b25b-4803-9a90-56fd4d9167e3
# ╠═3caf8f15-5852-4108-9636-74988155197d
# ╠═f7f0914f-c875-4fae-9c68-06d7d860c7f1
# ╠═1cfb96eb-9b3a-4c17-af86-1cd8f92c45cb
# ╠═dfc23226-03a8-485a-a43b-aa88e235f3e7
# ╠═486ac35e-5a6e-4f7d-8bfd-3d00782eeffb
# ╠═0d137d2c-db4f-4ec1-8290-9f621f956ee4
# ╠═37665a62-4c7b-4cb3-a353-3a436de3d8a8
# ╠═601b31ca-be6f-4b7d-bbb3-72c5dc4760f9
# ╠═27c7a2c0-ebd7-41bb-8a89-8a5d2c8d1300
# ╠═7a52ca66-f742-4a24-9aa9-3f9e4c27d50f
# ╠═2e8e8d45-1880-44f3-ad32-190d8c75d653
# ╠═f1af0ba2-2a0f-45dd-9495-2ca9aab8a40f
# ╠═e9a76936-545c-4683-b29f-c32ff255eb8c
# ╠═9e109da1-41bf-4b8e-8c65-2d67d65e55fa
# ╠═78fde993-c8db-4aad-a9a5-048957cc9366
# ╠═a928768c-83c5-46ff-9dc2-f8c0e9ce6828
# ╠═31025fb1-261d-4ac4-91c7-b8e931fb05cd
# ╠═626457c2-c4a2-438d-9d2b-bdb9ada0f429
# ╠═9797655e-803b-42ee-b393-67be988876ff
# ╠═d01d463e-0bb8-47c1-8279-b7925f83c5e8
# ╠═57128bbf-704e-4879-94ce-acd478afe8b4
# ╠═e2e6cc50-d02c-4bc9-8095-479207efa7bb
# ╠═9f737b1c-fe5a-4ec9-a47e-a05c44c15a35
# ╠═b0bdb22e-8196-4046-ad6a-317d1ef2d879
# ╠═567a79d9-a0a3-4142-864e-207d3f011895
# ╠═e2df135a-8ce2-462a-8a25-402de6c5cd26
# ╠═d0a6e9c5-1a39-4edc-bb67-56b95d045052
# ╠═55ecb9c4-8339-4690-84da-25407fdf4219
# ╠═2dddfc11-9907-4e5a-b949-ef1f4e27dfac
# ╠═090e9957-7f97-49f7-89b5-efe39b8f94f3
# ╠═19752951-1314-4df4-98d2-407d4a8c6e72
# ╠═1b2c6fe5-ec88-4a54-88d0-f3bde759d8da
# ╠═348b2e02-a9c4-4ac7-b34a-641a218ed69d
# ╠═e813d51a-1dbc-497d-a4b6-1f396fd59489
# ╠═f83759e2-eed2-46cc-95fc-c974e6af1f04
# ╠═1d0f0991-6709-4d1b-b316-3a4cf52cfdb7
# ╠═81f472c9-d4ee-41d3-8c26-bcb9a2d20a35
# ╠═4aacfa4d-22ef-4360-9f1b-5aa9712a7aa1
# ╠═e1931c59-6355-49da-95bc-4bcb761f5554
# ╠═6f5b2f2a-109d-46d1-8b29-0271e86d3a1e
# ╠═f567f2ec-225c-4ab2-a9d7-e0833b972868
# ╠═710d5efb-3c20-4655-a3cd-2490a80133c1
# ╠═1da45b0e-c430-4f35-8a6a-2a6a2bf0afc6
# ╠═c8d291ea-8505-4304-9bb5-6177f01a5c43
# ╠═086e0922-608a-4449-86e4-e86b32d20f41
# ╠═b71d3fc7-29be-49f9-994a-7371d03e81fc
# ╠═de87da99-5d60-4b60-b1bd-c89d18ba6bbd
# ╠═b578d4cb-eca6-4098-91c9-148ab6dbf52e
# ╠═6cff22b5-1e50-4d8b-ab3c-1b92ba1063c6
# ╠═a9ba36ff-0321-4cb9-a7af-0cbef12554ce
# ╠═e3487d37-edb3-46a2-909f-4e294626bae1
# ╠═72c1d033-42a9-4ad4-814b-4d5c1d0d375b
# ╠═0bb0694a-d4e2-4c15-a24d-d073a9bc514c
# ╠═0fb71566-73ee-4096-98b1-bfb7917f494d
# ╠═ba2efbb6-0029-4fee-b216-d537bd32b60e
# ╠═f45cc9db-03aa-46e4-be87-1af83a3b958d
# ╠═862a1e4c-0e68-468d-9149-94158926285a
# ╠═5053c71a-9236-4723-8dd3-d2fe2180c357
# ╠═8883aedb-7e8e-482d-9225-5ba3fd6794b1
# ╠═25b29473-507a-4d7f-9a5e-8267e7acbdb6
# ╠═f46fcd55-6630-4ac3-b86b-316db204bbdf
# ╠═9748255a-b13f-4cfe-b9a7-400a15ead1f5
# ╟─90100f84-e006-4915-9e82-c849c3d91351
