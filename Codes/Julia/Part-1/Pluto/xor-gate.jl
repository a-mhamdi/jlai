### A Pluto.jl notebook ###
# v0.20.4

#> [frontmatter]
#> title = "XOR GATE"
#> date = "2024-12-20"
#> tags = ["xor", "ann", "julialan", "pluto"]
#> description = "Build a typical `XOR` gate using artificial neural net."
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

# ╔═╡ aefe9c9f-52a9-49c1-95a5-decc25fd1317
begin
	cd(@__DIR__)
	using Pkg
	Pkg.activate("..")
end

# ╔═╡ 9b0100b8-695c-4699-a0a5-8e1672aa8d7d
using Flux

# ╔═╡ 60bdb844-225d-48e9-9cf9-4886837a5ad6
using Plots; theme(:dracula)

# ╔═╡ 06cf692e-a3ec-4bbe-a8d6-d32b8db5063e
using PlutoUI

# ╔═╡ 1aa84a54-7ad0-4154-a880-1326c34c9070
using ProgressMeter

# ╔═╡ 3d94a9b8-c4ac-446f-b3c4-fec5be847f18
md"# XOR GATE"

# ╔═╡ 067f56ad-4a5a-4272-aaa3-4a4bc00fcb4c
md"
```julia
versioninfo() -> v\"1.11.2\"
```
"

# ╔═╡ 7e20991f-a74a-49d4-bec6-2a356b38f0cd
md"Create the dataset for an \"XOR\" problem"

# ╔═╡ 754381f5-9920-47b1-ad5f-dad3a86ed038
begin
	X = rand(Float32, 2, 1_024);
	first(X, 5)
end

# ╔═╡ de74bfca-11df-476a-bee5-b7d24a363c55
begin
	y = [xor(col[1]>.5, col[2]>.5) for col in eachcol(X)]
	first(y, 5)
end

# ╔═╡ b043d570-294b-4d1d-8774-458ca9656d32
md"Scatter plot of `X`"

# ╔═╡ 5deb2c54-2bb6-4f52-83b1-b1b0dd1c1592
sc = scatter(X[1,:], X[2,:], group=y; labels=["False" "True"])

# ╔═╡ 830c4d67-8e16-457e-9a9c-89fbbeb5aa02
loader = Flux.DataLoader((X, y), batchsize=32, shuffle=true)

# ╔═╡ f08088c3-f9e9-4cdf-b919-d8fca57cdfef
md"`mdl` is the model to be built"

# ╔═╡ 0acd627b-cf06-464b-98e1-acf65fbe867e
mdl = Chain(Dense( 2 => 4, tanh ),
			Dense( 4 => 4, tanh ),
            Dense( 4 => 1, σ ),
            )

# ╔═╡ d7522f88-d66e-4ea4-9159-36c4f307f6ac
md"Raw output before training"

# ╔═╡ 023358b6-b112-4adc-b1b3-96fc378bcf5f
y_raw = mdl(X)

# ╔═╡ 9f178f7f-a89e-4626-8065-dc68e2219f9c
md"`opt` designates the optimizer"

# ╔═╡ a7a2cf85-1a98-4408-ae68-faa3ad078474
@bind η Slider(logrange(.001,1,length=10), default=.003)

# ╔═╡ 185b5a65-904a-402d-9c4d-10cb917704f0
opt = Adam(η)

# ╔═╡ 63c74e5a-0338-489a-ab1f-f09e93f03c07
md"`state` contains all trainable parameters"

# ╔═╡ 7b9cf9b7-18bc-4b27-8566-07cad5b75922
state = Flux.setup(opt, mdl)

# ╔═╡ ec19b265-dfa3-4807-a397-e6f6bf778b2e
md"**TRAINING PHASE**"

# ╔═╡ f4d384bb-9dfe-4839-99ca-60af08c1f475
@bind epochs Slider(1:2:16, default=4)

# ╔═╡ fec114ea-9fd1-44a1-8512-854eef73cc1f
begin
	vec_loss = []
	@showprogress for epoch in 1:epochs
	    for (Features, target) in loader
			# Begin a gradient context session
	        loss, grads = Flux.withgradient(mdl) do m
	            # Evaluate model:
	            target_hat = m(Features) |> vec # loss function expects size(ŷ) = (1, :) to match size(y) = (:,)
				# Evaluate loss:
	            Flux.binarycrossentropy(target_hat, target)
	        end
	        Flux.update!(state, mdl, grads[1])
	        push!(vec_loss, loss)  # Log `loss` to `losses` vector `vec_loss`
	    end
	end
end

# ╔═╡ 8b63c359-ce39-47d2-be3c-71c5d55c7fb2
md"Predicted output after being trained"

# ╔═╡ 8b18e2d2-a501-4e44-8379-b7512a4b8149
y_hat = mdl(X)

# ╔═╡ 356304bb-08de-4372-bdf1-435924976339
y_pred = (y_hat[1, :] .> .5)

# ╔═╡ a418763f-6372-4dd5-8626-1c41c0788690
md"Accuracy: How much we got right over all cases _(i.e., (TP+TN)/(TP+TN+FP+FN))_"

# ╔═╡ 7ddefa63-ea75-48d2-bfb9-efd2f0214e54
accuracy = Flux.Statistics.mean( (y_pred .> .5) .== y )

# ╔═╡ fe94ffc2-7089-4235-b5ae-f5b8d59e6ecf
md"Plot loss vs. iteration"

# ╔═╡ a25aa8e5-7dbe-4051-aeac-927137cd6fb0
plot(vec_loss; xaxis=(:log10, "Iteration"), yaxis="Loss", label="Per Batch")

# ╔═╡ 0099fef3-3f72-4beb-8e9a-bdd7cbf4595e
sc1 = scatter(X[1,:], X[2,:], group=y; title="TRUTH", labels=["False" "True"])

# ╔═╡ 7f82370b-f725-4b20-8e4f-f13bc7f052dd
sc2 = scatter(X[1,:], X[2,:], zcolor=y_raw[1,:]; title="BEFORE", label=:none, clims=(0,1))

# ╔═╡ 189fbddb-1208-40e3-83d7-afe29b7c4dc1
sc3 = scatter(X[1,:], X[2,:], group=y_pred; title="AFTER", labels=["False" "True"])

# ╔═╡ 65404f3d-4dd5-43bf-a90f-71a34182cfa8
md"Plot of both ground truth and results after training"

# ╔═╡ 7491441d-19aa-49aa-a692-628d73a96611
plot(sc1, sc3, layout=(1,2), size=(512,512))

# ╔═╡ 6e6c6526-c067-4c66-8cf0-9ffe339ebf0a
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
# ╠═3d94a9b8-c4ac-446f-b3c4-fec5be847f18
# ╠═067f56ad-4a5a-4272-aaa3-4a4bc00fcb4c
# ╠═aefe9c9f-52a9-49c1-95a5-decc25fd1317
# ╠═9b0100b8-695c-4699-a0a5-8e1672aa8d7d
# ╠═7e20991f-a74a-49d4-bec6-2a356b38f0cd
# ╠═754381f5-9920-47b1-ad5f-dad3a86ed038
# ╠═de74bfca-11df-476a-bee5-b7d24a363c55
# ╠═b043d570-294b-4d1d-8774-458ca9656d32
# ╠═60bdb844-225d-48e9-9cf9-4886837a5ad6
# ╠═5deb2c54-2bb6-4f52-83b1-b1b0dd1c1592
# ╠═830c4d67-8e16-457e-9a9c-89fbbeb5aa02
# ╠═f08088c3-f9e9-4cdf-b919-d8fca57cdfef
# ╠═0acd627b-cf06-464b-98e1-acf65fbe867e
# ╠═d7522f88-d66e-4ea4-9159-36c4f307f6ac
# ╠═023358b6-b112-4adc-b1b3-96fc378bcf5f
# ╠═9f178f7f-a89e-4626-8065-dc68e2219f9c
# ╠═06cf692e-a3ec-4bbe-a8d6-d32b8db5063e
# ╠═a7a2cf85-1a98-4408-ae68-faa3ad078474
# ╠═185b5a65-904a-402d-9c4d-10cb917704f0
# ╠═63c74e5a-0338-489a-ab1f-f09e93f03c07
# ╠═7b9cf9b7-18bc-4b27-8566-07cad5b75922
# ╠═ec19b265-dfa3-4807-a397-e6f6bf778b2e
# ╠═f4d384bb-9dfe-4839-99ca-60af08c1f475
# ╠═1aa84a54-7ad0-4154-a880-1326c34c9070
# ╠═fec114ea-9fd1-44a1-8512-854eef73cc1f
# ╠═8b63c359-ce39-47d2-be3c-71c5d55c7fb2
# ╠═8b18e2d2-a501-4e44-8379-b7512a4b8149
# ╠═356304bb-08de-4372-bdf1-435924976339
# ╠═a418763f-6372-4dd5-8626-1c41c0788690
# ╠═7ddefa63-ea75-48d2-bfb9-efd2f0214e54
# ╠═fe94ffc2-7089-4235-b5ae-f5b8d59e6ecf
# ╠═a25aa8e5-7dbe-4051-aeac-927137cd6fb0
# ╠═0099fef3-3f72-4beb-8e9a-bdd7cbf4595e
# ╠═7f82370b-f725-4b20-8e4f-f13bc7f052dd
# ╠═189fbddb-1208-40e3-83d7-afe29b7c4dc1
# ╠═65404f3d-4dd5-43bf-a90f-71a34182cfa8
# ╠═7491441d-19aa-49aa-a692-628d73a96611
# ╟─6e6c6526-c067-4c66-8cf0-9ffe339ebf0a
