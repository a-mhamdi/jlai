### A Pluto.jl notebook ###
# v0.20.5

using Markdown
using InteractiveUtils

# ╔═╡ 164fc0e9-5232-4248-afd1-9fd9ca91debc
begin
	cd(@__DIR__)
	import Pkg; Pkg.activate(".."); Pkg.status()
end

# ╔═╡ a96990ee-ecb2-4034-bca8-51687231b8ac
using MLJ

# ╔═╡ 66e672d8-1335-11f0-0279-1fd2a12c76ba
md"# NAIVE BAYES"

# ╔═╡ ace3a93e-bfae-4672-9490-39944dda0902
versioninfo()

# ╔═╡ 7c4a14b1-907e-4892-b752-8383f9d29fb1
import Languages, TextAnalysis

# ╔═╡ b5412a4f-4803-4c9b-8b4c-c0cc434e1a46
reviews = [
    ["I am furious! This is unacceptable.", "Negative"],
    ["Could you assist me with this problem? I'd appreciate it.", "Positive"],
    ["My coworker ignored me again. I hate this place.", "Negative"],
    ["The manager offered support. That was kind of her.", "Positive"],
    ["Thanks for your patience. You’re very understanding.", "Positive"],
    ["This is a disaster! Everything is going wrong.", "Negative"],
    ["The team handled the project well. Good job!", "Positive"],
    ["I’m so annoyed. Nobody ever helps me.", "Negative"],
    ["She gave me great advice. I’m grateful.", "Positive"],
    ["Why does this always happen? I’m done with this.", "Negative"],
    ["Your guidance was excellent. Thank you!", "Positive"],
    ["He yelled at me for no reason. So unfair.", "Negative"],
    ["The meeting went smoothly. No issues at all.", "Neutral"],
    ["I’m really happy with the results. Well done!", "Positive"],
    ["This company is terrible. I regret joining.", "Negative"],
]

# ╔═╡ fe9d420a-1333-40e3-b2ac-7df0e1f600f0
train, test = partition(1:length(reviews), .8; shuffle=true)

# ╔═╡ 1131fa61-fc97-4f20-88da-16024dac1883
sentiment = [review[2] for review in reviews]

# ╔═╡ f3373580-625b-4771-ae5c-bf671862e125
first(sentiment, 5)

# ╔═╡ a49f02a6-50cc-41c0-bc23-82b7305bc45f
y = coerce(sentiment, OrderedFactor)

# ╔═╡ 6bcfce8d-b311-4d8a-a9a8-bd266b609deb
scitype(y) <: AbstractVector{<:OrderedFactor}

# ╔═╡ c5de24b3-d02e-42c4-8246-abd92ffb2fe5
tokens = [TextAnalysis.tokenize(Languages.English(), review[1]) for review in reviews]

# ╔═╡ cbd5e85b-a8d5-4799-9121-283ae88cbee4
CountTransformer = @load CountTransformer pkg=MLJText

# ╔═╡ afdb2039-4ed5-423c-a8e0-fa704b6a457a
ct = CountTransformer()

# ╔═╡ 8d73a57c-bf77-4509-b170-0df96ed679fd
mach_ct = machine(ct, tokens) |> fit!

# ╔═╡ 707c6db5-3db5-4480-86f8-13c01047aaed
X = MLJ.transform(mach_ct, tokens)

# ╔═╡ 3faeb20d-8d24-4862-9adf-36c5ff217714
NaiveBayes = @load MultinomialNBClassifier pkg=NaiveBayes

# ╔═╡ c11cdb5b-15a2-4d2e-a40c-29f510fa2466
nb = NaiveBayes()

# ╔═╡ 90c721c8-5818-409b-98b1-7e2a436a0173
mach_nb = machine(nb, X, y)

# ╔═╡ b74172a5-ced0-4fe2-a51a-d9f1fb8a6c63
fit!(mach_nb, rows=train)

# ╔═╡ 2d09ec07-35ae-4337-aad0-7f16ba46b51e
md"Probabilistic predictions"

# ╔═╡ 84cc37c0-26f3-4b4b-a204-f83746a18f3a
y_prob = predict(mach_nb, rows=test)

# ╔═╡ 83a765d5-f92f-45d6-990c-886a3dfceeb1
pdf.(y_prob, "Positive")

# ╔═╡ 3366781f-b42e-46cd-aa3a-f5503c723618
log_loss(y_prob, y[test])

# ╔═╡ 6353acb0-04be-4084-be4a-216881b6c762
ŷ = mode.(y_prob)

# ╔═╡ d5400b11-d968-4560-95ac-57d5234c3f01
ŷ .== y[test] 

# ╔═╡ Cell order:
# ╠═66e672d8-1335-11f0-0279-1fd2a12c76ba
# ╠═ace3a93e-bfae-4672-9490-39944dda0902
# ╠═164fc0e9-5232-4248-afd1-9fd9ca91debc
# ╠═a96990ee-ecb2-4034-bca8-51687231b8ac
# ╠═7c4a14b1-907e-4892-b752-8383f9d29fb1
# ╠═b5412a4f-4803-4c9b-8b4c-c0cc434e1a46
# ╠═fe9d420a-1333-40e3-b2ac-7df0e1f600f0
# ╠═1131fa61-fc97-4f20-88da-16024dac1883
# ╠═f3373580-625b-4771-ae5c-bf671862e125
# ╠═a49f02a6-50cc-41c0-bc23-82b7305bc45f
# ╠═6bcfce8d-b311-4d8a-a9a8-bd266b609deb
# ╠═c5de24b3-d02e-42c4-8246-abd92ffb2fe5
# ╠═cbd5e85b-a8d5-4799-9121-283ae88cbee4
# ╠═afdb2039-4ed5-423c-a8e0-fa704b6a457a
# ╠═8d73a57c-bf77-4509-b170-0df96ed679fd
# ╠═707c6db5-3db5-4480-86f8-13c01047aaed
# ╠═3faeb20d-8d24-4862-9adf-36c5ff217714
# ╠═c11cdb5b-15a2-4d2e-a40c-29f510fa2466
# ╠═90c721c8-5818-409b-98b1-7e2a436a0173
# ╠═b74172a5-ced0-4fe2-a51a-d9f1fb8a6c63
# ╠═2d09ec07-35ae-4337-aad0-7f16ba46b51e
# ╠═84cc37c0-26f3-4b4b-a204-f83746a18f3a
# ╠═83a765d5-f92f-45d6-990c-886a3dfceeb1
# ╠═3366781f-b42e-46cd-aa3a-f5503c723618
# ╠═6353acb0-04be-4084-be4a-216881b6c762
# ╠═d5400b11-d968-4560-95ac-57d5234c3f01
