### A Pluto.jl notebook ###
# v0.20.4

#> [frontmatter]
#> title = "Natural Language Processing"
#> date = "2024-12-20"
#> tags = ["julialang", "nlp"]
#> 
#>     [[frontmatter.author]]
#>     name = "A. Mhamdi"
#>     url = "https://a-mhamdi.github.io/jlai/#"

using Markdown
using InteractiveUtils

# ╔═╡ 5f3b5700-d705-477d-8f20-6d09ecb4a8b3
import Pkg; Pkg.activate(".")

# ╔═╡ bf1da4bc-48f4-4042-a7ab-e042bbd32d41
using TextAnalysis

# ╔═╡ c1fb388b-9c59-429b-b036-9bc4395ab5af
using Embeddings

# ╔═╡ fdd4f0b1-1cae-467a-9fbf-6fbbe1c35c1c
using LinearAlgebra

# ╔═╡ d8292e75-d298-4e8e-b7bb-acc848e01b4a
md"# NATURAL LANGUAGE PROCESSING"

# ╔═╡ 384c0666-27c6-4bc8-8e9d-c20d0fd3c919
versioninfo() # -> v"1.11.2"

# ╔═╡ f336e7e2-99ae-4b91-8f93-0607eede9773
txt = "The quick brown fox is jumping over the lazy dog" # Pangram [modif.]

# ╔═╡ 40388d9d-16db-4e3c-9384-bc4638ba3663
md"Create a `Corpus` using `txt`"

# ╔═╡ 4febd42e-b2c5-46e2-af27-3e81f825fdab
crps = Corpus([StringDocument(txt)])

# ╔═╡ f38b2eee-ee30-4a4f-8500-2c076000ff95
lexicon(crps)

# ╔═╡ beb39317-c83c-4c0e-ac1d-ab64cde4307d
update_lexicon!(crps)

# ╔═╡ 2c3d4f5f-a0fd-42e0-ba1d-ba390c6da171
lexicon(crps)

# ╔═╡ 59170576-05f3-4edc-b26b-8b48152a4477
lexical_frequency(crps, "fox")

# ╔═╡ 2388744f-24b6-4ac1-8673-0feb223bfe37
md"Create a `StringDocument` using `txt`"

# ╔═╡ 811cdd9b-7cc8-444b-a621-479f785af38c
sd = StringDocument(txt)

# ╔═╡ 75322fd2-a075-4c89-b219-95b86753457a
md"Get a smaller set of words `text(sd)`"

# ╔═╡ ce5adafd-e6d3-4571-a289-c31940e3ebfc
prepare!(sd, strip_articles | strip_numbers | strip_punctuation | strip_case | strip_whitespace)

# ╔═╡ 356963d7-9999-43bd-a21b-49b3056a172e
stem!(sd)

# ╔═╡ 560a06c5-9078-49ad-9da6-9838656a0948
md"Get the tokens of `sd`"

# ╔═╡ 5b336441-b1f6-4ce1-a979-64e363bbc5fc
the_tokens = tokens(sd)

# ╔═╡ 25388144-78af-4f7d-9f24-7c190141e61f
md"Get the stemmed tokens of `sd`"

# ╔═╡ 6c5b1442-6297-44cb-8f97-200572f5dfb9
stemmer = Stemmer("english")

# ╔═╡ 1cf0c275-c1cc-4453-a5b8-0cfcd541faf8
stemmed_tokens = stem(stemmer, the_tokens)

# ╔═╡ 8295d84f-8aa6-4ad9-be45-fcc9aa3ae087
println("Original tokens: ", the_tokens)

# ╔═╡ 7a155470-e751-4137-9c87-b047bc2d44c6
println("Stemmed tokens: ", stemmed_tokens)

# ╔═╡ e1566e3f-4c08-4485-b345-9cefda319e45
md"**Part-of-speech tags**"

#= 
Common POS tags:

JJ: Adjective
NN: Noun, singular or mass
NNS: Noun, plural
VB: Verb, base form
VBZ: Verb, 3rd person singular present
VBG: Verb, gerund or present participle
VBD: Verb, past tense
RB: Adverb
IN: Preposition or subordinating conjunction
DT: Determiner
PRP: Personal pronoun
CC: Coordinating conjunction
=#

#=
using TextModels
pos = PoSTagger()
pos(crps)
=#

# ╔═╡ 38639182-7aaa-44e7-b11f-60aebe6a54d8
md"**Word embeddings**"

# ╔═╡ 88d15a75-7d44-405c-a8e4-19c68708d0a5
embtab = load_embeddings(GloVe{:en}, max_vocab_size=5)

# ╔═╡ eb53e251-d8e4-4b66-86fd-279fbbabe7e4
embtab.vocab

# ╔═╡ d1f06dd1-2078-4b4e-a33a-40191b96a1bb
embtab.embeddings

# ╔═╡ eadd26c5-d4ea-4408-ab78-5bc1cce4cd12
glove = load_embeddings(GloVe{:en}, 3, max_vocab_size=10_000)

# ╔═╡ d77c50ba-84af-419e-b127-a2deb9a3859c
const word_to_index = Dict(word => ii for (ii,word) in enumerate(glove.vocab))

# ╔═╡ 2be17f28-55e6-43c2-bece-77702ad72923
function get_word_vector(word)
    idx = word_to_index[word]
    return glove.embeddings[:, idx]
end

# ╔═╡ 6e1668bb-43af-49be-9d9d-623a101dba03
function cosine_similarity(v1::Vector{Float32}, v2::Vector{Float32})
    return *(v1', v2) / *(norm(v1), norm(v2))
end

# ╔═╡ 937e7d17-34cc-4e59-b2bf-98cf9e153bb5
md"_e.g. - \"king\" - \"man\" + \"woman\" ≈ \"queen\"_"

# ╔═╡ 37ee5b81-3444-49ab-8732-530004b00035
king = get_word_vector("king")

# ╔═╡ 1f128c17-8da6-4752-939b-25e75e8a8725
queen = get_word_vector("queen")

# ╔═╡ 4cc8fb94-43a6-44cd-8754-bb1ef0190573
man = get_word_vector("man")

# ╔═╡ 9c71279a-6ebd-4c86-82e2-c0af535f33b9
woman = get_word_vector("woman")

# ╔═╡ accb5901-6b0b-41f6-8d7c-3dd5a33d571a
cosine_similarity(king - man + woman, queen)

# ╔═╡ 07d3ff60-b7e9-4d99-9a73-601e421e1b64
md"_e.g. - \"Madrid\" - \"Spain\" + \"France\" ≈ \"Paris\"_"

# ╔═╡ 6bb3a773-c5c4-4d06-a28e-03737ce02797
Madrid = get_word_vector("madrid")

# ╔═╡ 3b8f4450-4248-4b60-a61c-5411c311d5d6
Spain = get_word_vector("spain")

# ╔═╡ cb9b1c25-9407-4ada-98eb-2b2462479729
France = get_word_vector("france")

# ╔═╡ 17e527a4-b80f-43bd-94a4-3e0411bcc642
Paris = get_word_vector("paris")

# ╔═╡ dca54445-4704-4726-99ce-3f60431459ba
cosine_similarity(Madrid - Spain + France, Paris)

# ╔═╡ 74d57c25-34f0-4f60-9ba9-6d6b507b61f2
md"**Text classification**"

# ╔═╡ 65f6b972-a4fe-4b62-8dc8-80f564de8fbf
md"https://github.com/JuliaText/TextAnalysis.jl/blob/master/docs/src/classify.md"

# ╔═╡ 11fe766c-bb25-478a-bd94-e786f98d21aa
begin
	mdl = NaiveBayesClassifier([:legal, :financial])
	fit!(mdl, "this is financial doc", :financial)
	fit!(mdl, "this is legal doc", :legal)
end

# ╔═╡ 33272333-27b7-4ed1-918c-fac97e0e18fd
predict(mdl, "this should be predicted as a legal document")

# ╔═╡ 1653799c-175e-4024-b9a8-ca08446cdc79
md"**Semantic analysis**"

# ╔═╡ 9120c455-745e-498a-acc7-827da3f8c9ae
m = DocumentTermMatrix(crps)

# ╔═╡ 3a6c37d3-7ea9-4d5b-b486-e3c48be5e076
md"*Latent Semantic Analysis*"

# ╔═╡ 2ded08e4-b048-41e7-a02c-8733e02ef928
lsa(m)

# ╔═╡ c011919b-0378-4116-a085-21ee18231104
md"*Latent Dirichlet Allocation*"

# ╔═╡ 4202f7f4-499b-44df-8d79-4779609c93fe
k = 2              # number of topics

# ╔═╡ cddae5b7-8f84-4407-9a09-9f7d19aac071
iterations = 1000  # number of Gibbs sampling iterations

# ╔═╡ 46c868e8-23f8-4c22-9742-2126c5b7e15e
α = 0.1            # hyper parameter

# ╔═╡ e4ea104e-0bd9-4d54-a4f3-800ad6a3dde9
β  = 0.1           # hyper parameter

# ╔═╡ 92520517-1ba9-4436-9472-91699cb2b56c
ϕ, θ  = lda(m, k, iterations, α, β)

# ╔═╡ 1b94b44f-2c26-463e-bb24-2e8b895a2b5a
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
# ╠═d8292e75-d298-4e8e-b7bb-acc848e01b4a
# ╠═384c0666-27c6-4bc8-8e9d-c20d0fd3c919
# ╠═5f3b5700-d705-477d-8f20-6d09ecb4a8b3
# ╠═bf1da4bc-48f4-4042-a7ab-e042bbd32d41
# ╠═f336e7e2-99ae-4b91-8f93-0607eede9773
# ╠═40388d9d-16db-4e3c-9384-bc4638ba3663
# ╠═4febd42e-b2c5-46e2-af27-3e81f825fdab
# ╠═f38b2eee-ee30-4a4f-8500-2c076000ff95
# ╠═beb39317-c83c-4c0e-ac1d-ab64cde4307d
# ╠═2c3d4f5f-a0fd-42e0-ba1d-ba390c6da171
# ╠═59170576-05f3-4edc-b26b-8b48152a4477
# ╠═2388744f-24b6-4ac1-8673-0feb223bfe37
# ╠═811cdd9b-7cc8-444b-a621-479f785af38c
# ╠═75322fd2-a075-4c89-b219-95b86753457a
# ╠═ce5adafd-e6d3-4571-a289-c31940e3ebfc
# ╠═356963d7-9999-43bd-a21b-49b3056a172e
# ╠═560a06c5-9078-49ad-9da6-9838656a0948
# ╠═5b336441-b1f6-4ce1-a979-64e363bbc5fc
# ╠═25388144-78af-4f7d-9f24-7c190141e61f
# ╠═6c5b1442-6297-44cb-8f97-200572f5dfb9
# ╠═1cf0c275-c1cc-4453-a5b8-0cfcd541faf8
# ╠═8295d84f-8aa6-4ad9-be45-fcc9aa3ae087
# ╠═7a155470-e751-4137-9c87-b047bc2d44c6
# ╠═e1566e3f-4c08-4485-b345-9cefda319e45
# ╠═38639182-7aaa-44e7-b11f-60aebe6a54d8
# ╠═c1fb388b-9c59-429b-b036-9bc4395ab5af
# ╠═88d15a75-7d44-405c-a8e4-19c68708d0a5
# ╠═eb53e251-d8e4-4b66-86fd-279fbbabe7e4
# ╠═d1f06dd1-2078-4b4e-a33a-40191b96a1bb
# ╠═eadd26c5-d4ea-4408-ab78-5bc1cce4cd12
# ╠═d77c50ba-84af-419e-b127-a2deb9a3859c
# ╠═2be17f28-55e6-43c2-bece-77702ad72923
# ╠═fdd4f0b1-1cae-467a-9fbf-6fbbe1c35c1c
# ╠═6e1668bb-43af-49be-9d9d-623a101dba03
# ╠═937e7d17-34cc-4e59-b2bf-98cf9e153bb5
# ╠═37ee5b81-3444-49ab-8732-530004b00035
# ╠═1f128c17-8da6-4752-939b-25e75e8a8725
# ╠═4cc8fb94-43a6-44cd-8754-bb1ef0190573
# ╠═9c71279a-6ebd-4c86-82e2-c0af535f33b9
# ╠═accb5901-6b0b-41f6-8d7c-3dd5a33d571a
# ╠═07d3ff60-b7e9-4d99-9a73-601e421e1b64
# ╠═6bb3a773-c5c4-4d06-a28e-03737ce02797
# ╠═3b8f4450-4248-4b60-a61c-5411c311d5d6
# ╠═cb9b1c25-9407-4ada-98eb-2b2462479729
# ╠═17e527a4-b80f-43bd-94a4-3e0411bcc642
# ╠═dca54445-4704-4726-99ce-3f60431459ba
# ╠═74d57c25-34f0-4f60-9ba9-6d6b507b61f2
# ╠═65f6b972-a4fe-4b62-8dc8-80f564de8fbf
# ╠═11fe766c-bb25-478a-bd94-e786f98d21aa
# ╠═33272333-27b7-4ed1-918c-fac97e0e18fd
# ╠═1653799c-175e-4024-b9a8-ca08446cdc79
# ╠═9120c455-745e-498a-acc7-827da3f8c9ae
# ╠═3a6c37d3-7ea9-4d5b-b486-e3c48be5e076
# ╠═2ded08e4-b048-41e7-a02c-8733e02ef928
# ╠═c011919b-0378-4116-a085-21ee18231104
# ╠═4202f7f4-499b-44df-8d79-4779609c93fe
# ╠═cddae5b7-8f84-4407-9a09-9f7d19aac071
# ╠═46c868e8-23f8-4c22-9742-2126c5b7e15e
# ╠═e4ea104e-0bd9-4d54-a4f3-800ad6a3dde9
# ╠═92520517-1ba9-4436-9472-91699cb2b56c
# ╟─1b94b44f-2c26-463e-bb24-2e8b895a2b5a
