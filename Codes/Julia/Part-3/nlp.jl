#= NLP =#

using TextAnalysis

txt = "The quick brown fox is jumping over the lazy dog" # Pangram [modif.]

## Create a `Corpus` using `txt`
crps = Corpus([StringDocument(txt)])
lexicon(crps)
update_lexicon!(crps)
lexicon(crps)
lexical_frequency(crps, "fox")

## Create a `StringDocument` using `txt`
sd = StringDocument(txt)

prepare!(sd, strip_articles | strip_numbers | strip_punctuation | strip_case | strip_whitespace)
stem!(sd) # Get a smaller set of words `text(sd)`

the_tokens = tokens(sd) # Get the tokens

## Stemming
stemmer = Stemmer("english")
stemmed_tokens = stem(stemmer, the_tokens)

println("Original tokens: ", the_tokens)
println("Stemmed tokens: ", stemmed_tokens)

## Part-of-speech tags

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

using TextModels
pos = PoSTagger()
pos(crps)

## Word embeddings
using Embeddings
embtab = load_embeddings(GloVe{:en}, max_vocab_size=5)

embtab.vocab
embtab.embeddings

glove = load_embeddings(GloVe{:en}, 3, max_vocab_size=10_000)
const word_to_index = Dict(word => ii for (ii,word) in enumerate(glove.vocab))
function get_word_vector(word)
    idx = word_to_index[word]
    return glove.embeddings[:, idx]
end

using LinearAlgebra
function cosine_similarity(v1::Vector{Float32}, v2::Vector{Float32})
    return *(v1', v2) / *(norm(v1), norm(v2))
end

### e.g. - "king" - "man" + "woman" ≈ "queen"
king = get_word_vector("king")
queen = get_word_vector("queen")
man = get_word_vector("man")
woman = get_word_vector("woman")

cosine_similarity(king - man + woman, queen)

### e.g. - "Madrid" - "Spain" + "France" ≈ "Paris"
Madrid = get_word_vector("madrid")
Spain = get_word_vector("spain")
France = get_word_vector("france")
Paris = get_word_vector("paris")

cosine_similarity(Madrid - Spain + France, Paris)

## Semantic analysis
m = DocumentTermMatrix(crps)
lsa(m) # Latent Semantic Analysis
lda # Latent Dirichlet Allocation 