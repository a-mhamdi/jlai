using Flux, Flux.Data.MNIST
using Flux: onehotbatch, onecold
using Base.Iterators: partition

# Load the IMDB movie review dataset
X_train, y_train, X_test, y_test = IMDB.load()

# Preprocess the text data
X_train = lowercase.(X_train)
X_test = lowercase.(X_test)

X_train = replace.(X_train, r"<[^>]*>" => "")
X_test = replace.(X_test, r"<[^>]*>" => "")

X_train = replace.(X_train, r"[^a-zA-Z]" => " ")
X_test = replace.(X_test, r"[^a-zA-Z]" => " ")

X_train = map(x -> split(x, " "), X_train)
X_test = map(x -> split(x, " "), X_test)

# Build the vocabulary
vocab = Dict{String,Int}()

for review in X_train
    for word in review
        if !haskey(vocab, word)
            vocab[word] = length(vocab) + 1
        end
    end
end

# Encode the text data as a sequence of integers
X_train = map(x -> map(y -> get(vocab, y, 1), x), X_train)
X_test = map(x -> map(y -> get(vocab, y, 1), x), X_test)

# Pad the encoded sequences to the same length
max_length = maximum(length.(X_train))

X_train = map(x -> vcat(x, fill(1, max_length - length(x))), X_train)
X_test = map(x -> vcat(x, fill(1, max_length - length(x))), X_test)

# Convert the labels to one-hot encoding
y_train = onehotbatch(y_train, [0, 1])
y_test = onehotbatch(y_test, [0, 1])

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = partition(X_train, y_train, 0.8)

# Define the model
model = Chain(
    Embedding(length(vocab), 32),
    LSTM(32,

