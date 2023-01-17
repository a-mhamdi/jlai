using Flux
using Flux.Data.MNIST
using Flux: onehotbatch, onecold
using Base.Iterators: repeated, partition

# Load the MNIST dataset
X_train, y_train, X_test, y_test = MNIST.load()

# Convert the images to floating-point tensors
X_train = Flux.data(X_train)
X_test = Flux.data(X_test)

# Normalize the pixel values
X_train = X_train .- 128f0
X_test = X_test .- 128f0

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = partition(X_train, y_train, 0.8)

# Convert the labels to one-hot encoding
y_train = onehotbatch(y_train, 0:9)
y_val = onehotbatch(y_val, 0:9)

# Define the model
model = Chain(
    Conv((3, 3), 1 => 8, pad=(1, 1), relu),
    x -> maxpool(x, (2, 2)),
    Conv((3, 3), 8 => 16, pad=(1, 1), relu),
    x -> maxpool(x, (2, 2)),
    x -> reshape(x, :, 16 * 7 * 7),
    Dense(16 * 7 * 7, 32, relu),
    Dense(32, 10),
    softmax
)

# Define the loss function and an optimizer
loss_fn = Flux.crossentropy
opt = Flux.ADAM()

# Train the model
for epoch in 1:10
    for (x, y) in zip(X_train, y_train)
        # Compute the gradient of the loss with respect to the model's parameters
        gs = gradient(params(model)) do
            y_pred = model(x)
            loss_fn(y_pred, y)
        end

        # Update the model's parameters using the optimizer
        Flux.update!(opt, params(model), gs)
    end

    # Calculate the accuracy on the validation set
    accuracy = sum(onecold(model(X_val)) .== onecold(y_val)) / length(y_val)

    println("Epoch: $epoch, Accuracy: $accuracy")
end

# Test the model on the test set
accuracy = sum(onecold(model(X_test)) .== onecold(y_test)) / length(y_test)
println("Test accuracy: $accuracy")

