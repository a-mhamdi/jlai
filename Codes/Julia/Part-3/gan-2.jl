using Flux
using Flux.Data.MNIST
using Flux: onehotbatch, onecold
using Flux.Data: DataLoader

# Load the MNIST dataset
X_train, y_train, X_test, y_test = MNIST.load()

# Convert the images to floating-point tensors
X_train = Flux.data(X_train)
X_test = Flux.data(X_test)

# Normalize the pixel values
X_train = X_train .- 128f0
X_test = X_test .- 128f0

# Define the generator network
function generator(z)
    z = Dense(256, relu)(z)
    z = Dense(512, relu)(z)
    z = Dense(784, tanh)(z)
    z = reshape(z, 28, 28, 1)
    return z
end

# Define the discriminator network
function discriminator(x)
    x = reshape(x, :, 784)
    x = Dense(512, relu)(x)
    x = Dense(256, relu)(x)
    x = Dense(1, sigmoid)(x)
    return x
end

# Define the GAN
function GAN(z)
    x_hat = generator(z)
    return discriminator(x_hat)
end

# Define the loss function
function loss(x, x_hat)
    reconstruction_loss = Flux.mse(x, x_hat)
    return reconstruction_loss
end

# Create a data iterator
batch_size = 128
train_data = DataLoader(X_train, batch_size, shuffle=true)

# Define the optimizers
g_opt = Flux.ADAM()
d_opt = Flux.ADAM()

# Train the GAN
for epoch in 1:10
    for (x, _) in train_data
        # Train the discriminator
        z = randn(100)
        x_hat = generator(z)
        d_loss = Flux.mse(discriminator(x), 1) + Flux.mse(discriminator(x_hat), 0)
        Flux.back!(d_loss)
        Flux.update!(d_opt)

        # Train the generator
        z = randn(100)
        x_hat = generator(z)
        g_loss = loss(x, x_hat)
        Flux.back!(g_loss)
        Flux.update!(g_opt)
    end
    println("Epoch: $epoch, D Loss: $(mean(d_loss)), G Loss: $(mean(g_loss))")
end

# Generate samples from the GAN
z = randn(100)
x_hat = generator(z)
Flux.mse(X_test[1], x_hat)

