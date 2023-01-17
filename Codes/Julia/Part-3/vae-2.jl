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

# Define the encoder network
function encoder(x)
    x = reshape(x, :, 784)
    x = Dense(512, relu)(x)
    x = Dense(256, relu)(x)
    mu = Dense(2)(x)
    logvar = Dense(2)(x)
    return mu, logvar
end

# Define the decoder network
function decoder(z)
    z = Dense(256, relu)(z)
    z = Dense(512, relu)(z)
    z = Dense(784, sigmoid)(z)
    z = reshape(z, 28, 28, 1)
    return z
end

# Define the VAE
function VAE(x)
    mu, logvar = encoder(x)
    z = mu .+ exp.(logvar) .* randn(2)
    x_hat = decoder(z)
    return x_hat, mu, logvar
end

# Define the loss function
function loss(x, x_hat, mu, logvar)
    reconstruction_loss = Flux.mse(x, x_hat)
    kl_divergence = -0.5 * sum(1 .+ logvar .- mu .^ 2 .- exp.(logvar))
    return reconstruction_loss + kl_divergence
end

# Create a data iterator
batch_size = 128
train_data = DataLoader(X_train, batch_size, shuffle=true)

# Define the optimizer
opt = Flux.ADAM()

# Train the VAE
for epoch in 1:10
    for (x, _) in train_data
        x_hat, mu, logvar = VAE(x)
        L = loss(x, x_hat, mu, logvar)
        Flux.back!(L)
        Flux.update!(opt)
    end
    println("Epoch: $epoch, Loss: $(mean(L))")
end

# Generate samples from the VAE
z = randn(2)
x_hat = decoder(z)
Flux.mse(X_test[1], x_hat)

