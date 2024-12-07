####################################
#= Generative Adverserial Network =#
####################################
# `versioninfo()` -> 1.11.1

using Flux # v0.14.25
using Images: Gray
using ProgressMeter

## Generator: noise vector -> synthetic sample.
function generator(; latent_dim=16, img_shape=(28,28,1,1))
    return Chain(
        Dense(latent_dim, 128, relu),
        Dense(128, 256, relu),
        Dense(256, prod(img_shape), tanh),
        x -> reshape(x, img_shape)
    )
end

## Discriminator : sample -> score indicating the probability that the sample is real.
function discriminator(; img_shape=(28,28,1,1))
    return Chain(
        x -> reshape(x, :, size(x, 4)),
        Dense(prod(img_shape), 256, relu),
        Dense(256, 128, relu),
        Dense(128, 1)
    )
end

## Loss functions
bce_loss(y_true, y_pred) = Flux.logitbinarycrossentropy(y_pred, y_true)

## Training function
function train_gan(gen, disc, gen_opt, disc_opt; n_epochs=16, latent_dim=16)
    @showprogress for epoch in 1:n_epochs

        ## Train the discriminator `disc`
        noise = randn(Float32, latent_dim, 1)
        fake_imgs = gen(noise) # pass the noise through the generator to get a synthetic sample
        real_imgs = rand(Float32, size(fake_imgs)...)

        disc_loss = bce_loss(ones(Float32, 1, 1), disc(real_imgs)) + 
                        bce_loss(zeros(Float32, 1, 1), disc(fake_imgs)) # compute the loss for the real and synthetic samples
        grads = gradient(() -> disc_loss, Flux.params(disc))
        Flux.update!(disc_opt, Flux.params(disc), grads) # update the discriminator weights

        ## Train the generator `gen`
        noise = randn(Float32, latent_dim, 1)
        gen_loss = bce_loss( ones(Float32, 1, 1), σ.(disc(gen(noise))) ) # compute the loss for the synthetic samples
        grads = gradient(() -> gen_loss, Flux.params(gen))
        Flux.update!(gen_opt, Flux.params(gen), grads) # update the generator weights

        println("Epoch $(epoch): Discriminator loss = $(disc_loss), Generator loss = $(gen_loss)")
        sleep(.1)
    end
end

## Setup the GAN
gen = generator()
disc = discriminator()

gen_opt = Adam(0.001)
disc_opt = Adam(0.0002)

## Train the GAN
train_gan(gen, disc, gen_opt, disc_opt)

## Generate and plot some images
latent_dim = 16
noise = randn(Float32, latent_dim, 16)
generated_images = [ gen(noise[:, i]) for i in 1:16 ]

using Plots

plot_images = [ plot(Gray.(generated_images[i])[:,:,1,1]) for i in 1:16 ]
titles = reshape([string(i) for i in 1:16], 1, :);

plot(
    plot_images...,
    layout = (4, 4), 
    title = titles, titleloc=:right, titlefont=font(8),
    size = (800, 800)
)
