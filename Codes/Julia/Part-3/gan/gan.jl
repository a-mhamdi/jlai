### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# ╔═╡ e085fb51-8865-4b12-80e9-e5a0ebcdb5da
####################################
#= Generative Adverserial Network =#
####################################
# `versioninfo()` -> 1.11.1

using Flux # v0.14.25

# ╔═╡ 01ca737b-829c-4697-aa5d-6b907cd4c3fd
using Images: Gray

# ╔═╡ 189414a9-03b9-42d7-93ca-15ed58a20ce8
using ProgressMeter

## Generator: noise vector -> synthetic sample.

# ╔═╡ 281920a0-6d99-4896-a8dd-3de46f1182fb
using Plots

# ╔═╡ eb7aa3c4-4340-4ed4-918e-2974002f613d
function generator(; latent_dim=16, img_shape=(28,28,1,1))
    return Chain(
        Dense(latent_dim, 128, relu),
        Dense(128, 256, relu),
        Dense(256, prod(img_shape), tanh),
        x -> reshape(x, img_shape)
    )
end

## Discriminator : sample -> score indicating the probability that the sample is real.

# ╔═╡ 69b4af29-b689-4b20-a773-78742bb68281
function discriminator(; img_shape=(28,28,1,1))
    return Chain(
        x -> reshape(x, :, size(x, 4)),
        Dense(prod(img_shape), 256, relu),
        Dense(256, 128, relu),
        Dense(128, 1)
    )
end

## Loss functions

# ╔═╡ 11471f2f-df52-4764-b511-9fe1a8848432
bce_loss(y_true, y_pred) = Flux.logitbinarycrossentropy(y_pred, y_true)

## Training function

# ╔═╡ e866f93b-221d-4444-9fff-28e7c20028d6
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

# ╔═╡ 6c3f364c-0d0d-4a34-832e-f3653f15ea54
gen = generator()

# ╔═╡ 3b43ad53-e53e-4849-82db-0441ed8daa3b
disc = discriminator()

# ╔═╡ 4e20b5ff-8b80-4850-9d59-4058a3c16a24
gen_opt = Adam(0.001)

# ╔═╡ 2fc91e4e-b736-4b24-856f-e997cea5e497
disc_opt = Adam(0.0002)

## Train the GAN

# ╔═╡ 0362230d-17e3-424e-b923-1e3b1c5359b5
train_gan(gen, disc, gen_opt, disc_opt)

## Generate and plot some images

# ╔═╡ b0bfaa5f-4e77-4b52-9970-dc0adb47bddb
latent_dim = 16

# ╔═╡ c7707565-b11c-45d1-a670-43b546d447d4
noise = randn(Float32, latent_dim, 16)

# ╔═╡ c685204f-dd86-48ad-96d6-3d94ac30321a
generated_images = [ gen(noise[:, i]) for i in 1:16 ]

# ╔═╡ e50e02cb-8601-4b8b-adad-e9c8f39e7ce5
plot_images = [ plot(Gray.(generated_images[i])[:,:,1,1]) for i in 1:16 ]

# ╔═╡ 62f695e1-6cba-49df-b722-99ccfa5f8c73
titles = reshape([string(i) for i in 1:16], 1, :);

# ╔═╡ 036ebfaf-e14e-4558-b795-d04923922489
plot(
    plot_images...,
    layout = (4, 4), 
    title = titles, titleloc=:right, titlefont=font(8),
    size = (800, 800)
)

# ╔═╡ Cell order:
# ╠═e085fb51-8865-4b12-80e9-e5a0ebcdb5da
# ╠═01ca737b-829c-4697-aa5d-6b907cd4c3fd
# ╠═189414a9-03b9-42d7-93ca-15ed58a20ce8
# ╠═eb7aa3c4-4340-4ed4-918e-2974002f613d
# ╠═69b4af29-b689-4b20-a773-78742bb68281
# ╠═11471f2f-df52-4764-b511-9fe1a8848432
# ╠═e866f93b-221d-4444-9fff-28e7c20028d6
# ╠═6c3f364c-0d0d-4a34-832e-f3653f15ea54
# ╠═3b43ad53-e53e-4849-82db-0441ed8daa3b
# ╠═4e20b5ff-8b80-4850-9d59-4058a3c16a24
# ╠═2fc91e4e-b736-4b24-856f-e997cea5e497
# ╠═0362230d-17e3-424e-b923-1e3b1c5359b5
# ╠═b0bfaa5f-4e77-4b52-9970-dc0adb47bddb
# ╠═c7707565-b11c-45d1-a670-43b546d447d4
# ╠═c685204f-dd86-48ad-96d6-3d94ac30321a
# ╠═281920a0-6d99-4896-a8dd-3de46f1182fb
# ╠═e50e02cb-8601-4b8b-adad-e9c8f39e7ce5
# ╠═62f695e1-6cba-49df-b722-99ccfa5f8c73
# ╠═036ebfaf-e14e-4558-b795-d04923922489
