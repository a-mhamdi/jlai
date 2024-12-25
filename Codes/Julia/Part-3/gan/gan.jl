### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ 879b46a6-88fc-4692-8892-4863906edc04
import Pkg; Pkg.activate("."); Pkg.status()

# ╔═╡ 98e12a37-d4be-4898-ad5d-0d2640140b5a
using Flux # v"0.16.0"

# ╔═╡ 01ca737b-829c-4697-aa5d-6b907cd4c3fd
using Images: Gray

# ╔═╡ 189414a9-03b9-42d7-93ca-15ed58a20ce8
using ProgressMeter

# ╔═╡ 281920a0-6d99-4896-a8dd-3de46f1182fb
using Plots; theme(:dracula)

# ╔═╡ e085fb51-8865-4b12-80e9-e5a0ebcdb5da
md"# GENERATIVE ADVERSERIAL NETWORK"

# ╔═╡ 1ce188bb-b8ef-46c0-9f7f-bcc2141061c6
versioninfo() # -> v"1.11.2"

# ╔═╡ 299be62a-f582-4a1c-8642-9daefdf86bcc
md"**Generator:** noise vector -> synthetic sample."

# ╔═╡ eb7aa3c4-4340-4ed4-918e-2974002f613d
function generator(; latent_dim=16, img_shape=(28,28,1,1))
    return Chain(
        Dense(latent_dim, 128, relu),
        Dense(128, 256, relu),
        Dense(256, prod(img_shape), tanh),
        x -> reshape(x, img_shape)
    )
end

# ╔═╡ 3d441a62-547a-4060-8829-248a3dfadbed
md"**Discriminator:** sample -> score indicating the probability that the sample is real."

# ╔═╡ 69b4af29-b689-4b20-a773-78742bb68281
function discriminator(; img_shape=(28,28,1,1))
    return Chain(
        x -> reshape(x, :, size(x, 4)),
        Dense(prod(img_shape), 256, relu),
        Dense(256, 128, relu),
        Dense(128, 1)
    )
end

# ╔═╡ 5677cdd3-eaaa-42dc-ad83-da87c8bc81db
md"Loss function"

# ╔═╡ 11471f2f-df52-4764-b511-9fe1a8848432
bce_loss(y_true, y_pred) = Flux.logitbinarycrossentropy(y_pred, y_true)

# ╔═╡ 17f6ab1b-01f0-4cbf-be6e-aded2a1cbe14
md"Training function"

# ╔═╡ e866f93b-221d-4444-9fff-28e7c20028d6
function train_gan(gen, disc, gen_state, disc_state; n_epochs=16, latent_dim=16)
    @showprogress for epoch in 1:n_epochs

        ## Train the discriminator `disc`
        noise = randn(Float32, latent_dim, 1)
        fake_imgs = gen(noise) # pass the noise through the generator to get a synthetic sample
        real_imgs = rand(Float32, size(fake_imgs)...)

        disc_loss(m) = bce_loss(ones(Float32, 1, 1), m(real_imgs)) + 
                        bce_loss(zeros(Float32, 1, 1), m(fake_imgs)) # compute the loss for the real and synthetic samples
        grads = gradient(m -> disc_loss(m), disc)
        Flux.update!(disc_state, Flux.trainable(disc), grads[1]) # update the discriminator weights

        ## Train the generator `gen`
        noise = randn(Float32, latent_dim, 1)
        gen_loss(m) = bce_loss( ones(Float32, 1, 1), σ.(disc(m(noise))) ) # compute the loss for the synthetic samples
        grads = gradient(m -> gen_loss(m), gen)
        Flux.update!(gen_state, Flux.trainable(gen), grads[1]) # update the generator weights

        println("Epoch $(epoch): Discriminator loss = $(disc_loss), Generator loss = $(gen_loss)")
        sleep(.1)
    end
end

# ╔═╡ 3f82abca-0d00-4b36-85a9-055cb2dcc2c9
md"Setup the GAN"

# ╔═╡ 6c3f364c-0d0d-4a34-832e-f3653f15ea54
gen = generator()

# ╔═╡ 3b43ad53-e53e-4849-82db-0441ed8daa3b
disc = discriminator()

# ╔═╡ 4e20b5ff-8b80-4850-9d59-4058a3c16a24
gen_opt = Adam(0.001); gen_state = Flux.setup(gen_opt, gen)

# ╔═╡ 2fc91e4e-b736-4b24-856f-e997cea5e497
disc_opt = Adam(0.0002); disc_state = Flux.setup(disc_opt, disc)

# ╔═╡ 72cf2bf6-ab9a-461f-aeb0-ed17ca6e6f49
md"Train the GAN"

# ╔═╡ 0362230d-17e3-424e-b923-1e3b1c5359b5
train_gan(gen, disc, gen_state, disc_state)

# ╔═╡ b0b0e135-2d72-4e78-baa7-919434855de4
md"Generate and plot some images"

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
# ╠═1ce188bb-b8ef-46c0-9f7f-bcc2141061c6
# ╠═879b46a6-88fc-4692-8892-4863906edc04
# ╠═98e12a37-d4be-4898-ad5d-0d2640140b5a
# ╠═01ca737b-829c-4697-aa5d-6b907cd4c3fd
# ╠═189414a9-03b9-42d7-93ca-15ed58a20ce8
# ╠═299be62a-f582-4a1c-8642-9daefdf86bcc
# ╠═eb7aa3c4-4340-4ed4-918e-2974002f613d
# ╠═3d441a62-547a-4060-8829-248a3dfadbed
# ╠═69b4af29-b689-4b20-a773-78742bb68281
# ╠═5677cdd3-eaaa-42dc-ad83-da87c8bc81db
# ╠═11471f2f-df52-4764-b511-9fe1a8848432
# ╠═17f6ab1b-01f0-4cbf-be6e-aded2a1cbe14
# ╠═e866f93b-221d-4444-9fff-28e7c20028d6
# ╠═3f82abca-0d00-4b36-85a9-055cb2dcc2c9
# ╠═6c3f364c-0d0d-4a34-832e-f3653f15ea54
# ╠═3b43ad53-e53e-4849-82db-0441ed8daa3b
# ╠═4e20b5ff-8b80-4850-9d59-4058a3c16a24
# ╠═2fc91e4e-b736-4b24-856f-e997cea5e497
# ╠═72cf2bf6-ab9a-461f-aeb0-ed17ca6e6f49
# ╠═0362230d-17e3-424e-b923-1e3b1c5359b5
# ╠═b0b0e135-2d72-4e78-baa7-919434855de4
# ╠═b0bfaa5f-4e77-4b52-9970-dc0adb47bddb
# ╠═c7707565-b11c-45d1-a670-43b546d447d4
# ╠═c685204f-dd86-48ad-96d6-3d94ac30321a
# ╠═281920a0-6d99-4896-a8dd-3de46f1182fb
# ╠═e50e02cb-8601-4b8b-adad-e9c8f39e7ce5
# ╠═62f695e1-6cba-49df-b722-99ccfa5f8c73
# ╠═036ebfaf-e14e-4558-b795-d04923922489
