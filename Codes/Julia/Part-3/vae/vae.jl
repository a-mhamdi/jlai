### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ 4c4e5a73-dfb5-4925-9b3e-5100879feee6
	import Pkg; Pkg.activate("."); Pkg.status()

# ╔═╡ 5431e024-d2ae-4279-bf9e-c72e1455c7bc
using Flux # v"0.16.0"

# ╔═╡ 1d0e8c52-3758-4520-afde-2c7cadd3b427
using Flux: @functor

# ╔═╡ 4040b360-b6f0-4882-97b0-89d1b1b113b8
using Flux: DataLoader

# ╔═╡ ae7ff38a-bc09-442f-b2ba-0d4862b60ce6
using Flux: onecold, onehotbatch

# ╔═╡ 4c61b87f-cb04-4298-a895-05ff66d95c9d
using ProgressMeter: Progress, next!

# ╔═╡ 165e4355-ada6-4c29-9687-01ca368735a2
using MLDatasets

# ╔═╡ be2eb522-922c-4cd9-9f60-3305eacef23a
md"# VARIATIONAL AUTOENCODER (VAE)"

# ╔═╡ aae8ee36-56b0-4594-9454-e0fb13bb9e32
versioninfo() # -> v"1.11.2"

# ╔═╡ 6cdf0ae0-ac6b-4938-9598-e834cad5a94f
md"**VAE** implemented in `Julia` using the `Flux.jl` library"

# ╔═╡ a294646d-fc11-472e-9b3a-f7567387a373
md"Import the machine learning library `Flux`"

# ╔═╡ fea7f6a8-40c0-4024-963a-45dfc62dc88f
d = MNIST()

# ╔═╡ 4a4498cf-db31-4e3d-b94a-4f6c34057299
Base.@kwdef mutable struct HyperParams
    η = 3f-3                        # Learning rate
    λ = 1f-2                        # Regularization parameter
    batchsize = 64                  # Batch size
    epochs = 16                     # Number of epochs
    split = :train                  # Split data into `train` and `test`
    input_dim = 28*28               # Input dimension
    hidden_dim = 512                # Hidden dimension
    latent_dim = 2                  # Latent dimension
    # save_path = "Output"          # Results folder
end

# ╔═╡ 8f0c05ef-c253-4cd0-958c-c496e7e1c554
md"Load the **MNIST** dataset"

# ╔═╡ fa5c5b96-33c3-454f-802e-4c94ccbad4fb
function get_data(; kws...)
    args = HyperParams(; kws...);
    md"Split data"
    data = MNIST(split=args.split);
    X = reshape(data.features, (args.input_dim, :));
    loader = DataLoader(X; batchsize=args.batchsize, shuffle=true);
    return loader
end

# ╔═╡ 5fc45103-6a38-4fa9-a724-3fb185f809d5
train_loader = get_data();

# ╔═╡ 6a0cc987-638e-45cf-9f8c-e7cdbbfa189b
test_loader = get_data(split=:test);

# ╔═╡ 52d26401-72b9-421f-85bb-42cab7acf738
md"Define the `encoder` network: The encoder network should return the parameters of the _latent distribution_ (μ and σ)."

# ╔═╡ 1f68c46b-443f-404a-b872-f2ac0f5b0b0f
Encoder = begin 
	struct Encoder
	    linear
	    μ
	    log_σ
	end

	@functor Encoder

	function (encoder::Encoder)(x)
    	h = encoder.linear(x)
    	encoder.μ(h), encoder.log_σ(h)
	end
end

# ╔═╡ 8940c576-206d-42e2-bb3f-12ecedf963c2
encoder(input_dim::Int, hidden_dim::Int, latent_dim::Int) = Encoder(
    Dense(input_dim, hidden_dim, tanh),   # linear
    Dense(hidden_dim, latent_dim),        # μ
    Dense(hidden_dim, latent_dim),        # log_σ
)

# ╔═╡ b413a818-19f5-4d89-b8de-04bf085a2ffb
md"Define the `decoder` network: The decoder network should return the reconstruction of the input data"

# ╔═╡ f2b696d0-06c5-4570-8419-aa921609500c
decoder(input_dim::Int, hidden_dim::Int, latent_dim::Int) = Chain(
    Dense(latent_dim, hidden_dim, tanh),
    Dense(hidden_dim, input_dim)
)

# ╔═╡ bf736186-b254-42be-a2ec-a6d27af51b6e
md"Reconstruction of the input data"

# ╔═╡ 7dc4a830-9289-49bb-a690-32a9e82ff262
function vae(x, enc, dec)
    # Encode `x` into the latent space
    μ, log_σ = enc(x)
    # `z` si a sample from the latent distribution
    z = μ + randn(Float32, size(log_σ)) .* exp.(log_σ)
    # Decode the latent representation into a reconstruction of `x`
    x̂ = dec(z)
    # Return μ, log_σ and x̂
    μ, log_σ, x̂
end

# ╔═╡ db131d6e-c245-46cb-a292-b14c44a845b7
function l(x, enc, dec, λ)
    μ, log_σ, x̂ = vae(x, enc, dec)
    len = size(x)[end]
    # The reconstruction loss measures how well the VAE was able to reconstruct the input data
    logp_x_z = -Flux.Losses.logitbinarycrossentropy(x̂, x, agg=sum) / len
    # The KL divergence loss measures how close the latent distribution is to the normal distribution
    kl_q_p = 5f-1 * sum(@. (-2f0 * log_σ - 1f0 + exp(2f0 * log_σ) + μ^2)) / len
    # L2 Regularization
    reg = λ * sum( θ -> sum(θ.^2), Flux.params(dec) )
    # Sum of the reconstruction loss and the KL divergence loss
    -logp_x_z + kl_q_p + reg
end

# ╔═╡ 91d78390-28d4-456b-9a24-8708e9172233
function train(; kws...)
    args = HyperParams(; kws...)
    
    # Initialize `encoder` and `decoder`
    enc_mdl = encoder(args.input_dim, args.hidden_dim, args.latent_dim)
    dec_mdl = decoder(args.input_dim, args.hidden_dim, args.latent_dim)
    
    # ADAM optimizers
    opt_enc = Flux.setup(Adam(args.η), enc_mdl)
    opt_dec = Flux.setup(Adam(args.η), dec_mdl)

    for epoch in 1:args.epochs
        printstyled("\t***\t === EPOCH $(epoch) === \t*** \n", color=:magenta, bold=true)
        progress = Progress(length(train_loader))
        for X in train_loader
                loss, back = Flux.pullback(enc_mdl, dec_mdl) do enc, dec
                    l(X, enc, dec, args.λ)
                end
                grad_enc, grad_dec = back(1f0)
                Flux.update!(opt_enc, enc_mdl, grad_enc) # Upd `encoder` params
                Flux.update!(opt_dec, dec_mdl, grad_dec) # Upd `decoder` params
                next!(progress; showvalues=[(:loss, loss)]) 
        end
    end
    
    # Save the model
    #=
    using DrWatson: struct2dict
    using BSON

    mdl_path = joinpath(args.save_path, "vae.bson")
    let args=struct2dict(args)
    	BSON.@save mdl_path encoder decoder args
    	@info "Model saved to $(mdl_path)"
    end
    =#
    
    enc_mdl, dec_mdl
end

# ╔═╡ 73aec761-ab49-4235-9d52-3c80419de7e2
enc_model, dec_model = train()

# ╔═╡ Cell order:
# ╠═be2eb522-922c-4cd9-9f60-3305eacef23a
# ╠═aae8ee36-56b0-4594-9454-e0fb13bb9e32
# ╠═6cdf0ae0-ac6b-4938-9598-e834cad5a94f
# ╠═4c4e5a73-dfb5-4925-9b3e-5100879feee6
# ╠═a294646d-fc11-472e-9b3a-f7567387a373
# ╠═5431e024-d2ae-4279-bf9e-c72e1455c7bc
# ╠═1d0e8c52-3758-4520-afde-2c7cadd3b427
# ╠═4040b360-b6f0-4882-97b0-89d1b1b113b8
# ╠═ae7ff38a-bc09-442f-b2ba-0d4862b60ce6
# ╠═4c61b87f-cb04-4298-a895-05ff66d95c9d
# ╠═165e4355-ada6-4c29-9687-01ca368735a2
# ╠═fea7f6a8-40c0-4024-963a-45dfc62dc88f
# ╠═4a4498cf-db31-4e3d-b94a-4f6c34057299
# ╠═8f0c05ef-c253-4cd0-958c-c496e7e1c554
# ╠═fa5c5b96-33c3-454f-802e-4c94ccbad4fb
# ╠═5fc45103-6a38-4fa9-a724-3fb185f809d5
# ╠═6a0cc987-638e-45cf-9f8c-e7cdbbfa189b
# ╠═52d26401-72b9-421f-85bb-42cab7acf738
# ╠═1f68c46b-443f-404a-b872-f2ac0f5b0b0f
# ╠═8940c576-206d-42e2-bb3f-12ecedf963c2
# ╠═b413a818-19f5-4d89-b8de-04bf085a2ffb
# ╠═f2b696d0-06c5-4570-8419-aa921609500c
# ╠═bf736186-b254-42be-a2ec-a6d27af51b6e
# ╠═7dc4a830-9289-49bb-a690-32a9e82ff262
# ╠═db131d6e-c245-46cb-a292-b14c44a845b7
# ╠═91d78390-28d4-456b-9a24-8708e9172233
# ╠═73aec761-ab49-4235-9d52-3c80419de7e2
