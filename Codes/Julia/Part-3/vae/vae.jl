###################################
#= VARIATIONAL AUTOENCODER (VAE) =#
###################################
# `versioninfo()` -> 1.11.1

using Markdown
md"VAE implemented in `Julia` using the `Flux.jl` library"

md"Import the machine learning library `Flux`" 
using Flux # v0.14.25
using Flux: @functor
using Flux: DataLoader
using Flux: onecold, onehotbatch

using ProgressMeter: Progress, next!

using MLDatasets
d = MNIST()

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

md"Load the **MNIST** dataset"
function get_data(; kws...)
    args = HyperParams(; kws...);
    md"Split data"
    data = MNIST(split=args.split);
    X = reshape(data.features, (args.input_dim, :));
    loader = DataLoader(X; batchsize=args.batchsize, shuffle=true);
    return loader
end

train_loader = get_data();
test_loader = get_data(split=:test);

md"Define the `encoder` network"
# The encoder network should return the parameters of the _latent distribution_ (μ and σ).
struct Encoder
    linear
    μ
    log_σ
end

@functor Encoder

encoder(input_dim::Int, hidden_dim::Int, latent_dim::Int) = Encoder(
    Dense(input_dim, hidden_dim, tanh),   # linear
    Dense(hidden_dim, latent_dim),        # μ
    Dense(hidden_dim, latent_dim),        # log_σ
)

function (encoder::Encoder)(x)
    h = encoder.linear(x)
    encoder.μ(h), encoder.log_σ(h)
end

md"Define the `decoder` network"
# The decoder network should return the reconstruction of the input data
decoder(input_dim::Int, hidden_dim::Int, latent_dim::Int) = Chain(
    Dense(latent_dim, hidden_dim, tanh),
    Dense(hidden_dim, input_dim)
)

md"Reconstruction of the input data"
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
    
    md"Save the model"
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

enc_model, dec_model = train()
