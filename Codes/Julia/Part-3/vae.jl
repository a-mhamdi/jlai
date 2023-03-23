###################################
#= VARIATIONAL AUTOENCODER (VAE) =#
###################################

using Markdown
md"VAE implemented in `Julia` using the `Flux.jl` library"

using BSON

md"Import the machine learning library `Flux`" 
using Flux
using Flux: @functor
using Flux: Data.DataLoader
using Flux: onecold, onehotbatch

using ProgressMeter: Progress, next!

using MLDatasets
d = MNIST()

Base.@kwdef mutable struct Args
    η = 3f-3                # Learning rate
    λ = 1f-2                # Regularization parameter
    batchsize = 64          # Batch size
    epochs = 8              # Number of epochs
    split = :train          # Split data into `train` and `test`
    input_dim = 28*28       # Input dimension
    hidden_dim = 512        # Hidden dimension
    latent_dim = 2          # Latent dimension
    save_path = "Output"    # Results folder
end

md"Load the **MNIST** dataset"
function get_data(; kws...)
    args = Args(; kws...);
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
function vae(X, enc, dec)
    # Encode `X` into the latent space
    μ, log_σ = enc(X)
    # `Z` si a sample from the latent distribution
    Z = μ + randn(Float32, size(log_σ)) .* exp.(log_σ)
    # Decode the latent representation into a reconstruction of `X`
    x̂ = dec(Z)
    # Return μ, log_σ and x̂
    μ, log_σ, x̂
end

function l(X, enc, dec)
    μ, log_σ, x̂ = vae(X, enc, dec)
    len = size(X)[end]
    # The reconstruction loss measures how well the VAE was able to reconstruct the input data
    logp_x_z = -Flux.logitbinarycrossentropy(x̂, x, agg=sum) / len
    # The KL divergence loss measures how close the latent distribution is to the prior distribution
    kl_q_p = 5f-1 * sum(@. (exp(2f0 * log_σ) + μ^2 - 1f0 - 2f0 * log_σ)) / len
    #=
    # L2 Regularization
    reg = args.λ * sum(x->sum(x.^2), Flux.params(decoder))
    =#
    # Sum of the reconstruction loss and the KL divergence loss
    -logp_x_z + kl_q_p # + reg
end

function train(; kws...)
    args = Args(; kws...)
    
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
                    l(X, enc, dec)
                end
                grad_enc, grad_dec = back(1f0)
                Flux.update!(opt_enc, enc_mdl, grad_enc) # Upd `encoder` params
                Flux.update!(opt_dec, dec_mdl, grad_dec) # Upd `decoder` params
                next!(progress; showvalues=[(:loss, loss)]) 
        end
    end
    
    #=
    md"Save the model"
    mdl_path = joinpath(args.save_path, "vae.bson")
    let encoder=enc_mdl, decoder=dec_mdl, args=struct2dict(args)
    	BSON.@save mdl_path encoder decoder args
    	@info "Model saved to $(mdl_path)"
    end
    =#
    enc_mdl, dec_mdl
end

a, b = train()
