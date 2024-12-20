### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# ╔═╡ c6090d38-8307-42c1-8265-9804ce090371
############################
#= REINFORCEMENT LEARNING =#
############################
# `versioninfo()` -> 1.11.1

using ReinforcementLearning

# ╔═╡ 1ec39a7a-2f80-4988-8985-8c01b2cd9961
using Flux: Descent

## Define the environment

# ╔═╡ 68cf99a5-c64d-4a12-b4ef-d8e17021ff7c
env = RandomWalk1D()

## Instantiate the agent

# ╔═╡ e9addb75-be17-432c-bc3e-6d5a701314f7
agent = Agent(
    policy = QBasedPolicy(
        learner = TDLearner(
            approximator = TabularQApproximator(
                n_state = 11,
                n_action = 2,
                init = 0.0,
                opt = Descent(0.1) # Learning rate
            ),
            method = :SARSA,
            γ = 0.99
        ),
        explorer = EpsilonGreedyExplorer(0.1),
    ),
    trajectory = VectorSARTTrajectory(),
)

## Run the experiment

# ╔═╡ 0e93b315-bcdb-48ad-be03-7f1a80d41b8b
hook = TotalRewardPerEpisode()

# ╔═╡ 4668e284-098b-49f5-bb13-70d0b7111122
run(agent, env, StopAfterEpisode(10_000), hook)

## Print rewards

# ╔═╡ 67199d43-7854-4f2c-a2bb-49ef612f13a4
println("Total reward per episode:")

# ╔═╡ 0070ffc4-9886-4bc9-98fd-57c0651edd03
println(hook.rewards)

## Print `Q-table``

# ╔═╡ 0f1e09b8-063c-470e-b8f7-14abed9df8d4
q_table = agent.policy.learner.approximator.table

# ╔═╡ 370a523d-6802-4bfe-9789-0e546a0ae537
println("\nLearned Q-table:")

# ╔═╡ 51821e1b-8fd1-4242-a9e5-f8dc4495d47a
println(q_table)

# ╔═╡ Cell order:
# ╠═c6090d38-8307-42c1-8265-9804ce090371
# ╠═1ec39a7a-2f80-4988-8985-8c01b2cd9961
# ╠═68cf99a5-c64d-4a12-b4ef-d8e17021ff7c
# ╠═e9addb75-be17-432c-bc3e-6d5a701314f7
# ╠═0e93b315-bcdb-48ad-be03-7f1a80d41b8b
# ╠═4668e284-098b-49f5-bb13-70d0b7111122
# ╠═67199d43-7854-4f2c-a2bb-49ef612f13a4
# ╠═0070ffc4-9886-4bc9-98fd-57c0651edd03
# ╠═0f1e09b8-063c-470e-b8f7-14abed9df8d4
# ╠═370a523d-6802-4bfe-9789-0e546a0ae537
# ╠═51821e1b-8fd1-4242-a9e5-f8dc4495d47a
