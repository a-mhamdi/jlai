### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ f812809a-db39-42db-81cb-25a460401789
import Pkg; Pkg.activate("."); Pkg.status()

# ╔═╡ 674599bd-8d39-4847-9e42-187cd4bdbab2
using ReinforcementLearning

# ╔═╡ 1ec39a7a-2f80-4988-8985-8c01b2cd9961
using Flux: Descent

# ╔═╡ c6090d38-8307-42c1-8265-9804ce090371
md"# REINFORCEMENT LEARNING"

# ╔═╡ 9ba566c6-470a-4232-912a-dc5d4f5ce318
versioninfo() # -> v"1.11.2"

# ╔═╡ fb20a86c-d7aa-4282-a8b9-49b31c0ea0e7
md"## Import required librairies"

# ╔═╡ ab5ce601-1f6b-4de1-bdf9-36f440abb49f
md"## Define the environment"

# ╔═╡ 68cf99a5-c64d-4a12-b4ef-d8e17021ff7c
env = RandomWalk1D()

# ╔═╡ 8a8e49c5-a07a-409b-a58f-3e88936c261b
md"## Instantiate the agent"

# ╔═╡ cfd93955-b3c1-451a-8ff3-e44f1b3c0536
begin
	approximator = TabularQApproximator(; n_state=7, n_action=2)
	println(approximator)
end

# ╔═╡ 288620f7-784e-41d9-a297-84150b286203
begin
	learner = TDLearner(approximator, :SARS; γ=0.99)
	println(learner)
end

# ╔═╡ b4831f65-57d7-4164-bf92-3d4a3db3aaa5
policy = QBasedPolicy(;
		learner = learner,
        explorer = EpsilonGreedyExplorer(0.1)
    )

# ╔═╡ a304b40a-f8bb-475e-a157-28bed3d6d27a
run(policy, env, StopAfterNEpisodes(10), TotalRewardPerEpisode())

# ╔═╡ 95b16227-4bed-4af0-9624-8cd02b04299a
trajectory = Trajectory(
           ElasticArraySARTSTraces(;
               state = Int64 => (),
               action = Int64 => (),
               reward = Float64 => (),
               terminal = Bool => (),
           ),
           DummySampler(),
           InsertSampleRatioController(),
       )

# ╔═╡ cf565284-147d-47a6-86a4-2c8900ca5083
agent = Agent(
           policy = policy,
           trajectory = trajectory
       )

# ╔═╡ 58020816-c6f4-49ff-b6fe-2c41a7c1d125
md"## Run the experiment"

# ╔═╡ be99789f-8497-4c34-b7a2-38726693b6b2
hook = TotalRewardPerEpisode()

# ╔═╡ 4668e284-098b-49f5-bb13-70d0b7111122
run(policy, env, StopAfterNEpisodes(10), hook)

# ╔═╡ b3e2309f-1f1a-4b13-b6d3-07ae732c5e28
md"## Print rewards"

# ╔═╡ 8a69b7a1-934c-45da-ac79-b7fead1dbca3
begin
	println("Total reward per episode:")
	println(hook.rewards)
end

# ╔═╡ 23790d54-d24f-41f5-a72e-6ecbe054d17d
md"## Print `Q-table`"

# ╔═╡ 370a523d-6802-4bfe-9789-0e546a0ae537
begin
	println("\nLearned Q-table:")
	println(agent.policy.learner.approximator)
end

# ╔═╡ Cell order:
# ╠═c6090d38-8307-42c1-8265-9804ce090371
# ╠═9ba566c6-470a-4232-912a-dc5d4f5ce318
# ╠═f812809a-db39-42db-81cb-25a460401789
# ╠═fb20a86c-d7aa-4282-a8b9-49b31c0ea0e7
# ╠═674599bd-8d39-4847-9e42-187cd4bdbab2
# ╠═1ec39a7a-2f80-4988-8985-8c01b2cd9961
# ╠═ab5ce601-1f6b-4de1-bdf9-36f440abb49f
# ╠═68cf99a5-c64d-4a12-b4ef-d8e17021ff7c
# ╠═8a8e49c5-a07a-409b-a58f-3e88936c261b
# ╠═cfd93955-b3c1-451a-8ff3-e44f1b3c0536
# ╠═288620f7-784e-41d9-a297-84150b286203
# ╠═b4831f65-57d7-4164-bf92-3d4a3db3aaa5
# ╠═a304b40a-f8bb-475e-a157-28bed3d6d27a
# ╠═95b16227-4bed-4af0-9624-8cd02b04299a
# ╠═cf565284-147d-47a6-86a4-2c8900ca5083
# ╠═58020816-c6f4-49ff-b6fe-2c41a7c1d125
# ╠═be99789f-8497-4c34-b7a2-38726693b6b2
# ╠═4668e284-098b-49f5-bb13-70d0b7111122
# ╠═b3e2309f-1f1a-4b13-b6d3-07ae732c5e28
# ╠═8a69b7a1-934c-45da-ac79-b7fead1dbca3
# ╠═23790d54-d24f-41f5-a72e-6ecbe054d17d
# ╠═370a523d-6802-4bfe-9789-0e546a0ae537
