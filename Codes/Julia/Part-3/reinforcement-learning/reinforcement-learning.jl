############################
#= REINFORCEMENT LEARNING =#
############################
# `versioninfo()` -> 1.11.1

using ReinforcementLearning
using Flux: Descent

## Define the environment
env = RandomWalk1D()

## Instantiate the agent
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
hook = TotalRewardPerEpisode()
run(agent, env, StopAfterEpisode(10_000), hook)

## Print rewards
println("Total reward per episode:")
println(hook.rewards)

## Print `Q-table``
q_table = agent.policy.learner.approximator.table
println("\nLearned Q-table:")
println(q_table)
