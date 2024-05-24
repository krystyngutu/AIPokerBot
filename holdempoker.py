# **No-Limit Holdem**

import os
import torch
import sys

import rlcard
from rlcard.agents import RandomAgent, DQNAgent

from rlcard.agents.human_agents.nolimit_holdem_human_agent import HumanAgent as NoLimitHoldemHumanAgent, HumanAgent
from rlcard.agents.human_agents.leduc_holdem_human_agent import HumanAgent as LeducHoldemHumanAgent
from rlcard.utils import tournament, Logger, plot_curve, reorganize, print_card


def getGameName(isTexas):
    return "no-limit-holdem" if isTexas else "leduc-holdem"


IS_TEXAS = True
GAME = getGameName(IS_TEXAS)


def getAgent(isTexas):
    return NoLimitHoldemHumanAgent if isTexas else LeducHoldemHumanAgent


env = rlcard.make(GAME, config = {"allow_step_back": True})
print(GAME)
evalEnv = rlcard.make(GAME)

agent = RandomAgent(num_actions = env.num_actions)

env.set_agents([agent for _ in range(env.num_players)])
trajectories, playerWins = env.run(is_training = False)

print(trajectories)
print(playerWins)

dqnPath = f"experiments/{GAME.replace('-', '_')}_dqn_result/dqn_model"
dqnFilename = "model.path"

agent = DQNAgent(num_actions = env.num_actions, state_shape = env.state_shape[0], mlp_layers = [64, 64], save_path = dqnPath)
agents = [agent]

for _ in range(1, env.num_players):
    agents.append(RandomAgent(num_actions = env.num_actions))
env.set_agents(agents)

with Logger(dqnPath) as logger:
    for episode in range(1000):

        trajectories, payoffs = env.run(is_training=True)
        trajectories = reorganize(trajectories, payoffs)

        for ts in trajectories[0]:
            agent.feed(ts)

        if episode % 100 == 0:
            logger.log_performance(episode, tournament(env, 10000)[0])

    csvPath, figPath = logger.csv_path, logger.fig_path
agent.save_checkpoint(dqnPath, filename = "model.path")
modelOutputPath = os.path.join(logger.log_dir, "model.path")
torch.save(agent, modelOutputPath)
print(f"Model file saved to: {modelOutputPath}")


plot_curve(csvPath, figPath, "DQN")
print("CSV Path: ", csvPath)
print("Fig Path: ", figPath)


def createEnvForUser(game, unloadedAgent):
    envUser = rlcard.make(game)

    print(game)
    humanAgent = HumanAgent(envUser.num_actions)

    # dqnAgent = DQNAgent.from_checkpoint(checkpoint = torch.load(unloadedAgent.save_path + "/model.path"))
    dqnAgent = torch.load(unloadedAgent.save_path + "/model.path")
    envUser.set_agents([humanAgent, dqnAgent])
    return envUser

print(agent.save_path)
gameEnv = createEnvForUser(getGameName(IS_TEXAS), agents[0])
# print(agent.get_state(0))

def runGame():
    print(">> No-Limit Hold'Em Pre-Trained Model")
    print(gameEnv.name)
    while True:
        print(">> Start a New Game")

        trajectories, payoffs = gameEnv.run(is_training = False)

        # If the human does not take the final action, print other players' actions

        finalState = trajectories[0][-1]
        actionRecord = finalState["action_record"]
        state = finalState["raw_obs"]
        actionList = []

        for i in range(1, len(actionRecord) + 1):
            if actionRecord[-i][0] == state["current_player"]:
                break
            actionList.insert(0, actionRecord[-i])

        for pair in actionList:
            print('>> Player', pair[0], "chooses", pair[1])

        # Let's look at what the agent card is

        print("===============     DQN Agent    ===============")
        print_card(gameEnv.get_perfect_information()["hand_cards"][1])

        print("===============     Result     ===============")
        if payoffs[0] > 0:
            print("You win {} chips!".format(payoffs[0]))
        elif payoffs[0] == 0:
            print("It is a tie.")
        else:
            print("You lose {} chips!".format(-payoffs[0]))
        print("")

        inputs = input("Press any key to continue, Q to exit\n")
        if inputs.lower() == "q":
            break


runGame()
