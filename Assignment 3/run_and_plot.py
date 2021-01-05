import sys
import numpy as np
import matplotlib.pyplot as plt
from environments import WindyGridworld, WindyGridworldK, WindyGridworldS
from algos import SARSA, exp_SARSA, Q_learning

if __name__ == "__main__":
    graphTitle = ""
    combine = 0
    try:
        arg = sys.argv[1]
    except IndexError:
        print("Input format: ./run.sh <argument>")
        exit()
    if (arg == "default"):
        wG = WindyGridworld()
        graphTitle = "SARSA for regular Windy Gridworld"
    elif (arg == "kings"):
        wG = WindyGridworldK()
        graphTitle = "SARSA for Windy Gridworld with King's moves"
    elif (arg == "stochastic"):
        wG = WindyGridworldS()
        graphTitle = "SARSA for Windy Gridworld with Stochasticity"
    elif (arg == "default_all"):
        wG = WindyGridworld()
        combine = 1
        graphTitle = "Combined plot for Windy Gridworld"
    elif (arg == "kings_all"):
        wG = WindyGridworldK()
        combine = 1
        graphTitle = "Combined plot for Windy Gridworld with King's moves"
    elif (arg == "stochastic_all"):
        wG = WindyGridworldS()
        combine = 1
        graphTitle = "Combined plot for Windy Gridworld with Stochasticity"        
    else:
        print("Incorrect argument")
        exit()
    numStates = wG.getNumStates()
    numActions = wG.getNumActions()
    discount = wG.getDiscount()
    transitions = wG.getTransition()
    start = wG.getStartState()
    end = wG.getEndState()
    numEpisodes = 900
    
    yMean = np.zeros((numEpisodes, ))
    yMean1 = np.zeros((numEpisodes, ))
    yMean2 = np.zeros((numEpisodes, ))
    yMean3 = np.zeros((numEpisodes, ))
    seedvals = [30, 46, 73, 92, 29, 65, 8, 50, 11, 81]

    for seedval in seedvals:
        x3, y3 = SARSA(seedval, transitions, numStates, numActions, discount, start, end, numEpisodes)
        yMean3 += y3
    yMean3 /= len(seedvals)
    plt.plot(yMean3, x3, label='Sarsa')

    if (combine == 1):
	    for seedval in seedvals:
	        x1, y1 = Q_learning(seedval, transitions, numStates, numActions, discount, start, end, numEpisodes)
	        yMean1 += y1
	    yMean1 /= len(seedvals)
	    plt.plot(yMean1, x1, label='Q-learning')

	    for seedval in seedvals:
	        x2, y2 = exp_SARSA(seedval, transitions, numStates, numActions, discount, start, end, numEpisodes)
	        yMean2 += y2
	    yMean2 /= len(seedvals)
	    plt.plot(yMean2, x2, label='Expected Sarsa')

    plt.xlabel("Time Steps")
    plt.ylabel("Episodes")
    plt.title(graphTitle)
    # plt.show()
    plt.legend()
    plt.savefig(arg+'.jpg')