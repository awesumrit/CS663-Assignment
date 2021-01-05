import numpy as np

class WindyGridworld:

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__r = 7
        self.__c = 10
        self.__numActions = 4
        self.__numStates = self.__r * self.__c
        self.__discount = 1
        self.__windStrength = np.zeros((self.__r, self.__c))
        self.__windStrength[:, [3, 4, 5, 8]] = 1
        self.__windStrength[:, [6, 7]] = 2
        self.__start = (3, 0)
        self.__end = (3, 7)
        self.__transitions = np.zeros((self.__numStates, self.__numActions, 2), dtype = int)
        self.__calculateTransition()

    def getNumStates(self):
        return self.__numStates
    
    def getNumActions(self):
        return self.__numActions

    def getDiscount(self):
        return self.__discount

    def getTransition(self):
        return self.__transitions

    def getStartState(self):
        return int(np.ravel_multi_index(self.__start, (self.__r, self.__c)))

    def getEndState(self):
        return int(np.ravel_multi_index(self.__end, (self.__r, self.__c)))

    def __calculateTransition(self):
        for state in range(self.__numStates):
            curPos = np.unravel_index(state, (self.__r, self.__c))
            for action in range(self.__numActions):
                # UP    - 0
                # RIGHT - 1
                # DOWN  - 2
                # LEFT  - 3
                if (action == 0):
                    newPos = curPos + np.array([1, 0]) + self.__windStrength[curPos] * np.array([1, 0])
                elif (action == 1):
                    newPos = curPos + np.array([0, 1]) + self.__windStrength[curPos] * np.array([1, 0])
                elif (action == 2):
                    newPos = curPos + np.array([-1, 0]) + self.__windStrength[curPos] * np.array([1, 0])
                elif (action == 3):
                    newPos = curPos + np.array([0, -1]) + self.__windStrength[curPos] * np.array([1, 0])
                newPos = self.__confineWithinGrid(newPos)
                nextState = np.ravel_multi_index(newPos.astype(int), (self.__r, self.__c))
                self.__transitions[state][action][0] = int(nextState)
                self.__transitions[state][action][1] = -1

    def __confineWithinGrid(self, pos):
        pos[0] = min(self.__r - 1, pos[0])
        pos[0] = max(0, pos[0])
        pos[1] = min(self.__c - 1, pos[1])
        pos[1] = max(0, pos[1])
        return pos

######################################################################################################################

class WindyGridworldK:

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__r = 7
        self.__c = 10
        self.__numActions = 8
        self.__numStates = self.__r * self.__c
        self.__discount = 1
        self.__windStrength = np.zeros((self.__r, self.__c))
        self.__windStrength[:, [3, 4, 5, 8]] = 1
        self.__windStrength[:, [6, 7]] = 2
        self.__start = (3, 0)
        self.__end = (3, 7)
        self.__transitions = np.zeros((self.__numStates, self.__numActions, 2), dtype = int)
        self.__calculateTransition()

    def getNumStates(self):
        return self.__numStates
    
    def getNumActions(self):
        return self.__numActions

    def getDiscount(self):
        return self.__discount

    def getTransition(self):
        return self.__transitions

    def getStartState(self):
        return int(np.ravel_multi_index(self.__start, (self.__r, self.__c)))

    def getEndState(self):
        return int(np.ravel_multi_index(self.__end, (self.__r, self.__c)))

    def __calculateTransition(self):
        for state in range(self.__numStates):
            curPos = np.unravel_index(state, (self.__r, self.__c))
            for action in range(self.__numActions):
                # N - 0     S - 4
                # NE - 1    SW - 5
                # E - 2     W - 6
                # SE - 3    NW - 7
                if (action == 0):
                    newPos = curPos + np.array([1, 0]) + self.__windStrength[curPos] * np.array([1, 0])
                elif (action == 1):
                    newPos = curPos + np.array([1, 1]) + self.__windStrength[curPos] * np.array([1, 0])
                elif (action == 2):
                    newPos = curPos + np.array([0, 1]) + self.__windStrength[curPos] * np.array([1, 0])
                elif (action == 3):
                    newPos = curPos + np.array([-1, 1]) + self.__windStrength[curPos] * np.array([1, 0])
                elif (action == 4):
                    newPos = curPos + np.array([-1, 0]) + self.__windStrength[curPos] * np.array([1, 0])
                elif (action == 5):
                    newPos = curPos + np.array([-1, -1]) + self.__windStrength[curPos] * np.array([1, 0])
                elif (action == 6):
                    newPos = curPos + np.array([0, -1]) + self.__windStrength[curPos] * np.array([1, 0])
                elif (action == 7):
                    newPos = curPos + np.array([1, -1]) + self.__windStrength[curPos] * np.array([1, 0])
                newPos = self.__confineWithinGrid(newPos)
                nextState = np.ravel_multi_index(newPos.astype(int), (self.__r, self.__c))
                self.__transitions[state][action][0] = int(nextState)
                self.__transitions[state][action][1] = -1

    def __confineWithinGrid(self, pos):
        pos[0] = min(self.__r - 1, pos[0])
        pos[0] = max(0, pos[0])
        pos[1] = min(self.__c - 1, pos[1])
        pos[1] = max(0, pos[1])
        return pos

######################################################################################################################

class WindyGridworldS:

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__r = 7
        self.__c = 10
        self.__numActions = 8
        self.__numStates = self.__r * self.__c
        self.__discount = 1
        self.__windStrength = np.zeros((self.__r, self.__c))
        self.__windStrength[:, [3, 4, 5, 8]] = 1
        self.__windStrength[:, [6, 7]] = 2
        self.__start = (3, 0)
        self.__end = (3, 7)
        self.__transitions = np.zeros((self.__numStates, self.__numActions, 2), dtype = object)
        self.__calculateTransition()

    def getNumStates(self):
        return self.__numStates
    
    def getNumActions(self):
        return self.__numActions

    def getDiscount(self):
        return self.__discount

    def getTransition(self):
        return self.__transitions

    def getStartState(self):
        return int(np.ravel_multi_index(self.__start, (self.__r, self.__c)))

    def getEndState(self):
        return int(np.ravel_multi_index(self.__end, (self.__r, self.__c)))

    def __calculateTransition(self):
        for state in range(self.__numStates):
            curPos = np.unravel_index(state, (self.__r, self.__c))
            for action in range(self.__numActions):
                # N - 0     S - 4
                # NE - 1    SW - 5
                # E - 2     W - 6
                # SE - 3    NW - 7
                if (action == 0):
                    newPos = curPos + np.array([1, 0]) + self.__windStrength[curPos] * np.array([1, 0])
                elif (action == 1):
                    newPos = curPos + np.array([1, 1]) + self.__windStrength[curPos] * np.array([1, 0])
                elif (action == 2):
                    newPos = curPos + np.array([0, 1]) + self.__windStrength[curPos] * np.array([1, 0])
                elif (action == 3):
                    newPos = curPos + np.array([-1, 1]) + self.__windStrength[curPos] * np.array([1, 0])
                elif (action == 4):
                    newPos = curPos + np.array([-1, 0]) + self.__windStrength[curPos] * np.array([1, 0])
                elif (action == 5):
                    newPos = curPos + np.array([-1, -1]) + self.__windStrength[curPos] * np.array([1, 0])
                elif (action == 6):
                    newPos = curPos + np.array([0, -1]) + self.__windStrength[curPos] * np.array([1, 0])
                elif (action == 7):
                    newPos = curPos + np.array([1, -1]) + self.__windStrength[curPos] * np.array([1, 0])
                newPos = self.__confineWithinGrid(newPos)
                nextState = np.ravel_multi_index(newPos.astype(int), (self.__r, self.__c))
                if (self.__windStrength[curPos] > 0):
                    newPos1 = newPos + np.array([1, 0])
                    newPos2 = newPos + np.array([-1, 0])
                    newPos1 = self.__confineWithinGrid(newPos1)
                    newPos2 = self.__confineWithinGrid(newPos2)
                    nextState1 = np.ravel_multi_index(newPos1.astype(int), (self.__r, self.__c))
                    nextState2 = np.ravel_multi_index(newPos2.astype(int), (self.__r, self.__c))
                    self.__transitions[state][action][0] = np.array([int(nextState), int(nextState1), int(nextState2)])
                else:
                    self.__transitions[state][action][0] = int(nextState)
                self.__transitions[state][action][1] = -1

    def __confineWithinGrid(self, pos):
        pos[0] = min(self.__r - 1, pos[0])
        pos[0] = max(0, pos[0])
        pos[1] = min(self.__c - 1, pos[1])
        pos[1] = max(0, pos[1])
        return pos