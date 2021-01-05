import sys
import numpy as np
import pickle
import argparse
import os

ap = argparse.ArgumentParser()

ap.add_argument("--grid")
ap.add_argument("--value_policy")

args = ap.parse_args()

maze_gridfilename = args.grid
value_and_policy_file_name = args.value_policy
maze_gridfile = open(maze_gridfilename,"r")


line = maze_gridfile.readline()
args = line.split(' ')
n = len(args)-1
maze_grid = np.zeros((n,n))

for i in range(n):
	maze_grid[0,i] = int(args[i])

for i in range(n-1):
	line = maze_gridfile.readline()
	args = line.split( )
	for j in range(n):
		maze_grid[i+1][j] = int(args[j])

a_file = open("data.pkl", "rb")
states_dict = pickle.load(a_file)

startState = -1
endStates = []
for i in range(n):
	for j in range(n):
		if(int(maze_grid[i][j]==2)):
			s = i*n+j
			startState = i*n+j
		elif(int(maze_grid[i][j]==3)):
			endStates.append((i*n+j))



total_states = len(states_dict)
total_actions = 4 # left = 0 right = 1 up = 2 down = 3

value_policy_file = open(value_and_policy_file_name,"r")

policy = []
for i in range(total_states):
	line = value_policy_file.readline()
	args = line.split( )
	policy.append(int(float(args[1])))

def position(s):
	return(int(s/n),int(s%n))

def validArray(i,j):
	by_action = []
	to_state = []

	if(maze_grid[i][j] != 1):
		if(j>0):
			if(maze_grid[i][j-1] != 1):
				by_action.append(0)
				to_state.append(i*n+j-1)
		if(j<n-1):
			if(maze_grid[i][j+1] != 1):
				by_action.append(1)
				to_state.append(i*n+j+1)
		if(i>0):
			if(maze_grid[i-1][j] != 1):
				by_action.append(2)
				to_state.append((i-1)*n+j)
		if(i<n-1):
			if(maze_grid[i+1][j] != 1):
				by_action.append(3)
				to_state.append((i+1)*n+j)
	return by_action, to_state


s = states_dict[startState]
currState = startState
moves = []
while(1):
	flag = 0
	for i in range(len(endStates)):
		if(s == states_dict[endStates[i]]):
			flag = 1
	if flag:
		break
	move = policy[s]
	moves.append(move)
	if(move==0):
		currState-=1
	elif(move==1):
		currState+=1
	elif(move==2):
		currState-=n
	elif(move==3):
		currState+=n

	s = states_dict[currState]

for i in range(len(moves)):
	if(moves[i]==0):
		moves[i] = "W"
	elif(moves[i]==1):
		moves[i] = "E"
	elif(moves[i]==2):
		moves[i] = "N"
	elif(moves[i]==3):
		moves[i] = "S"

print (" ".join(moves))

# file_path = 'data.pkl'
# os.remove(file_path)
