import sys
import numpy as np
import pickle
import argparse

ap = argparse.ArgumentParser()

ap.add_argument("--grid")

args = ap.parse_args()

filename = args.grid
file = open(filename,"r")
probability = 1.0
line = file.readline()
args = line.split(' ')
n = len(args)-1

maze_grid = np.zeros((n,n))

for i in range(n):
	maze_grid[0,i] = int(args[i])

for i in range(n-1):
	line = file.readline()
	args = line.split( )
	for j in range(n):
		maze_grid[i+1][j] = int(args[j])

total_states = n**2
total_actions = 4 # left = 0 right = 1 up = 2 down = 3

startState = -1
endStates = []
discount = 1

def validArray(i,j):
	by_action = []
	to_state = []
	row = []
	col = []

	if(maze_grid[i][j] != 1):
		if(j>0):
			if(maze_grid[i][j-1] != 1):
				by_action.append(0)
				to_state.append(i*n+j-1)
				row.append(i)
				col.append(j-1)
		if(j<n-1):
			if(maze_grid[i][j+1] != 1):
				by_action.append(1)
				to_state.append(i*n+j+1)
				row.append(i)
				col.append(j+1)
		if(i>0):
			if(maze_grid[i-1][j] != 1):
				by_action.append(2)
				to_state.append((i-1)*n+j)
				row.append(i-1)
				col.append(j)
		if(i<n-1):
			if(maze_grid[i+1][j] != 1):
				by_action.append(3)
				to_state.append((i+1)*n+j)
				row.append(i+1)
				col.append(j)
	return by_action, to_state, row, col


def transform_maze_grid():
	while(1):
		flag = 0
		for i in range(n):
			for j in range(n):
				by_action, to_state, row, col = validArray(i,j)
				total_valid_actions = len(by_action)
				if(total_valid_actions == 1 and maze_grid[i][j] != 2 and maze_grid[i][j]!=3):
					maze_grid[i][j] = 1
					flag = 1
		if(flag == 0):
			break
	return

transform_maze_grid()

with open('matrix.txt', 'w') as testfile:
    for row in maze_grid:
        testfile.write(' '.join([str(int(a)) for a in row]) + '\n')

transition = [[[] for _ in range(total_actions)] for _ in range(total_states)]
for i in range(n):
	for j in range(n):
		if(maze_grid[i][j]==1):
			continue
		if(maze_grid[i][j]==2):
			startState = i*n+j
		elif(maze_grid[i][j]==3):
			endStates.append(str(i*n+j))
			continue

		by_action, to_state, row, col = validArray(i,j)
		total_valid_actions = len(by_action)

		for u in range(total_valid_actions):
			if(maze_grid[row[u]][col[u]]==3):
				transition[i*n+j][by_action[u]].append((to_state[u], 10000.0, 1))
			else:
				transition[i*n+j][by_action[u]].append((to_state[u], -1.0, 1))

distinct_states = []
distinct_states.append(startState)
for i in range(total_states):
	for j in range(total_actions):
		for items in transition[i][j]:
			if items[0] not in distinct_states:
				distinct_states.append(items[0])

states_dict = {val : idx for idx, val in enumerate(distinct_states)}

print("total_states",len(distinct_states))
print("total_actions",total_actions)
print("start",startState)
print("end",' '.join(endStates))

a_file = open("data.pkl", "wb")
pickle.dump(states_dict, a_file)
a_file.close()

for i in range(total_states):
	for j in range(total_actions):
		for items in transition[i][j]:
			if i in states_dict:
				print("transition",states_dict[i],j,states_dict[items[0]],items[1],items[2])
print('mdp_encoder')
print("discount",discount)