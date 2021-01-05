import sys
import numpy as np
from pulp import *
import argparse

ap = argparse.ArgumentParser()

ap.add_argument("--mdp")
ap.add_argument("--algorithm")

args = ap.parse_args()

d_type = 'double'

def readMDP(file_path=None):
	f = open(file_path, 'r')
	nstates = int(''.join(f.readline().replace("\n", "").split(' ')[1:]))
	nactions = int(''.join(f.readline().replace("\n", "").split(' ')[1:]))
	start = int(''.join(f.readline().replace("\n", "").split(' ')[1:]))
	end = int(''.join(f.readline().replace("\n", "").split(' ')[1:]))
	rewards = np.zeros((nstates, nactions, nstates), dtype = d_type)
	transistion_p = np.zeros((nstates, nactions, nstates), dtype= d_type)

	results = []
	input_line = ''

	distinct_states = []
	while(1):
		input_line = f.readline().split(' ')[1:]
		if(len(input_line)<4):
			break
		results = [float(k) for k in input_line]
		rewards[int(results[0])][int(results[1])][int(results[2])] = results[3]
		transistion_p[int(results[0])][int(results[1])][int(results[2])]  = results[4]

	gamma = float(''.join(f.readline().replace("\n", "").split(' ')[1:]))

	return rewards, transistion_p, nstates, nactions, gamma

def value_iteration():
	V = np.ones(S)
	policy = np.zeros(S, dtype=int)
	action = np.zeros((S, A), dtype = d_type)
	const_vec = np.multiply(R, T).sum(axis=2)
	while(1):
		delta = 0.0
		v = V
		action = df *(np.multiply(V, T)).sum(axis=2) + const_vec
		V = np.max(action,axis=1)
		delta = max(delta, np.max(np.abs(v - V)))
		if delta < 0.000000000001:
			policy = np.argmax(action,axis=1)
			break

	return V, policy

def howard_policy_iteration():

	V = np.zeros(S)
	policy = np.zeros(S, dtype=int)
	new_policy = np.zeros(S, dtype=int)
	action = np.zeros((S, A), dtype = d_type)
	const_vec = np.multiply(R, T).sum(axis=2)
	change = True
	while(change):
		change = False
		rhs_vec = np.take_along_axis(const_vec, policy[:, None], 1)
		coeff_mat = np.eye(S) - df*np.take_along_axis(T, policy[:, None, None], 1)[:, 0]
		
		singular = np.array(coeff_mat==0).all(1)
		singular_states = np.where(singular)[0]
		for j in range(len(singular_states)):
			coeff_mat[singular_states[j]][singular_states[j]] = 1

		rhs_vec = np.array(rhs_vec)
		coeff_mat = np.array(coeff_mat)
		V = np.linalg.solve(coeff_mat,rhs_vec)
		V = V.reshape(S,)

		action = df *(np.multiply(V, T)).sum(axis=2) + const_vec
		new_policy = np.argmax(action,axis=1)

		if (new_policy == policy).all():
			break
		else:
			change = True
			policy = new_policy

	return V, policy


def linear_programmming():

	policy = np.zeros(S, dtype=int)
	prob = pulp.LpProblem("lp", LpMaximize)
	var_state = pulp.LpVariable.dicts("v", range(S))

	prob += pulp.lpSum([-var_state[i] for i in range(S)])
	for s in range(S):
		
		for a in range(A):      
			v_temp = 0
			for sPrime in range(S):
				v_temp += T[s][a][sPrime] * \
					(R[s][a][sPrime] +
					 df * var_state[sPrime])

			prob += var_state[s] >= v_temp

	LpSolverDefault.msg = 0
	prob.solve()

	V = np.zeros(S)
	for s in range(S):
		V[s] = pulp.value(var_state[s])

	vpi = np.zeros(S)
	for s in range(S):
		V_temp = np.zeros(A)
		for a in range(A):
			v_temp = 0
			for sPrime in range(S):
				v_temp += T[s][a][sPrime] * (
					R[s][a][sPrime] + df * V[sPrime])

			V_temp[a] = v_temp

		vpi[s] = np.max(V_temp)
		policy[s] = np.argmax(V_temp)

	return V, policy

def printOptimalPolicy(v, pi):
	for i in range(len(v)):
		print (str(format(v[i],'.6f')) + '\t' + str(pi[i]))

if __name__ == '__main__':
	mdp_file_path = args.mdp

	R,T,S,A,df = readMDP(mdp_file_path)

	if args.algorithm == "hpi":
		v_star, pi_star = howard_policy_iteration()

	elif args.algorithm == "lp":
		v_star, pi_star = linear_programmming()

	elif args.algorithm == "vi":
		v_star, pi_star = value_iteration()

	printOptimalPolicy(v_star, pi_star)