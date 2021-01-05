import numpy as np
import math
import sys
import matplotlib.pyplot as plt
import argparse
		
parser = argparse.ArgumentParser()
parser.add_argument("--instance", type=str)
parser.add_argument("--algorithm", type=str)
parser.add_argument("--randomSeed", type=int)
parser.add_argument("--epsilon", type=float)
parser.add_argument("--horizon", type=int)
args = parser.parse_args()

alpha = 2 # parameter of ucb
T = int(args.horizon)
epsilon = float(args.epsilon)
args.randomSeed = int(args.randomSeed)
np.random.seed(args.randomSeed)
f = open(args.instance, 'r')
mean_instance = []
for line in f:
	mean_instance.append(float(line.strip()))

k = len(mean_instance)
numArms = k
num_arms = k 
p = mean_instance[0] 
for i in range(1, len(mean_instance)): 
    if mean_instance[i] > p: 
        p = mean_instance[i]

def select_arm(arm):
	#args.randomSeed += 100
	return np.random.binomial(1, mean_instance[arm])


def divergence(p,q):
	if p==0:
		if q==1:
			return math.inf
		else:
			return (1-p)*math.log((1-p)/(1-q))

	elif p==1:
		if q==0:
			return math.inf
		else:
			return p*math.log(p/q)
	else:
		return (p*math.log(p/q)+(1-p)*math.log((1-p)/(1-q)))

def updateQMax(counts, rounds_until_now, mu_hat):

	precision = 1e-4;
	qMax = np.zeros(numArms)
	for i in range(numArms):
		p = mu_hat[i]
		prev = p
		end = 1
		mid = (prev+end)/2
		if p==1:
			qMax[i] = 1
		else:
			while 1:
				mid = (prev+end)/2
				kl = divergence(p,mid)
				rhs = (math.log(rounds_until_now))/counts[i]
				if abs(kl - rhs) <= precision:
					break
				if kl-rhs<0:
					prev = mid
				else:
					end = mid
			qMax[i] = mid
	return qMax

   
  
def e_greedy(total_rounds):

	eps = epsilon
	counts = np.ones(k)
	values = np.zeros(k)
	prob = np.zeros(k)
	for i in range(k):
		prob[i] = float(1)/k


	for explore in range(2*k):
		It = explore%k
		values[It] += select_arm(It)
		counts[It] +=1
	
	mu_hat = np.zeros(k)

	regret = np.zeros(total_rounds)
	mu_hat = values/counts
	Aj = np.argmax(mu_hat)
	max_mu = p

	for t in range(2*k, total_rounds):
		
		probs = np.array([1-eps if i == Aj else eps/(k-1) for i in range(k)])
		# np.random.seed(args.randomSeed)
		It = np.random.choice(np.arange(0,k), p=probs)
		reward = select_arm(It)
		# It
		values[It] += reward
		counts[It] += 1
		mu_hat = values/counts
		Aj = np.argmax(mu_hat)
		regret[t] = regret[t-1] + max_mu - (mu_hat[It]) 

	return regret[total_rounds-1]

def kl_ucb(num_arms, total_rounds):
	k = num_arms
	eps = [0]*k
	qMax = np.zeros(k) 
	counts = np.zeros(k)
	values = np.zeros(k)
	loss = np.zeros(k)
	x = np.zeros(k)
	nx = np.ones(k)
	max_mu = p
	cumulativeReward = 0
	mu_hat = np.zeros(k)

	regret = np.zeros(total_rounds)
	

	for explore in range(k):
		It = explore
		reward = select_arm(It)
		values[It] += reward
		loss[It] += 1-reward
		counts[It] +=1
		cumulativeReward += reward
		
	mu_hat = values/counts
	qMax = updateQMax(counts, k, mu_hat)

	for t in range(k, total_rounds):

		qMax = updateQMax(counts, t, mu_hat)
		Aj = np.argmax(qMax)
		reward = select_arm(Aj)
		values[Aj] += reward
		counts[Aj] += 1

		mu_hat = values/counts
		cumulativeReward += reward

		regret[t] += regret[t-1] + max_mu -(mu_hat[Aj])

	return regret[total_rounds-1]

def ucb(num_arms, total_rounds):
	k = num_arms
	eps = [0]*k
	counts = np.zeros(k)
	values = np.zeros(k)
	loss = np.zeros(k)
	x = np.zeros(k)
	nx = np.ones(k)

	max_mu = p
	mu_hat = np.zeros(k)
	numTotalPulls = 0
	cumulativeReward = 0
	for explore in range(k):
		It = explore
		reward = select_arm(It)
		values[It] += reward
		loss[It] += 1-reward
		counts[It] +=1
		cumulativeReward += reward

	mu_hat = values/counts
	regret = np.zeros(total_rounds)

	for t in range(k, total_rounds):
		Aj = np.argmax(mu_hat + np.sqrt(alpha*math.log(t)/counts))
		# Aj
		reward = select_arm(Aj)
		values[Aj] += reward
		loss[Aj] += 1-reward
		counts[Aj] += 1
		mu_hat = values/counts

		cumulativeReward += reward
		index = np.argmax(counts)

		regret[t] += regret[t-1] + max_mu -(mu_hat[Aj])

	return regret[total_rounds-1]

def thompson(num_arms, total_rounds):
	k = num_arms
	eps = [0]*k
	counts = np.zeros(k)
	values = np.zeros(k)
	loss = np.zeros(k)
	s1 = np.zeros(k)
	f1 = np.zeros(k)
	mu_hat = np.zeros(k)
	theta = np.zeros(k)
	regret1 = np.zeros(total_rounds)
	max_mu = p
	for t in range(total_rounds):
		for i in range(k):
			a = s1[i] + 1
			b = f1[i] + 1
			#args.randomSeed += 100
			# np.random.seed(args.randomSeed)
			theta[i] = np.random.beta(a, b)

		It = np.argmax(theta)
		reward = select_arm(It)
		if(reward == 1):
			s1[It] += 1
		else:
			f1[It] += 1
		mu_hat[It] = s1[It]/(s1[It]+f1[It])

		regret1[t] += regret1[t-1] + max_mu - (mu_hat[It])

	return regret1[total_rounds-1]

def thompson_hint(num_arms, total_rounds):
	k = num_arms
	eps = [0]*k
	counts = np.zeros(k)
	values = np.zeros(k)
	loss = np.zeros(k)
	s = np.zeros(k)
	f = np.zeros(k)
	mu_hat = np.zeros(k)
	theta = np.zeros(k)
	regret = np.zeros(total_rounds)
	max_mu = p
	for i in range(k):
		s[i] = 1e4*max_mu
		f[i] = 1e4 - s[i]

	for t in range(total_rounds):
		for i in range(k):
			a = s[i] + 1
			b = f[i] + 1
			# np.random.seed(args.randomSeed)
			theta[i] = np.random.beta(a, b)
			#args.randomSeed += 100
		# theta
		It = np.argmax(theta)
		# It
		reward = select_arm(It)
		if(reward == 1):
			s[It] += 1
		else:
			f[It] += 1
		mu_hat[It] = (s[It])/(s[It]+f[It])

		regret[t] += regret[t-1] + max_mu - (mu_hat[It])

	return regret[total_rounds-1]


if args.algorithm == 'epsilon-greedy':
	REG = e_greedy(T)
	print(args.instance, args.algorithm, args.randomSeed, epsilon, T, REG, sep=', ')

elif args.algorithm == 'kl-ucb':
	REG = kl_ucb(k,T)
	print(args.instance, args.algorithm, args.randomSeed, epsilon, T, REG, sep=', ')

elif args.algorithm == 'ucb':
	REG = ucb(k,T)
	print(args.instance, args.algorithm, args.randomSeed, epsilon, T, REG, sep=', ')

elif args.algorithm == 'thompson-sampling-with-hint':
	REG = thompson_hint(k,T)
	print(args.instance, args.algorithm, args.randomSeed, epsilon, T, REG, sep=', ')
	
elif args.algorithm == 'thompson-sampling':
	REG = thompson(k,T)
	print(args.instance, args.algorithm, args.randomSeed, epsilon, T, REG, sep=', ')

