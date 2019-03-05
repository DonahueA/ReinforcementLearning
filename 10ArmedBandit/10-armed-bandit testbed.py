import random
import matplotlib.pyplot as plt

class Bandit:
	def __init__(self, k):
		self.arms = [random.gauss(0, 1) for _ in range(k)]

	def reset(self):
		self.arms = [random.gauss(0, 1) for _ in self.arms]

	def pull(self, k):
		assert(k>=0 and k<len(self.arms))
		return (k, self.arms[k] + random.gauss(0,1))

	def get_actions(self):
		return len(self.arms)

class GreedyAgent:
	def __init__(self, actions):
		self.actions = actions
		self.Q = [0 for _ in range(actions)] 
		self.N = [0 for _ in range(actions)] #Number of times each action has been chosen

	def reset(self):
		self.Q = [0 for _ in range(self.actions)] 
		self.N = [0 for _ in range(self.actions)] 

	def choose_action(self):
		max_reward = self.Q[0]
		max_index = 0

		for index, expected_reward in enumerate(self.Q): #Simple argmax
			if expected_reward > max_reward:
				max_reward = expected_reward
				max_index = index

		return max_index

	def learn(self, observation):
		action, reward = observation
		self.N[action] += 1
		self.Q[action] = self.Q[action] + 1/self.N[action] * (reward-self.Q[action])

	def __str__(self):
		return "Greedy Agent"


class OptimisticGreedyAgent(GreedyAgent):
	def __init__(self, actions):
		self.actions = actions
		self.Q = [5 for _ in range(actions)] 
		#High initial value so all values get explored at least once.
		self.N = [0 for _ in range(actions)]
	def reset(self):
		self.Q = [5 for _ in range(self.actions)] 
		self.N = [0 for _ in range(self.actions)] 
	def __str__(self):
			return "Optimistic Greedy Agent"

class EpsilonGreedyAgent(GreedyAgent):
	def __init__(self, actions, e):
		self.actions = actions
		self.Q = [0 for _ in range(actions)]
		self.N = [0 for _ in range(actions)]
		self.e = e

	def choose_action(self):
		#Epsilon Agent has an e chance of choosing a random action.
		if(random.random() < self.e):
			return random.randint(0, self.actions-1)

		max_reward = self.Q[0]
		max_index = 0
		for index, expected_reward in enumerate(self.Q): 
			if expected_reward > max_reward:
				max_reward = expected_reward
				max_index = index
		return max_index

	def __str__(self):
		return "Epsilon Greedy, e = " + str(self.e)



env = Bandit(10)
g_agent = GreedyAgent(env.get_actions())
og_agent = OptimisticGreedyAgent(env.get_actions())
eg_agent = EpsilonGreedyAgent(env.get_actions(), 0.1)
eg_agent2 = EpsilonGreedyAgent(env.get_actions(), 0.01)
agents = [g_agent, og_agent, eg_agent, eg_agent2]

plt.plot()
for num, agent in enumerate(agents):
	env = Bandit(10)
	all_rewards = [0] * 1000
	for j in range(2000):
		agent.reset()
		env.reset()
		for i in range(1000):
			choice, reward = env.pull(agent.choose_action())
			if j != 0:
				all_rewards[i] = all_rewards[i] + 1/j * (reward- all_rewards[i])
			else:
				all_rewards[i] = reward
			agent.learn((choice, reward))


	plt.plot(range(1,1001), all_rewards)

plt.ylim(0,2)
plt.legend([str(agent) for agent in agents], loc='upper left')
plt.show()

	

