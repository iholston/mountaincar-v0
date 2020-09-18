import gym
import numpy as np
gym.logger.set_level(40) # Might need this to remove logger warning https://stackoverflow.com/questions/60149105/userwarning-warn-box-bound-precision-lowered-by-casting-to-float32

class mountaincar_driver:
    def __init__(self, alpha, gamma, epsilon):
        print("------------------------------------")
        print("Initializing Training...")
        print("Learning Rate: {}".format(alpha))
        print("Future Discount: {}".format(gamma))
        print("Random Action Rate: {}".format(epsilon))
        print("(Average reward per 100 episodes)")
        print("------------------------------------")
        self.qtable = np.random.uniform(low = -1, high = 1, size = (19, 15, 3)) 
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.env = gym.make('MountainCar-v0')

    # Round state[0] to .1 and multiply by 10, round state[1] to .01 and multiply by 100
    def descretize_state(self, state):
        position = int(round(state[0], 1) * 10)
        velocity = int(round(state[1], 2) * 100)
        return [position, velocity]
 
    def train(self, episodes):
        ave_reward = 0
        last100sucessrate = 0
        for i in range(episodes):

            # Initialize params
            done = False
            reward = 0
            total_reward = 0
            state = self.descretize_state(self.env.reset())
            
            # Run episode
            while not done:
                
                # Determine action
                if np.random.random() < self.epsilon: # random action
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.qtable[state[0], state[1]])

                # Get reward and Next state
                next_state, reward, done, _ = self.env.step(action)
                next_state = self.descretize_state(next_state)

                # update qtable
                old_value = self.qtable[state[0], state[1], action]
                nextmax = np.max(self.qtable[next_state[0], next_state[1]])
                self.qtable[state[0], state[1], action] = (1-self.alpha)*old_value + self.alpha*(reward + self.gamma*nextmax)
               
                # update variables
                total_reward += reward
                state = next_state


            # Keep track of average reward over 100 episodes
            ave_reward += total_reward
            if (i + 1) % 100 == 0:
                average = ave_reward/100
                ave_reward = 0
                if i < 999:
                    print("Episode {}.  Average Reward: {}".format(i+1, average))
                else:
                    print("Episode {}. Average Reward: {}".format(i+1, average))

            # Track successful completions in last 100 episodes
            if i > (episodes - 100):
                if total_reward > -200:
                    last100sucessrate += 1
                if i == episodes - 1:
                    print("\nEnvironment was successfully solved {} times in the last 100 episodes".format(last100sucessrate))
                
    

    def display(self, episodes):
        for _ in range(episodes):
            state = self.descretize_state(self.env.reset())
            done = False
            while done != True:
                self.env.render()
                action = np.argmax(self.qtable[state[0], state[1]])
                next_state, _, done, _ = self.env.step(action)
                next_state = self.descretize_state(next_state)
                state = next_state
        self.env.close()
        
# Main
alpha = .2 # learning rate
gamma = .9 # future reward discount
epsilon = .1 # random action rate
episodes = 5000

isaac = mountaincar_driver(alpha, gamma, epsilon)
isaac.train(episodes)
isaac.display(3)





