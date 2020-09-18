# mountaincar-v0
Simple reinforcement learning solution to the [mountaincar-v0](https://gym.openai.com/envs/MountainCar-v0/) environment.

## Synopsis
The basic algorithm:
1. Initialize Q(s1,s2,a) to small random values.
2. Observe current state.
3. Choose an action based on exploration strategy.
4. Execute action and observe reward and new state.
5. Update Q(s1,s2,a) based on the q-learning algorithm:  
   <i>Q(s1,s2,a) = (1-alpha)Q(s1,s2,a) + alpha(reward + gamma(maxQ(nextstate, all actions))</i>  
   <i>alpha</i> = learning rate  
   <i>gamma</i> = discount rate  
6. Repeat 2-5 until convergence.

My particular implementation would be better the addition of an epsilon-greedy exploration strategy where epsilon decays over time. As long as epsilon is small this works fine though.
