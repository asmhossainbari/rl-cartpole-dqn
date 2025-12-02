import gymnasium as gym

'''
State:
CartPole state means a vector of 4 numbers: cart position, cart velocity, pole angle, and pole angular velocity

Action:
action = 0 means move cart left and action = 1 means move cart right

Reward;
Every time pole stays up, agent receives reward from the environment

Termination:
terminated = True means episode ends due to failure or success
truncated = True means forced end like time limit exceeds

Episode:
An episode is one full run of the environment from start until failure ( pole falls -> pole angle too large or time runs out -> 500 steps)
Once episode ends, the agent restarts the environment 

Random agent has no intelligence, no strategy, no learning. Only luck determines how long the pole stays upright.
'''

def run_random_agent(episodes=5, max_steps=500, render=False):
    env = gym.make(id="CartPole-v1", render_mode="human" if render else None)
    for episode in range(episodes):
        state, info = env.reset(seed=episode)  # seed for reproducibility
        total_reward = 0

        for step in range(max_steps):
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            state = next_state

            if terminated or truncated:
                break

        print(f"Episode {episode + 1}: reward = {total_reward}")

    env.close()

if __name__ == "__main__":
    run_random_agent(episodes=5, render=True)