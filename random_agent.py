import gymnasium as gym

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