import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

def collect_trajectory(steps=200):
    env = gym.make(id="CartPole-v1")
    state, info = env.reset()

    states = []
    for i in range(steps):
        states.append(state)

        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)

        state = next_state
        if terminated or truncated:
            break

    env.close()
    return np.array(states)


def plot_states(states):
    labels = ["Cart Position", "Cart Velocity", "Pole Angle", "Pole Angular Velocity"]
    plt.figure(figsize=(12, 8))

    for i in range(4):
        plt.subplot(4, 1, i+1)
        plt.plot(states[:, i])
        plt.ylabel(labels[i])
        plt.grid(True)

    plt.xlabel("Timestep")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    states = collect_trajectory()
    plot_states(states)