import numpy as np
import random

class Bandit:
    def __init__(self, mean, std_dev):
        self.mean = mean
        self.std_dev = std_dev

    def pull_lever(self):
        return np.random.normal(self.mean, self.std_dev)

class Guest:
    def __init__(self, num_bandits, epsilon=0.1):
        self.num_bandits = num_bandits
        self.means = np.zeros(num_bandits)
        self.std_devs = np.ones(num_bandits)
        self.total_rewards = np.zeros(num_bandits)
        self.total_pulls = np.zeros(num_bandits)
        self.epsilon = epsilon

    def select_bandit(self):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_bandits - 1)
        else:
            return np.argmax(self.means)

    def update_statistics(self, bandit_idx, reward):
        self.total_pulls[bandit_idx] += 1
        self.total_rewards[bandit_idx] += reward
        self.means[bandit_idx] = self.total_rewards[bandit_idx] / self.total_pulls[bandit_idx]
        if self.total_pulls[bandit_idx] > 1:
            self.std_devs[bandit_idx] = np.sqrt(((self.total_rewards[bandit_idx] - self.means[bandit_idx]) ** 2) / (self.total_pulls[bandit_idx] - 1))

class RewardConverter:
    @staticmethod
    def convert_to_readable(value):
        abs_value = abs(value)
        if abs_value >= 1e6:
            converted_value = abs_value / 1e6
            unit = "mln"
        elif abs_value >= 1e3:
            converted_value = abs_value / 1e3
            unit = "tys"
        else:
            converted_value = abs_value
            unit = ""

        return converted_value if value >= 0 else -converted_value, unit

def main():
    N = 100  # Number of bandits
    M = 100  # Number of guests
    Tmax = 10000  # Number of iterations

    bandits = [Bandit(np.random.uniform(-10, 10), np.random.uniform(1, 5)) for _ in range(N)]
    guests = [Guest(N) for _ in range(M)]

    for t in range(Tmax):
        for guest in guests:
            bandit_idx = guest.select_bandit()
            reward = bandits[bandit_idx].pull_lever()
            guest.update_statistics(bandit_idx, reward)

    total_rewards = np.zeros(N)
    for guest in guests:
        total_rewards += guest.total_rewards
        
    total_rewards_sum = np.sum(total_rewards)
    total_rewards_rounded= np.round(total_rewards,2)
    converted_rewards = [RewardConverter.convert_to_readable(value) for value in total_rewards_rounded]
    for i, (value, unit) in enumerate(converted_rewards, 1):
      print(f"Nagroda dla bandyty {i}: {value:.2f} {unit}")


    print("Suma wszystkich wygranych dla bandytów:", np.round(total_rewards_sum,2))

if __name__ == "__main__":
    main()
    
