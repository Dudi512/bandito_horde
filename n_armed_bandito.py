import numpy as np

class BanditMachine:
    def __init__(self, mean, std_deviation):
        self.mean = mean
        self.std_deviation = std_deviation

    def play(self):
        return np.random.normal(self.mean, self.std_deviation)

class Casino:
    def __init__(self, num_machines):
        self.num_machines = num_machines
        self.bandit_machines = [BanditMachine(np.random.normal(0, 1), np.random.uniform(0.5, 1.5)) for _ in range(num_machines)]

    def ucb_select_machine(self, num_plays, num_trials):
        ucb_values = np.zeros(self.num_machines)

        for i in range(self.num_machines):
            mean = 0
            if num_plays[i] > 0:
                mean = num_trials[i] / num_plays[i]
            uncertainty = np.sqrt(2 * np.log(sum(num_plays)) / num_plays[i])
            ucb_values[i] = mean + uncertainty

        return np.argmax(ucb_values)

def main():
    N = 100
    M = 50
    num_machines = N
    num_guests = M
    num_plays = np.zeros(num_machines)
    num_trials = np.zeros(num_machines)

    casino = Casino(num_machines)

    for guest in range(num_guests):
        for i in range(num_machines):
            reward = casino.bandit_machines[i].play()
            num_plays[i] += 1
            num_trials[i] += reward

    for _ in range(1000):
        machine_index = casino.ucb_select_machine(num_plays, num_trials)
        reward = casino.bandit_machines[machine_index].play()
        num_plays[machine_index] += 1
        num_trials[machine_index] += reward

    optimal_machine_index = np.argmax(num_trials / num_plays)
    print("Najlepsza maszyna do gry to:", optimal_machine_index)

if __name__ == "__main__":
    main()
