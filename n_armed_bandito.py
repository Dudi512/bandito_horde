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
        self.global_best_machine = None

    def play_guest(self, guest_id, f_prawdziwy):
        best_own_machine = None
        best_global_machine = self.global_best_machine

        for i in range(self.num_machines):
            if best_own_machine is None or f_prawdziwy[guest_id][i] > f_prawdziwy[guest_id][best_own_machine]:
                best_own_machine = i
            if best_global_machine is None or f_prawdziwy[guest_id][i] > f_prawdziwy[guest_id][best_global_machine]:
                best_global_machine = i

        machine_index = np.random.choice([best_own_machine, best_global_machine])
        reward = self.bandit_machines[machine_index].play()

        return machine_index, reward

    def update_beliefs(self, guest_id, machine_index, reward, a_ij, b_ij):
        a_ij[guest_id][machine_index] += reward
        b_ij[guest_id][machine_index] += 1
        self.update_global_best(machine_index)

    def update_global_best(self, machine_index):
        if self.global_best_machine is None:
            self.global_best_machine = machine_index
        elif self.bandit_machines[machine_index].mean > self.bandit_machines[self.global_best_machine].mean:
            self.global_best_machine = machine_index

def main():
    print("Rozpoczynam symulację...")
    
    N = 100
    M = 50
    Tmax = 10000  # Ustawiamy maksymalną liczbę kroków
    num_machines = N
    num_guests = M

    print("Tworzę maszyny w kasynie...")
    f_prawdziwy = np.random.normal(0, 1, size=(num_guests, num_machines))
    a_ij = np.zeros((num_guests, num_machines))
    b_ij = np.zeros((num_guests, num_machines))

    casino = Casino(num_machines)

    t = 0  # Inicjalizujemy licznik kroków
    while t < Tmax:  # Warunek stopu - wykonuj pętlę dopóki liczba kroków jest mniejsza niż Tmax
        print(f"Krok {t+1}/{Tmax}")
        for guest_id in range(num_guests):
            machine_index, reward = casino.play_guest(guest_id, f_prawdziwy)
            casino.update_beliefs(guest_id, machine_index, reward, a_ij, b_ij)
        
        t += 1  # Inkrementujemy licznik kroków

    global_best_machine_index = np.argmax(np.mean(a_ij, axis=0))
    print("Symulacja zakończona.")
    print("Najlepsza maszyna do gry globalnie to:", global_best_machine_index)

if __name__ == "__main__":
    main()
