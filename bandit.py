import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

class Bandit:

    def __init__(self,
                 mean=0,
                 sd=1):
        self.mean = mean
        self.sd = sd

    def run(self):
        return np.random.normal(self.mean,
                                self.sd,
                                1)

class Agent:

    def __init__(self, bandit_nb):
        self.bandit_nb = bandit_nb
        self.bandits = [Bandit(5 * np.random.random_sample())
                        for _ in range(bandit_nb)]

    def play(self, game_nb, step_nb, epsilon):

        reward_by_step = np.zeros(step_nb)
        for game in range(game_nb):
            self.Q = np.zeros(self.bandit_nb)
            self.N = np.zeros(self.bandit_nb)
            for step in range(step_nb):
                r = np.random.random_sample()
                if r > epsilon:
                    bandit_index = np.argmax(self.Q)
                else:
                    bandit_index = np.random.randint(self.bandit_nb)
                reward = self.bandits[bandit_index].run()
                reward_by_step[step] += reward

                self.N[bandit_index] += 1
                self.Q[bandit_index] += (reward - self.Q[bandit_index])/self.N[bandit_index]

        average_reward = reward_by_step/game_nb
        return average_reward

def main():
    agent =  Agent(10)
    game_nb = 2000
    step_nb = 2000
    avr_01 = agent.play(game_nb, step_nb, 0.1)
    avr_005 = agent.play(game_nb, step_nb, 0.05)
    avr_001 = agent.play(game_nb, step_nb, 0.01)
    avr_0 = agent.play(game_nb, step_nb, 0)

    plt.plot(avr_01, label="0.1")
    plt.plot(avr_005, label="0.05")
    plt.plot(avr_001, label="0.01")
    plt.plot(avr_0, label="0")
    plt.legend(loc='lower right', frameon=False)
    plt.show()

if __name__ == "__main__":
    main()
