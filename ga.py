import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
import pickle
import random
import copy
import numpy as np
import numpy.typing as npt
from typing import Annotated, Protocol, Callable, List
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

from mlp import MLP
from main import game_loop


random.seed(42)
np.random.seed(42)

Vector3x1 = Annotated[npt.NDArray[np.float64], "Shape(2, 1)"]

def sphere_function(x: Vector3x1):
    r"""
    https://www.sfu.ca/~ssurjano/spheref.html

    Find a solution that minimizes the Sphere function.

    Parameters
    ----------
    x : npt.ArrayLike
        Candidate solution for $\arg \min_x f$ with $x \in \mathbb{R}^d$ where d is the problem's dimension.

    Returns
    ----------
    f(X) : npt.ArrayLike 
        The objective, $f$ evaluated at $x$
    """

    x = x.flatten()

    return float(np.sum(x**2))

class Solution:
    def __init__(self, high: float, low: float, d: int):
        self.high_bound = high
        self.low_bound = low
        self.dim = d

        self.body = np.random.uniform(self.low_bound, self.high_bound, size=[self.dim])
        self.fitness = None

    def calc_fitness(self, f: Callable):
        self.fitness = f(self.body)

    def mutate(self, l_mutation: float, h_mutation: float):
        self.body += np.random.uniform(l_mutation, h_mutation, size=[self.dim,])
    
class GeneticAlgorithm(Solution):
    def __init__(self, f: Callable, n_pop: int, d: int, f_bias: float):
        self.f = f
        self.npop = n_pop
        self.dim = d
        self.fitness_bias = int(f_bias * n_pop) 
        self.n_offspring = 2 * int(f_bias * n_pop) 

        random_layers = np.random.randint(3, 6, size=random.randint(3, 6))
        random_layers[0] = 12
        random_layers[-1] = 2

        self._pop = [MLP(random_layers, -100, 100) for _ in range(n_pop)]
        
        # calculate fitness and sort the population
        for pop in self._pop:
            pop.calc_fitness()
        self._pop = sorted(self._pop, key=lambda x: x.fitness, reverse=True)
        
    def crossover(self, n: int) -> List[Solution]:
        par1 = self._pop[0]
        par2 = self._pop[1]
        offspring = copy.deepcopy(par1)

        offsprings = []
        for _ in range(n):
            random_layer = random.choice(offspring.name_layers)
            random_parent = random.choice([par1, par2])

            offspring.model_dict[random_layer] = random_parent.model_dict[random_layer]

            offsprings.append(offspring)

        # overwritting to return only n offsprings of the list
        return offsprings

    def run(self):
        results = {"fitness": [], "solution": Solution}

        curr_best_candidate = self._pop[0]

        iter = 0
        for _ in range(30):
            for pop in self._pop:
                pop.mutate(-5/(iter+1), 5/(iter+1))
    
            # [-] n population removed due to unfitness
            self._pop = self._pop[:(self.npop - self.fitness_bias)]

            # [+] n offspring added for the next round
            #? insert them directly to population?
            offpsrings = self.crossover(n=self.n_offspring)

            self.next_pop = self._pop[:2] + offpsrings

            # calculate fitness and sort the population
            for pop_idx, pop in enumerate(self.next_pop):
                pop.calc_fitness()

            self.next_pop = sorted(self.next_pop, key=lambda x: x.fitness, reverse=True)
            for idx, pop in enumerate(self.next_pop):
                print(f"Pop idx {idx} with fitness -> {pop.fitness}")
            curr_best_candidate = self.next_pop[0]

            # propagate the population to the next epoch
            self._pop = self.next_pop

            results["fitness"].append(curr_best_candidate.fitness)
            print(f"Iteration {iter} with fitness {results["fitness"]}")

            iter += 1
        results["solution"] = curr_best_candidate
        
        return results


if __name__ == "__main__":
    ga = GeneticAlgorithm(sphere_function, 50, 2, 0.3)
    res = ga.run()

    with open('data.pkl', 'wb') as f:
        pickle.dump(res["solution"], f)

    with open('data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    os.environ.pop("SDL_VIDEODRIVER", None)
    game_loop(5000, res["solution"])

    plt.plot(range(0, len(res["fitness"])), res["fitness"], color='blue', marker='o', markerfacecolor='red', markeredgecolor='black', markersize=4)
    plt.grid()
    plt.show()
