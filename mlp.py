import numpy as np
import numpy.typing as npt

from typing import List

from main import game_loop


class MLP:
    def __init__(self, layers: npt.NDArray[np.int64], l_bound: int, u_bound: int):
        self.layers = layers
        self.l_bound = l_bound
        self.u_bound = u_bound

        self.name_layers: List[str] = ["input_layer"]

        self.name_layers.extend(
            ["inter_layer_" + str(i) for i in range(len(self.layers) - 2)]
        )
        self.name_layers.append("output_layer")

        self.model_dict = {
            self.name_layers[l]: np.full([self.layers[l], self.layers[l + 1]], np.nan)
            for l in range(len(self.layers) - 1)
        }
        self.model_dict[self.name_layers[-1]] = np.full(
            [self.layers[-2], self.layers[-1]], np.nan
        )

        self.fitness: float = -9999.0

        for layer_idx in range(len(self.layers) - 1):
            self._generate_layer(layer_idx)

    def calc_fitness(self):
        self.fitness: float = game_loop(100, self)
        return float(self.fitness)

    def mutate(self, high: float, low: float):
        for layer in self.model_dict:
            self.model_dict[layer] += np.random.uniform(
                low, high, size=self.model_dict[layer].shape
            )

    def _relu(self, x: npt.ArrayLike):
        return np.maximum(0, x)

    def _sigmoid(self, x: npt.ArrayLike):
        pass

    def _generate_layer(self, layer: int):
        self.model_dict[self.name_layers[layer]] = np.random.uniform(
            low=self.l_bound,
            high=self.u_bound,
            size=[self.layers[layer], self.layers[layer + 1]],
        )
        self.model_dict[self.name_layers[-1]] = np.random.uniform(
            low=self.l_bound, high=self.u_bound, size=[self.layers[-2], self.layers[-1]]
        )

    def forward_pass(self, x: npt.ArrayLike):
        out = self._relu(np.dot(x, self.model_dict[self.name_layers[0]]))
        for layer in range(1, len(self.layers) - 1):
            out = self._relu(
                np.dot(out, self.model_dict[self.name_layers[layer]])
            ).flatten()
            out[1] = out[1] / 1000
        return out
