import math
import random
from typing import List, Optional, Self, Dict
from abc import ABC, abstractmethod


class Solution(ABC):
    @abstractmethod
    def try_evolve(self) -> Optional[Self]:
        ...

    @abstractmethod
    def get_score(self) -> float:
        ...


class Hyperparams(ABC):
    @abstractmethod
    def step(self) -> None:
        ...


class Algorithm(ABC):
    @abstractmethod
    def step(self, hyperparams: Hyperparams, solution: Solution) -> Solution:
        ...


class IterativeHyperparams(Hyperparams):
    def __init__(self, score_decay: float = 1.0, decay_cooling_rate: float = 1.0):
        self.score_decay = score_decay
        self.decay_cooling_rate = decay_cooling_rate

    def step(self) -> None:
        self.score_decay = min(1.0, self.score_decay*self.decay_cooling_rate)


class IterativeAlgorithm(Algorithm):
    def __init__(self, *args):
        self.current_decay = 1.0

    def step(self, hyperparams: IterativeHyperparams, solution: Solution) -> Solution:
        old_score = solution.get_score() * self.current_decay
        self.current_decay *= hyperparams.score_decay
        new_solution = solution.try_evolve()
        if new_solution is None:
            return solution
        new_score = new_solution.get_score()
        if new_score > old_score:
            self.current_decay = 1.0
            return new_solution
        return solution

class BestOfKHyperparams(Hyperparams):
    def step(self) -> None:
        pass

class BestOfKAlgorithm(Algorithm):
    def __init__(self, init_solution: Solution, *args):
        self.init_solution = init_solution

    def step(self, hyperparams: Hyperparams, solution: Solution) -> Solution:
        old_score = solution.get_score()
        new_solution = self.init_solution.try_evolve()
        if new_solution is None:
            return solution
        new_score = new_solution.get_score()
        if new_score > old_score:
            return new_solution
        return solution

class SimulatedAnnealHyperparams(IterativeHyperparams):
    def __init__(
        self,
        temperature: float = 0.5,
        cooling_rate: float = 0.9,
    ):
        self.temperature = temperature
        self.cooling_rate = cooling_rate

    def step(self) -> None:
        self.temperature *= self.cooling_rate

    @classmethod
    def from_iter_count(cls, iter_count: int) -> "SimulatedAnnealHyperparams":

        t_0 = 0.3
        t_target = 1e-3
        try:
            cooling_rate = float((t_target/t_0) ** (1 / iter_count))
        except Exception:
            cooling_rate = 0.95
        cooling_rate = min(0.9999, max(0.5, cooling_rate))

        return cls(temperature=t_0, cooling_rate=cooling_rate)


class SimulatedAnneal(IterativeAlgorithm):
    def __init__(self, *args, rng: Optional[random.Random] = None):
        super().__init__()
        self.rng = rng or random.Random()

    def step(self, hyperparams: SimulatedAnnealHyperparams, solution: Solution) -> Solution:
        current = solution

        new_solution = current.try_evolve()
        if new_solution is None:
            return current

        current_score = current.get_score()
        new_score = new_solution.get_score()
        temperature = hyperparams.temperature
        hyperparams.step()
        if new_score >= current_score:
            return new_solution

        delta = new_score - current_score
        try:
            prob = math.exp(delta / temperature)
        except OverflowError:
            prob = 0.0
        if self.rng.random() < prob:
            return new_solution
        return current

