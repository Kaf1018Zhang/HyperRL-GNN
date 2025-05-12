import random

class SimpleRoutingController:
    def __init__(self, candidates):
        self.candidates = candidates
        self.best_choice = {}

    def sample_random(self):
        return random.choice(self.candidates)

    def select_module(self, dataset_name):
        return self.best_choice.get(dataset_name, self.candidates[0])

    def set_best(self, dataset_name, module_name):
        self.best_choice[dataset_name] = module_name
