from modules.routing_controller import SimpleRoutingController
from experiments.train_random_controller import train_and_evaluate
from core.reward_utils import compute_reward


def train_controller_with_reward(dataset_name, candidates, trials=5):
    controller = SimpleRoutingController(candidates)
    results = {}

    for _ in range(trials):
        choice = controller.sample_random()
        print(f"[RL Trainer] Trying strategy: {choice}")

        # Train and eval with selected module
        acc = train_and_evaluate(dataset_name, controller=controller, override_choice=choice)

        # Optional: estimate model complexity here (e.g., #params)
        complexity = 0.0
        reward = compute_reward(accuracy=acc, model_complexity=complexity, alpha=0.0)
        results[choice] = max(reward, results.get(choice, 0))

    best = max(results, key=results.get)
    controller.set_best(dataset_name, best)
    print(f"[RL Trainer] Best strategy for {dataset_name}: {best}")
    return controller
