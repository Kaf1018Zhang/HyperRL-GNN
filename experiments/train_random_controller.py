import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.routing_controller import SimpleRoutingController
from experiments.deploy_controller import run_with_fixed_strategy
from core.reward_utils import compute_reward

def train_controller_with_reward(dataset_name, candidates, trials=5):
    """
    Randomly sample different GNN module combinations multiple times,
    calculate their accuracy on the validation set/test set and use it as reward,
    so as to select the best strategy and save it.
    """
    controller = SimpleRoutingController(candidates)
    results = {}

    for _ in range(trials):
        choice = controller.sample_random()
        print(f"[Controller Trainer] Trying strategy: {choice}")

        # Use this strategy to run a complete training + validation + test, and get test_acc
        acc = run_with_fixed_strategy(dataset_name, controller, override_choice=choice)

        # Complexity (such as params) indicators can be added here
        complexity = 0.0
        reward = compute_reward(accuracy=acc, model_complexity=complexity, alpha=0.0)
        results[choice] = max(reward, results.get(choice, 0))

    # Find the strategy with the highest score
    best = max(results, key=results.get)
    controller.set_best(dataset_name, best)
    print(f"[Controller Trainer] Best strategy for {dataset_name}: {best}")
    return controller

if __name__ == "__main__":
    candidates = ["GCN", "GAT", "GIN"]

    print("=== Training strategy controller on PROTEINS ===")
    controller_proteins = train_controller_with_reward("PROTEINS", candidates)

    print("=== Training strategy controller on ENZYMES ===")
    controller_enzymes = train_controller_with_reward("ENZYMES", candidates)
