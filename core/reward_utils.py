reward_history = []

def compute_reward(accuracy, model_complexity=None, alpha=0.0, baseline_weight=0.9):
    if model_complexity is not None:
        raw_reward = accuracy - alpha * model_complexity
    else:
        raw_reward = accuracy

    # moving average baseline
    if reward_history:
        baseline = sum(reward_history) / len(reward_history)
    else:
        baseline = 0.0

    # Advantage = current - baseline
    advantage = raw_reward - baseline

    # update reward history
    reward_history.append(raw_reward)
    if len(reward_history) > 50:  # Limit the length
        reward_history.pop(0)

    return advantage
