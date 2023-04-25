import numpy as np
from reinforcement.episodes import evaluate


def test(env, agent, config, episodes, attacked_policy_id):
    results, matrix = evaluate(env, agent, config, episodes, verbose=True)
    recovered_attack_rewards = results.pop(attacked_policy_id)
    other_reward = results.popitem()[1]

    # matrix = agent.matrix[attacked_policy_id]
    matrix = matrix[attacked_policy_id]
    TNR, _, _, TPR = np.true_divide(matrix, matrix.sum(axis=1)[:, None],
                                    out=np.ones_like(matrix, dtype=float),
                                    where=matrix.sum(axis=1)[:, None] > 0).flatten()
    mask = matrix.sum(axis=1) > 0
    acc = (mask*(TNR, TPR)).sum() / mask.sum()
    print(recovered_attack_rewards)
    return {
        'recovered_reward': recovered_attack_rewards,
        'recovered_reward_mean': np.mean(recovered_attack_rewards),
        'other_reward': np.mean(other_reward),
        'recovered_reward_std': recovered_attack_rewards,
        'recovered_reward_std_mean': np.mean(recovered_attack_rewards),
        'other_reward_std': np.std(other_reward),
        'matrix': str(matrix).replace('\n', ' '),
        'TNR': TNR,
        'TPR': TPR,
        'balanced_accuracy': acc
    }
