### MDP Value Iteration and Policy Iteration

import numpy as np
from riverswim import RiverSwim

np.set_printoptions(precision=3)

def bellman_backup(state, action, R, T, gamma, V):
    """
    Perform a single Bellman backup.

    Parameters
    ----------
    state: int
    action: int
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)
    gamma: float
    V: np.array (num_states)

    Returns
    -------
    backup_val: float
    """
    backup_val = None
    ############################
    ### START CODE HERE ###
    #backup_val = R + gamma * np.sum(T * V)
    Q = np.zeros((2,2))
    Q = R + gamma * np.sum(T * V, axis=2)
    backup_val = Q[state, action]
    ### END CODE HERE ###
    ############################

    return backup_val

def policy_evaluation(policy, R, T, gamma, tol=1e-3):
    """
    Compute the value function induced by a given policy for the input MDP
    Parameters
    ----------
    policy: np.array (num_states)
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)
    gamma: float
    tol: float

    Returns
    -------
    value_function: np.array (num_states)
    """
    num_states, _ = R.shape
    value_function = None
    
    ############################
    ### START CODE HERE ###
    #print("policy:", policy)
    #print("gamma:", gamma)
    #print("T:", T)
    #print("R:", R)
    #print("tol:", tol)
    V = np.zeros(num_states)
    #tol = 1e-3
    diff = np.inf
    while(diff > tol):
        Q = R + gamma * np.sum(T * V, axis=2)
        if policy.ndim == 1:
            policy_one_hot = np.zeros_like(Q)
            policy_one_hot[np.arange(Q.shape[0]), policy] = 1
            policy = policy_one_hot
        V_new = np.sum(policy * Q, axis=1)
        diff = np.max(np.abs(V_new - V))
        V = V_new
    value_function = V
    ### END CODE HERE ###
    ############################
    return value_function


def policy_improvement(R, T, V_policy, gamma):
    """
    Given the value function induced by a given policy, perform policy improvement
    Parameters
    ----------
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)
    V_policy: np.array (num_states)
    gamma: float

    Returns
    -------
    new_policy: np.array (num_states)
    """
    num_states, num_actions = R.shape
    new_policy = None

    ############################
    ### START CODE HERE ###
    Q = R + gamma * np.sum(T * V_policy, axis=2)
    new_policy = np.argmax(Q, axis=1)
    ### END CODE HERE ###
    ############################
    return new_policy


def policy_iteration(R, T, gamma, tol=1e-3):
    """Runs policy iteration.

    You should call the policy_evaluation() and policy_improvement() methods to
    implement this method.
    Parameters
    ----------
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)

    Returns
    -------
    V_policy: np.array (num_states)
    policy: np.array (num_states)
    """
    num_states, _ = R.shape
    V_policy = None
    policy = None
    ############################
    ### START CODE HERE ###
    policy = np.random.randint(0, 1, size=num_states)
    diff = np.inf
    while(diff > 0):
        V_policy = policy_evaluation(policy, R, T, gamma, tol=1e-3)
        new_policy = policy_improvement(R, T, V_policy, gamma)
        diff = np.linalg.norm(new_policy - policy, ord=np.inf)
        policy = new_policy
    ### END CODE HERE ###
    ############################
    return V_policy, policy


def value_iteration(R, T, gamma, tol=1e-3):
    """Runs value iteration.
    Parameters
    ----------
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)

    Returns
    -------
    value_function: np.array (num_states)
    policy: np.array (num_states)
    """
    num_states, num_actions = R.shape
    value_function = None
    policy = None
    ############################
    ### START CODE HERE ###
    V = np.zeros(num_states)
    diff = np.inf
    while diff >= tol:
            V_new = R + gamma * np.sum(T * V, axis=2)
            policy = np.argmax(V_new, axis=1)
            BV = np.max(V_new, axis=1)
            diff = np.linalg.norm(BV - V, ord=np.inf)
            V = BV
    value_function = V
    policy = policy
    ### END CODE HERE ###
    ############################
    return value_function, policy


# Edit below to run policy and value iteration on different configurations
# You may change the parameters in the functions below
if __name__ == "__main__":
    SEED = 1234

    RIVER_CURRENT = 'STRONG'
    assert RIVER_CURRENT in ['WEAK', 'MEDIUM', 'STRONG']
    env = RiverSwim(RIVER_CURRENT, SEED)

    R, T = env.get_model()
    discount_factor = 0.92

    print("\n" + "Current River Current: " + RIVER_CURRENT)
    print("\n" + str(discount_factor * 100), "% Discount Factor")
    print("\n" + "-" * 25 + "\nBeginning Policy Iteration" + "\n" + "-" * 25)

    V_pi, policy_pi = policy_iteration(R, T, gamma=discount_factor, tol=1e-3)
    print(V_pi)
    print([['L', 'R'][a] for a in policy_pi])

    print("\n" + "-" * 25 + "\nBeginning Value Iteration\n" + "-" * 25)

    V_vi, policy_vi = value_iteration(R, T, gamma=discount_factor, tol=1e-3)
    print(V_vi)
    print([['L', 'R'][a] for a in policy_vi])