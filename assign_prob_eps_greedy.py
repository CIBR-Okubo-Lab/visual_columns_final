#!/usr/bin/env python
import argparse
import concurrent.futures
import copy
from datetime import datetime
from scipy.sparse import csr_matrix
import numpy as np
import os
import pickle
from tqdm import tqdm
import utils
import wandb


def process_epsilon(X, W, W_sparse, f_curr, d_neurons_type, eps, T, seed, decay_rate, decay_iter, patient_period, max_iter=200):
    """
    Subprocess to compute the best assignment for current seed and epsilon. 

    Args:
        X (np.array): neuron-column assigment
        W (np.array): synaptic connectivity matrix
        W_sparse (csr_matrix): sparse matrix version of W
        f_curr (float): current value of the objective function
        d_neurons_type (dict): cell type index -> list of neuron indices that belong to that cell type
        eps (float): epsilon for random selection
        T (int): number of types
        seed (int): random seed
        decay_rate (float): the rate of decay for epsilon
        decay_iter (int): starting iteration for epsilon decay
        max_iter (int): maximum iterations, default is 200
    
    Returns:
        f_history (list, int): history of objective functions
        diff_history (list, int): history of increase in objective functions
        X_max (np.array): assignment matrix with highest objective function
        f_max (int): highest objective function
        eps (float): epsilon
        seed (int): random seed

    """
    np.random.seed(seed)

    # wandb configuration
    config = {
        'decay_rate': decay_rate, 
        'f_init': f_curr,
    }
    run = wandb.init(project = "epsilon-greedy-search", config=config, name=f"run_{datetime.now().strftime('%Y%m%d-%H%M%S.%f')}_{eps}_{seed}")
    
    f_history = []
    diff_history = []
    X_max = X
    f_max = f_curr
    waiting_period = 0
    
    for i in range(max_iter):
        diff_per_type = []
        X_new_per_type = []

        for t in tqdm(range(T)):
            diff, X_new = utils.diff_given_type(X, W, W_sparse, t, f_curr, d_neurons_type)
            diff_per_type.append(diff)
            X_new_per_type.append(X_new)
        diff_history.append(dict(zip(range(T), diff_per_type)))
        
        # perform the reassignment
        max_diff = max(diff_per_type)
        print(max_diff)
        if np.random.uniform(0, 1) < eps:
            type_idx = np.random.choice(range(T))
        else: 
            type_idx = diff_per_type.index(max(diff_per_type))  # cell type that has the max improvement in the objective function

        X = copy.deepcopy(X_new_per_type[type_idx])
        f_curr = utils.f(X, W_sparse)  # update the current objective function
        print(f_curr)

        # save assignment matrix with higher objective function
        if f_curr > f_max:
            f_max = f_curr
            X_max = X
        f_history.append(f_curr)

        # termination condition
        if max_diff == 0:
            waiting_period += 1
            if waiting_period >= patient_period: 
                break
        
        if i >= decay_iter:
            eps *= (1-decay_rate)

        run.log({'iteration': i, 'objective': f_curr, "diff_max": max_diff, "random seed":seed, "eps": eps})
    run.finish()
    return f_history, diff_history, X_max, f_max, eps, seed

def main(args): 
    
    # original data path
    path_neuron = args.data_path + '/ol_columns.csv'
    path_synapse = args.data_path + '/ol_connections.csv'
    
    # load arguments from arg parser
    eps = args.epsilon
    max_iter = args.max_iter
    num_seed = args.num_seed
    patient_period = args.patient_period


    ## read data
    df_neurons, N, K, T = utils.load_neurons(path_neuron)
    d_neuron, d_type = utils.define_dict(df_neurons, N, T)
    X, Y = utils.define_X_Y(df_neurons, d_type, N, K, T)
    W = utils.load_synapses(path_synapse, d_neuron, N)
    W_sparse = csr_matrix(W)

    d_neurons_type = utils.neuron_list_per_type(Y)

    save_path = args.save_path

    decay_rate = args.decay_rate
    decay_iter = args.decay_iter

    if not args.orig_obj: 
        assignment_result = args.assignment_result+ "/X_" + args.cur_obj + ".npy"
        X = np.load(assignment_result, allow_pickle=True)
    f_curr = utils.f(X, W_sparse) 

    result = {}
    all_f_max = {}
    all_X = {}

    seeds = np.random.randint(0, 10000, num_seed)
    # seeds = [440]

    # launch jobs for each random seed in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_seed) as executor:
        futures = [
            executor.submit(process_epsilon, X, W, W_sparse, f_curr, d_neurons_type, eps, T, seed, decay_rate, decay_iter, patient_period, max_iter=max_iter)
            for seed in seeds
        ]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            f_history, diff_history, X_max,f_max, eps, seed = future.result()
            result["f_history"] = f_history
            result["diff_history"] = diff_history
            result["X_max"] = X_max
            result["f_max"] = f_max
            all_f_max[seed] = f_max
            all_X[seed] = X_max

            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)
            with open(f"{save_path}/eps_{eps}_seed_{seed}_fmax_{int(f_max)}_result.pkl", "wb+") as f:
                pickle.dump(result, f)
    
    # save final result
    max_key = max(all_f_max, key=all_f_max.get) 
    X_max_final = all_X[max_key]
    num_assigned = np.where(X_max!=0)[0].shape[0]
    np.save(save_path+f"/X_{int(all_f_max[max_key])}_{num_assigned}.npy", X_max_final)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", 
        "-d", 
        help="Path to data folder", 
        default=r"../data"
    )
    parser.add_argument(
        "--assignment_result", 
        "-ap", 
        help="Path to assignment result folder", 
        default=r"../results"
    )
    parser.add_argument(
        "--orig_obj", 
        action="store_true",
        help = "Flag indicating to use the original assignment to start the optimization"
    )
    parser.add_argument(
        "--cur_obj", 
        "-f", 
        help="Objective value to start the optimization", 
    )

    parser.add_argument(
        "--epsilon",
        "-e",  
        type = float,
        help = "epsilon", 
        default = 0.1
    )

    parser.add_argument(
        "--save_path", 
        "-sp", 
        help="Path to save the results", 
        default="../eps_results"
    )

    parser.add_argument(
        "--max_iter", 
        help = "maximum iteration", 
        type=int, 
        default = 200
    )

    parser.add_argument(
        "--num_seed", 
        "-s", 
        help="Number of random seed to test for each epsilon", 
        type=int, 
        default=10
    )

    parser.add_argument(
        "--decay_rate", 
        "-dr", 
        help="epsilon decay rate", 
        type=float, 
        default=0.02
    )

    parser.add_argument(
        "--decay_iter", 
        "-di", 
        help="start to decay epsilon after this iteration", 
        type=int, 
        default=200
    )

    parser.add_argument(
        "--patient_period", 
        "-pp", 
        help="patient period before stopping the process, after the max diff is 0", 
        type=int, 
        default=30
    )

    args = parser.parse_args()
    wandb.login()
    # wandb.login(key="") # add key to login to wandb
    main(args)
