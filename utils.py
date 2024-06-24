import copy
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix

def load_neurons(path_neuron):
    """
    Args:
        path_neuron (str): path to CSV file that contains neuron information
    
    Returns:
        N (int): total number of neurons
        K (int): total number of columns (not including the "not assigned")
        T (int): total number of cell types
    """
    df_neurons = pd.read_csv(path_neuron, header=0)
    N = len(df_neurons)
    K = len(df_neurons['column id'].unique()) - 1 
    T = len(df_neurons['cell type'].unique())
    return df_neurons, N, K, T


def define_dict(df_neurons, N, T):
    """
    Args:
        df_neurons (pd.DataFrame)
        N (int): total number of neurons
        T (int): total number of cell types
        
    Returns:
        d_neuron (dict): cell id -> cell index mapping
        d_type (dict): cell type -> cell type index mapping
    """
    d_neuron = dict(zip(df_neurons['cell id'], range(N)))
    d_type = dict(zip(df_neurons['cell type'].unique(), range(T)))  
    return d_neuron, d_type


def define_X_Y(df_neurons, d_type, N, K, T):
    """
    Args:
        df_neurons (pd.DataFrame)
        d_type (dict): mapping between cell type and cell type idx
        N (int): total number of neurons
        K (int): total number of columns
        T (int): total number of cell types
        
    Returns:
        X (np.array): neuron assignment matrix (N, K)
        Y (np.array): neuron type matrix (N, T)
    """
    X = np.zeros((N, K))
    Y = np.zeros((N, T))

    for idx, rows in df_neurons.iterrows():
        Y[idx, d_type[rows['cell type']]] = 1
        if rows['column id'] != 'not assigned':
            col = int(rows['column id']) - 1  # subtract one to convert to 0-index
            X[idx, col] = 1
            
    return X, Y


def load_synapses(path_synapse, d_neuron, N):
    """
    Args:
        path_synapse (str): path to CSV file that contains synapse information
        d_neuron (dict): cell id -> cell index mapping
        N (int): total number of neurons
    Returns:
        W (np.array): synaptic connectivity matrix (N, N)
    
    """
    df_conn = pd.read_csv(path_synapse, header=0)
    W = np.zeros((N, N))
    for _, rows in df_conn.iterrows():
        i = rows['from cell id']
        j = rows['to cell id']
        w = rows['synapses']
        W[d_neuron[i], d_neuron[j]] = w
    return W


def f(X, W):
    """calculate the value of the objective function
    
    Args:
        X (np.array): neuron-column assigment
        W (csr_matrix): synaptic connectivity matrix
    Returns:
        result (float): total number of within-column synapses
    """
    assert isinstance(W, csr_matrix), "W needs to be sparse matrix"
    X_sparse = csr_matrix(X)
    result = (X_sparse.T @ W @ X_sparse).trace()
    return result
        
    
def neuron_list_per_col(X):
    """creates a dictionary to get a list of neurons for each column
    
    Args:
        X (np.array): neuron-column assignment matrix (N, K)
        
    Returns:
        d_neurons_col (dict): column number -> list of neuron indices that belong to that column
    """
    d_neurons_col = {}
    for k in range(X.shape[1]):
        neuron_list = np.where(X[:, k] == 1)[0].tolist()
        d_neurons_col[k] = neuron_list

    return d_neurons_col


def neuron_list_per_type(Y):
    """creates a dictionary optimmized to get a list of neurons for each type
    
    Args:
        Y (np.array): neuron-type assignment matrix (N, T)
        
    Returns:
        d_neurons_type (dict): cell type index -> list of neuron indices that belong to that cell type
    """
    d_neurons_type = {}
    for t in range(Y.shape[1]):
        neuron_list = np.where(Y[:, t] == 1)[0].tolist()
        d_neurons_type[t] = neuron_list
    return d_neurons_type


def diff_given_type(X, W, W_sparse, t, f_curr, d_neurons_type):
    """Perform the optimal assignment for a given cell type t
    
    Args:
        X (np.array): neuron-column assignment matrix (N, K)
        W (np.array): synaptic connectivity matrix (N, N)
        W (csr_matrix): sparse matrix version of W
        t (int): cell type index
        f_curr (float): current value of the objective function
        d_neurons_type (dict): cell type index -> list of neuron indices that belong to that cell type
    
    Returns:
        diff (float): difference in the objective function compared to the f_curr
        X_new (np.array): new neuron-column assignment matrix (N, K)
    """
    neurons_type = d_neurons_type[t]  # get a list of neurons of a given type
    X_new = copy.deepcopy(X)  # first copy the current assignment
    X_new[neurons_type] = np.zeros(X_new[neurons_type].shape)  # unassign all the neurons of a given type first!
    d_neurons_col = neuron_list_per_col(X_new)  # create a dictionary with column number -> list of neurons
    
    f_unassign = f(X_new, W_sparse)  # calculate the objective function after unassignment (value will decrease)
    
    # create a cost matrix for the assignment problem
    cost = []
    for neuron in neurons_type:  # for each neuron of a given type
        cost_neuron = []
        for k_new in range(X.shape[1]):  # for each column
            colmates_new = d_neurons_col[k_new]
            diff = W[neuron, colmates_new].sum() + W[colmates_new, neuron].sum() - W[neuron, neuron]  # difference in t
            cost_neuron.append(diff)
        cost.append(cost_neuron)
    cost = np.array(cost)
    
    row_new, col_new = linear_sum_assignment(cost, maximize=True)  # solve the assignment problem
    diff = f_unassign + cost[row_new, col_new].sum() - f_curr
    
    # new assignment
    one_hot = np.zeros(X_new[neurons_type].shape)
    one_hot[row_new, col_new] = 1
    X_new[neurons_type] = one_hot
    
    return diff, X_new


def format_result(df_neurons, X, d_neuron): 
    """
    Args: 
        df_neurons (pd.DataFrame): dataframe of the ol_column file
        X (np.array): neuron-column assigment 
        d_neuron (dict): cell id -> cell index mapping
    
    Returns:
        df_results (pd.DataFrame): dataframe of the new assignment
    """
    df_result = df_neurons.copy()
    d_neuron_re = {v: k for k, v in d_neuron.items()}

    for i in range(X.shape[0]):
        if np.where(X[i,:] == 1)[0].size > 0:
            column_id = str(np.where(X[i,:] == 1)[0][0] + 1)
        else:
            column_id = "not assigned"
        neuron_id = d_neuron_re[i]
        df_result.loc[df_result["cell id"] == neuron_id, "column id"] = column_id 
    
    return df_result