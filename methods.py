def compute_mse(y_tr,tx_tr,weight):
    return np.linalg.norm(y_tr-tx_tr.dot(np.array(weight)))/len(y_tr)
    
