import numpy as np
import torch
import random
'''
The function is to deal the softmax data: 
if data is smallest, the data is set to 0. 
and, the sum of smallest value is average to others 
'''

def deal_mindata(data_array):
    m, n = data_array.shape
    for i in range(m):
        data = data_array[i]
        a_min = np.min(data)
        min_index = np.where(data == a_min)
        min_sum = float(len(min_index[0]) * a_min)
        data += min_sum / float((n - len(min_index[0])))
        data[min_index] = 0
        data_array[i] = data
    return data_array


'''
random seed
'''
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

'''
write
'''
def writetxt(result_list,save_file):
    with open(save_file,"w") as f:
        for i in result_list:
            #print(i)
            f.write('''{} {} {}'''.format(i[0],i[1],i[2]))
            f.write("\n")
    f.close()

'''
It is used to watch the parameter of grad.
Example: watch the loss grad
    loss.register_hook(save_grad('loss'))
    loss.backward()
    print(grads['loss'])
'''
grads = {}
def save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook
 