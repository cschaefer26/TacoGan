import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph._shortest_path import dijkstra
from scipy.sparse.csr import csr_matrix

from text.text_cleaner import basic_cleaners
from text.tokenizer import Tokenizer
from utils.config import Config

target = np.load('/tmp/target_28000.npy')[:72]
pred = np.load('/tmp/pred_28000.npy')

cfg = Config.load('config.yaml')
tokenizer = Tokenizer(basic_cleaners, cfg.symbols)

target_len = target.shape[0]
pred_len = pred.shape[0]

pred_max = np.zeros((pred_len, target_len))

white_ind = cfg.symbols.index(' ')
#for i in range(len(target)):
#    if target[i] == white_ind:
#        target[i] = 0

#pred[:, 0] = pred[:, white_ind]

for i in range(pred_len):
    weight = 1. - pred[i, target]
    pred_max[i] = weight

def to_node_index(i, j, cols):
    return cols * i + j

def from_node_index(node_index, cols):
    return node_index // cols, node_index % cols

def to_adj_matrix(mat):
    rows = mat.shape[0]
    cols = mat.shape[1]

    row_ind = []
    col_ind = []
    data = []

    for i in range(rows):
        for j in range(cols):

            node = to_node_index(i, j, cols)

            if j < cols - 1:
                right_node = to_node_index(i, j + 1, cols)
                weight_right = mat[i, j + 1]
                row_ind.append(node)
                col_ind.append(right_node)
                data.append(weight_right)

            if i < rows -1:
                bottom_node = to_node_index(i + 1, j, cols)
                weight_bottom = mat[i + 1, j]
                row_ind.append(node)
                col_ind.append(bottom_node)
                data.append(weight_bottom)

    #print(f'max row_ind {max(row_ind)} max col_ind {max(col_ind)} dim {ro}')
    adj_mat = coo_matrix((data, (row_ind, col_ind)), shape=(rows * cols, rows * cols))
    return adj_mat.tocsr()

adj_matrix = to_adj_matrix(pred_max)

dist_matrix, predecessors = dijkstra(csgraph=adj_matrix, directed=True, indices=0, return_predecessors=True)

path = []
pr_index = predecessors[-1]
while pr_index != 0:
    path.append(pr_index)
    pr_index = predecessors[pr_index]
path.reverse()
cols = pred_max.shape[1]

result = []
for node_index in path:
    i, j = from_node_index(node_index, cols)
    letter = tokenizer.decode([target[j]])
    pred_letter = tokenizer.decode([np.argmax(pred[i], axis=-1)])
    print(f'{i} {j} {letter} {pred_letter} {pred_max[i, j]}')
    result.append(j)

#print(result)

indices = []
for r in result:
    indices.append(target[r])


print(tokenizer.decode(indices))
print()
max_indices = np.argmax(pred, axis=-1)
print(tokenizer.decode(max_indices))
print()
print(tokenizer.decode(target))
#print(tokenizer.decode(result))

#print(dist_matrix)
#print(predecessors)
#print(adj_matrix)

#print(adj_matrix)
#print(pred_max[:10, :10])

#print(target)
#print(tokenizer.decode(target.tolist()))
#print(pred)