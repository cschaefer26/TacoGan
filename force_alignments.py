import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph._shortest_path import dijkstra
from scipy.sparse.csr import csr_matrix
import torch
import argparse

from model.aligner import Aligner
from text.tokenizer import Tokenizer
from utils.config import Config
from utils.dataset import new_aligner_dataset
from utils.io import unpickle_binary
from utils.paths import Paths


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Entrypoint for training the TacoGan model.')
    parser.add_argument(
        '--model', '-m', help='Point to the model pyt.')
    args = parser.parse_args()
    text_dict = unpickle_binary('data/text_dict.pkl')

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('Using device:', device)

    checkpoint = torch.load(args.model, map_location=device)

    cfg = Config.from_string(checkpoint['config'])
    model = Aligner.from_config(cfg).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    paths = Paths()

    print(f'loaded aligner step {model.get_step()}')
    batch_size = 8
    train_set = new_aligner_dataset(
        paths=paths, batch_size=batch_size, cfg=cfg)

    tokenizer = Tokenizer(cfg.symbols)

    for i, (seqs, mels, seq_lens, mel_lens, mel_ids) in enumerate(train_set):
        print(f'{i} / {len(train_set)}')
        pred_batch = model(mels)
        pred_batch = torch.log_softmax(pred_batch, dim=-1)
        pred_batch = pred_batch.detach().cpu().numpy()
        for b in range(batch_size):
            seq_len, mel_len, mel_id = seq_lens[b], mel_lens[b], mel_ids[b]
            pred = pred_batch[b, :mel_len]
            target = seqs[b, :seq_len].numpy()
            target_len = target.shape[0]
            pred_len = pred.shape[0]
            pred_max = np.zeros((pred_len, target_len))

            for i in range(pred_len):
                weight = - pred[i, target]
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
            path = [0] + path + [dist_matrix.size-1]
            cols = pred_max.shape[1]
            mel_text = {}
            text_mel = {}
            text_mel_prob = {}
            durations_new = np.zeros(seq_len)
            for node_index in path:
                i, j = from_node_index(node_index, cols)

                k = target[j]
                prob = pred[i, k]
                tm_prob = text_mel_prob.get(j, -1e10)
                if prob > tm_prob:
                    text_mel[j] = i
                    text_mel_prob[j] = prob

            for node_index in path:
                i, j = from_node_index(node_index, cols)
                mel_text[i] = j

            for t, j in enumerate(text_mel):
                i = text_mel[j]
                k = target[j]
                sym = tokenizer.decode([k])[0]
                if sym == ' ' and 0 < j < len(text_mel) - 1:
                    before = text_mel[j]
                    text_mel[j] = (text_mel[j - 1] + text_mel[j + 1]) // 2

            sum_durs = 0
            for j in range(len(text_mel) - 1):
                durations_new[j] = (text_mel[j] + text_mel[j + 1]) // 2 - sum_durs
                sum_durs += durations_new[j]
            durations_new[-1] = len(mel_text) - sum(durations_new)

            #print('durs new')
            #print(durations_new)
            #print(f'sum durs: {sum(durations)} mel shape {mel.shape}')
            #print(f'sum durs new: {sum(durations_new)} mel len {mel_len}')
            #print(f'sum durs new2: {sum(durations_new2)} mel shape {mel.shape}')
            #print(f'sum durs new2: {sum(durations_new2)} mel shape {mel.shape}')

            np.save(paths.dur/f'{mel_id}.npy', np.array(durations_new))
        #    np.save(paths.alg2/f'{id}.npy', np.array(durations_new))
        #    np.save(paths.alg2/f'{id}.npy', np.array(durations_new2))
