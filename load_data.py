from torch import tensor, from_numpy
from dgl import batch, from_scipy
from torch.utils.data import DataLoader
from torch import sum
import numpy as np
from scipy import sparse


class BatchGraphDataSet(object):
    """
    :param graphs is a list. type of the element is DGLGraph
    :param labels is a list. type of the element is int or float
    """

    def __init__(self, graphs, labels):
        super(BatchGraphDataSet, self).__init__()
        self.graphs = graphs
        self.labels = labels

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return len(self.graphs)

    def __getitem__(self, idx):
        """Get the i^th sample.

        Paramters
        ---------
        idx : int
            The sample index.

        Returns
        -------
        (dgl.DGLGraph, int)
            The graph and its label.
        """
        return self.graphs[idx], self.labels[idx]


def collate(samples):
    graphs, labels = map(list, zip(*samples))
    bathched_graph = batch(graphs)
    return bathched_graph, tensor(labels).long()


def mat2graph(adjacent_matrix, weighted=False, init_feat=None):
    # adjacent_matrix[range(adjacent_matrix.shape[0]), range(adjacent_matrix.shape[0])] = 0
    g = from_scipy(sparse.csc_matrix(adjacent_matrix))
    g.ndata['in_degrees'] = sum(tensor(adjacent_matrix), 0)
    g.ndata['out_degrees'] = sum(tensor(adjacent_matrix), 1)
    if init_feat is not None:
        g.ndata['init_h'] = tensor(init_feat).float()
    if weighted:
        weight = adjacent_matrix.flatten()
        g.edata['w'] = tensor(weight[weight != 0]).float()
    return g


def mat2vec(matrix):
    index_mat = np.triu(np.ones(matrix[1].shape), k=1)
    _index = index_mat.flatten()
    index = np.squeeze(np.argwhere(_index != 0)).tolist()
    vec = matrix.reshape((matrix.shape[0], -1))
    return vec[:, index]


def get_dataloader(adj_list, feat_list, labels, batch_size=32, shuffle=True, weighted=False):
    g_list = [mat2graph(adj, weighted=weighted, init_feat=feat) for adj, feat in zip(adj_list, feat_list)]
    dataset = BatchGraphDataSet(g_list, labels)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            collate_fn=collate)
    return dataloader

