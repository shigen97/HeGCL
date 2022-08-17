from models import GCNLayer
import torch
from torch import nn, optim
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import random
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pickle
from sklearn.linear_model import LogisticRegression
from ogb.graphproppred.mol_encoder import AtomEncoder
from ogb.graphproppred import collate_dgl
from torch.utils.data import DataLoader
from dgl import function as fn
import dgl
from torch.nn import init
import torch.nn.functional as F
from dgl.nn.pytorch.softmax import edge_softmax
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair
from load_data import mat2graph
import argparse


def make_prediction(x, y, seed=123):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)
    model = LogisticRegression(random_state=seed, solver='liblinear', max_iter=200)
    accs, f1s = [], []
    for train_idx, test_idx in skf.split(np.zeros(y.shape[0]), y):
        model.fit(x[train_idx], y[train_idx])
        y_pred = model.predict(x[test_idx])
        acc, f1 = accuracy_score(y[test_idx], y_pred), \
                  f1_score(y[test_idx], y_pred)
        accs.append(acc)
        f1s.append(f1)
    return accs, f1s


class Discriminator(nn.Module):
    def __init__(self, n_h1, n_h2):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h1, n_h2, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, anchor_feat, pos_feat, neg_feat):

        sc_pos = torch.sigmoid(torch.squeeze(self.f_k(pos_feat, anchor_feat)))
        sc_neg = torch.sigmoid(torch.squeeze(self.f_k(neg_feat, anchor_feat)))

        return sc_pos, sc_neg


class HeteroGraphConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 bias=False,
                 weighted=False,
                 activation=F.relu):
        super(HeteroGraphConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._weighted = weighted

        self.send = fn.copy_src(src='h', out='m')
        self.recv = fn.mean(msg='m', out='h')
        if self._weighted:
            self.send = fn.u_mul_e('h', 'w', 'm')
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self._activation = activation

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, graph, feat, etype):
        graph = graph.local_var()
        src_type = etype[:etype.index('-')]
        dst_type = etype[etype.index('-') + 1:]

        if self._in_feats > self._out_feats:
            # mult W first to reduce torche feature size for aggregation.
            feat = torch.matmul(feat, self.weight)
            graph.nodes[src_type].data['h'] = feat
            graph.update_all(self.send, self.recv, etype=etype)
            rst = graph.nodes[dst_type].data['h']
        else:
            # aggregate first torchen mult W
            graph.nodes[src_type].data['h'] = feat
            graph.update_all(self.send, self.recv, etype=etype)
            rst = graph.nodes[dst_type].data['h']
            rst = torch.matmul(rst, self.weight)

        if self.bias is not None:
            rst = rst + self.bias

        if self._activation is not None:
            rst = self._activation(rst)

        return rst

    def extra_repr(self):
        """Set the extra representation of the module,
        which will come into effect when printing the model.
        """
        summary = 'in={_in_feats}, out={_out_feats}'
        summary += ', normalization={_norm}'
        if '_activation' in self.__dict__:
            summary += ', activation={_activation}'
        return summary.format(**self.__dict__)


class HeteroGCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, activation=F.relu, weighted=False, bias=False,
                 residual=False, batchnorm=False, dropout=0.):
        super(HeteroGCNLayer, self).__init__()

        self.activation = activation
        self.graph_conv = HeteroGraphConv(in_feats=in_feats, out_feats=out_feats, bias=bias,
                                          weighted=weighted, activation=activation)
        self.dropout = nn.Dropout(dropout)

        self.residual = residual
        if residual:
            self.res_connection = nn.Linear(in_feats, out_feats)

        self.bn = batchnorm
        if batchnorm:
            self.bn_layer = nn.BatchNorm1d(out_feats)

    def forward(self, bg, feats, etype):
        new_feats = self.graph_conv(bg, feats, etype)
        if self.residual:
            res_feats = self.res_connection(feats)
            if self.activation is not None:
                res_feats = self.activation(res_feats)
            new_feats = new_feats + res_feats
        new_feats = self.dropout(new_feats)
        if self.bn:
            new_feats = self.bn_layer(new_feats)

        return new_feats


class HeteroRGCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, etypes, activation=F.relu, weighted=False, bias=False,
                 residual=False, batchnorm=False, dropout=0., self_loop=True):
        super(HeteroRGCNLayer, self).__init__()
        self.gcn_layers = nn.ModuleDict({
                name: HeteroGCNLayer(in_feats=in_feats, out_feats=out_feats, activation=activation,
                                     weighted=weighted, bias=bias, residual=residual,
                                     batchnorm=batchnorm, dropout=dropout) for name in etypes
            })
        self.etypes = etypes
        self.self_loop = self_loop
        if self.self_loop:
            self.self_loop_layer = nn.Linear(in_feats, out_feats)

    def forward(self, G, feat_dict):
        for etype in self.etypes:
            srctype = etype[:etype.index('-')]
            dsttype = etype[etype.index('-') + 1:]
            if etype in G.etypes and G.num_edges(etype) != 0:
                etype_gcn_layer = self.gcn_layers[etype]
                src_feat = feat_dict[srctype]
                dst_etype_feat = etype_gcn_layer(G, src_feat, etype)
                G.nodes[dsttype].data[etype] = dst_etype_feat
        conv_feature_dict = {}
        for ntype in G.ntypes:
            if G.num_nodes(ntype) != 0:
                ntype_feats = [G.nodes[ntype].data[etype] for etype in self.etypes
                               if etype[etype.index('-') + 1:] == ntype and G.num_edges(etype) != 0]
                if self.self_loop:
                    ntype_feats.append(self.self_loop_layer(feat_dict[ntype]))
                ntype_feat = torch.mean(torch.stack(ntype_feats), 0)
                conv_feature_dict[ntype] = ntype_feat
            # else:
            #     conv_feature_dict[ntype] =

        return conv_feature_dict


class RGCN(nn.Module):
    def __init__(self, in_feats, hidden_feats, etypes,
                 activation=F.relu, weighted=False, bias=False,
                 residual=False, batchnorm=False, dropout=0., self_loop=True):
        super(RGCN, self).__init__()
        self.gcn_layer = HeteroRGCNLayer(in_feats=in_feats, out_feats=hidden_feats, etypes=etypes,
                                         activation=activation, weighted=weighted, bias=bias,
                                         residual=residual, batchnorm=batchnorm, dropout=dropout, self_loop=self_loop)

    def forward(self, bg, feat_dict):
        return self.gcn_layer(bg, feat_dict)


class GATConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None):
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self.cache_atte = None
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_v = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else: # bipartite graph neural networks
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def forward(self, graph, feat):
        graph = graph.local_var()
        if isinstance(feat, tuple):
            h_src = self.feat_drop(feat[0])
            h_dst = self.feat_drop(feat[1])
            feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
            feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
        else:
            h_src = h_dst = self.feat_drop(feat)
            feat_v = self.fc_v(h_src).view(
                -1, self._num_heads, self._out_feats)
            feat_src = feat_dst = self.fc(h_src).view(
                -1, self._num_heads, self._out_feats)
        el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
        er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
        graph.srcdata.update({'ft': feat_v, 'el': el})
        graph.dstdata.update({'er': er})
        graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
        e = self.leaky_relu(graph.edata.pop('e'))
        atte_coef = self.attn_drop(edge_softmax(graph, e))
        graph.edata['a'] = atte_coef
        graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                         fn.sum('m', 'ft'))
        rst = graph.dstdata['ft']
        if self.res_fc is not None:
            resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
            rst = rst + resval
        if self.activation:
            rst = self.activation(rst)
        return rst


class GATLayer(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads, feat_drop, attn_drop,
                 alpha=0.2, residual=False, agg_mode='flatten', activation=None):
        super(GATLayer, self).__init__()
        self.gnn = GATConv(in_feats=in_feats, out_feats=out_feats, num_heads=num_heads,
                           feat_drop=feat_drop, attn_drop=attn_drop,
                           negative_slope=alpha, residual=residual)
        assert agg_mode in ['flatten', 'mean']
        self.agg_mode = agg_mode
        self.activation = activation

    def forward(self, bg, feats):
        new_feats = self.gnn(bg, feats)
        if self.agg_mode == 'flatten':
            new_feats = new_feats.flatten(1)
        else:
            new_feats = new_feats.mean(1)

        if self.activation is not None:
            new_feats = self.activation(new_feats)

        return new_feats

class SALayer(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads, feat_drop, attn_drop,
                 alpha=0.2, residual=True, agg_mode='flatten', activation=None):
        super(SALayer, self).__init__()
        self.layer = GATLayer(in_feats, out_feats, num_heads, feat_drop, attn_drop,
                 alpha=alpha, residual=residual, agg_mode=agg_mode, activation=activation)
        self.all_ntypes = None

    def forward(self, bg, feat_dict):
        if self.all_ntypes is None:
            self.all_ntypes = bg.ntypes
            self.all_ntypes.sort()

        bnn = 0
        for ntype in self.all_ntypes:
            bnn += bg.batch_num_nodes(ntype)
        homo_bg = self.get_batch_FC_graph(bnn).to(bg.device)

        homo_feat = []
        sub_ntype_nodes = np.zeros((bg.batch_size, len(self.all_ntypes))).astype(int)
        for sub in range(bnn.shape[0]):
            sub_feat = []
            for type_number, ntype in enumerate(self.all_ntypes):
                if bg.num_nodes(ntype) != 0:
                    ntype_bnn = bg.batch_num_nodes(ntype)
                    sub_ntype_nodes[sub, type_number] = ntype_bnn[sub]
                    ntype_nodes_idx = np.arange(feat_dict[ntype].shape[0]).tolist()
                    sub_ntype_node_idx = ntype_nodes_idx[ntype_bnn[:sub].sum().item():
                                                   ntype_bnn[:sub + 1].sum().item()]
                    sub_feat.append(feat_dict[ntype][sub_ntype_node_idx])
            sub_feat = torch.cat(sub_feat, 0)
            homo_feat.append(sub_feat)
        homo_feat = torch.cat(homo_feat, 0)

        conv_feat = self.layer(homo_bg, homo_feat)

        conv_hetero_feat = {}
        for sub in range(bnn.shape[0]):
            sub_feat = conv_feat[bnn[:sub].sum().item():
                                 bnn[:sub+1].sum().item()]
            for type_number, ntype in enumerate(self.all_ntypes):
                if bg.num_nodes(ntype) != 0:
                    sub_idx = np.arange(sub_feat.shape[0])
                    sub_ntype_node_idx = sub_idx[sub_ntype_nodes[sub, :type_number].sum():
                                                 sub_ntype_nodes[sub, :type_number+1].sum()]
                    if sub == 0:
                        conv_hetero_feat[ntype] = sub_feat[sub_ntype_node_idx]
                    else:
                        conv_hetero_feat[ntype] = torch.cat([conv_hetero_feat[ntype],
                                                             sub_feat[sub_ntype_node_idx]], 0)
        return conv_hetero_feat

    def get_batch_FC_graph(self, batch_graph_nodes):
        g_list = [mat2graph(np.ones((n_node, n_node))) for n_node in batch_graph_nodes]
        batch_g = dgl.batch(g_list)
        return batch_g



class Model(nn.Module):
    def __init__(self, in_feats, hidden_feats, etypes, ntypes, meta_paths_list,
                 activation=F.relu, weighted=False, bias=False, norm=True,
                 residual=False, batchnorm=False, dropout=0., self_loop=True, num_heads=8):
        super(Model, self).__init__()

        self.ntypes = ntypes
        self.meta_paths_list = meta_paths_list
        self.heteroGNN_layer = RGCN(in_feats=in_feats, hidden_feats=hidden_feats, etypes=etypes,
                                    activation=activation, weighted=weighted, bias=bias,
                                    residual=residual, batchnorm=batchnorm, dropout=dropout,
                                    self_loop=self_loop)
        self.SAlayer = SALayer(in_feats=hidden_feats, out_feats=hidden_feats // num_heads, num_heads=num_heads,
                               feat_drop=dropout, attn_drop=dropout, alpha=0.2, residual=True,
                               agg_mode='flatten', activation=activation)
        self.ntypes_meta_path_emb = nn.ModuleDict({})
        self.scores_func = nn.ModuleDict({})
        for step, ntype in enumerate(self.ntypes):
            self.ntypes_meta_path_emb[ntype] = nn.ModuleDict({
                meta_path_name: GCNLayer(in_feats=in_feats, out_feats=hidden_feats, activation=activation,
                                         weighted=weighted, bias=bias, norm=norm, residual=residual,
                                         batchnorm=batchnorm, dropout=dropout)
                for meta_path_name, _ in self.meta_paths_list[step]})

            self.scores_func[ntype] = Discriminator(hidden_feats, hidden_feats)

        self.atom_encoder = AtomEncoder(emb_dim=in_feats)

    def forward(self, bg, feat_dict=None):
        if feat_dict is None:
            feat_dict = bg.ndata['feat']
            for ntype in feat_dict.keys():
                if len(feat_dict[ntype]) != 0:
                    feat_dict[ntype] = self.atom_encoder(feat_dict[ntype])

        conv_feat = self.heteroGNN_layer(bg, feat_dict)
        conv_feat = self.SAlayer(bg, conv_feat)

        pos_scores, neg_scores = [], []
        for step, ntype in enumerate(self.ntypes):
            h_anchor = conv_feat[ntype]
            bg_ntype_meta_graphs = {meta_path_name: dgl.metapath_reachable_graph(bg, meta_path)
                .remove_self_loop().add_self_loop().to(bg.device)
                              for meta_path_name, meta_path in self.meta_paths_list[step]}

            h_pos = [self.ntypes_meta_path_emb[ntype][meta_path_name](bg_ntype_meta_graphs[meta_path_name], feat_dict[ntype])
                     for meta_path_name, _ in self.meta_paths_list[step]]

            all_idx = np.arange(h_anchor.shape[0]).tolist()
            bnn = bg.batch_num_nodes(ntype)
            for i in range(bnn.shape[0]):
                batch_idx = all_idx[bnn[:i].sum().item(): bnn[:i + 1].sum().item()]
                random.shuffle(batch_idx)
                all_idx[bnn[:i].sum().item(): bnn[:i + 1].sum().item()] = batch_idx
            h_neg = [h[all_idx] for h in h_pos]


            for i in range(len(h_pos)):
                pos_score, neg_score = self.scores_func[ntype](h_anchor, h_pos[i], h_neg[i])
                pos_scores.append(pos_score)
                neg_scores.append(neg_score)
        return torch.cat(pos_scores), torch.cat(neg_scores)

    def get_emb(self, bg, feat_dict=None):
        if feat_dict is None:
            feat_dict = bg.ndata['feat']
            for ntype in feat_dict.keys():
                if len(feat_dict[ntype]) != 0:
                    feat_dict[ntype] = self.atom_encoder(feat_dict[ntype])

        conv_feat = self.heteroGNN_layer(bg, feat_dict)
        conv_feat = self.SAlayer(bg, conv_feat)
        bg.ndata['conv_feat'] = conv_feat
        hg = 0
        for ntype in conv_feat.keys():
            hg = hg + dgl.sum_nodes(bg, 'conv_feat', ntype=ntype)
        return hg.detach().cpu().numpy()

def train_SSL(args):
    seed = 123
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    epochs = args.epochs
    hidden_feats = args.hidden_feats
    in_feats = args.in_feats
    dropout = args.dropout
    lr = args.lr
    weight_decay = args.weight_decay
    num_heads = args.num_heads

    dataset = pickle.load(open('data/molbace_dataset.pkl', 'rb'))

    idx = np.random.permutation(len(dataset))
    idx = torch.from_numpy(idx).long()
    loader = DataLoader(dataset[idx], batch_size=256, shuffle=False, collate_fn=collate_dgl)
    emb_loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate_dgl)
    etypes = list(pickle.load(open('molbace_etypes.pkl', 'rb')))
    etypes.sort()
    device = torch.device('cuda:0')

    ntypes = ['C']
    meta_paths_list = [[('CC', ['C-C']), ('COC', ['C-O', 'O-C']), ('CNC', ['C-N', 'N-C'])]
                       ]
    
    activation = F.relu
    weighted = False
    bias = False
    residual = False
    batchnorm = False
    self_loop = True
    norm = True
    
    loss_func = nn.BCELoss()

    model = Model(in_feats=in_feats, hidden_feats=hidden_feats, ntypes=ntypes, norm=norm, etypes=etypes,
                  activation=activation, weighted=weighted, bias=bias, meta_paths_list=meta_paths_list,
                  residual=residual, batchnorm=batchnorm, dropout=dropout, self_loop=self_loop, num_heads=num_heads)

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    losses = []

    for epoch in range(1, epochs+1):
        model.train()
        epoch_loss = 0.0
        for step, (bg, _) in enumerate(loader):
            bg = bg.to(device)
            pos_scores, neg_scores = model(bg)
            pos_labels, neg_labels = torch.ones(pos_scores.shape[0]).to(device), \
                                     torch.zeros(neg_scores.shape[0]).to(device)

            loss = (loss_func(pos_scores, pos_labels) + loss_func(neg_scores, neg_labels)) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        losses.append(epoch_loss / (step + 1))
        print(epoch, 'loss:', losses[-1])

    emb = []
    with torch.no_grad():
        model.eval()
        for step, (bg, _) in enumerate(emb_loader):
            bg = bg.to(device)
            emb.append(model.get_emb(bg))


    emb = np.concatenate(emb, 0)
    y = dataset.labels.numpy().squeeze()
    accs, f1s = make_prediction(emb, y)


    print('acc: ', np.mean(accs), np.std(accs))
    print('f1 : ', np.mean(f1s), np.std(f1s))



parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=15,
                    help="number of total training epochs")
parser.add_argument("--in_feats", type=int, default=100,
                    help="number of hidden layers and hidden dimensions")
parser.add_argument("--hidden_feats", type=int, default=256,
                    help="number of hidden layers and hidden dimensions")
parser.add_argument("--dropout", type=float, default=.3,
                    help="dropout")
parser.add_argument("--lr", type=float, default=1e-3,
                    help="learning rate")
parser.add_argument('--weight-decay', type=float, default=1e-4,
                    help="weight decay")
parser.add_argument('--num_heads', type=int, default=8,
                    help="the number of SA heads")
args = parser.parse_args()

train_SSL(args)












