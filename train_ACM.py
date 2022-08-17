from models import GCNLayer
import torch
from torch import nn, optim
import numpy as np
import dgl
from dgl.data.utils import load_graphs
import torch.nn.functional as F
from sklearn.metrics import f1_score
import random
from models import HeteroRGCNLayer
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from load_data import mat2graph
from dgl.nn.pytorch.softmax import edge_softmax
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair
from dgl import function as fn
import argparse


def make_prediction(x, y, train_idx, val_idx, test_idx, seed=123):
    model = LogisticRegression(random_state=seed, solver='liblinear',
                               multi_class='ovr', max_iter=200)
    model.fit(x[train_idx], y[train_idx])
    val_prob = model.predict_proba(x[val_idx])
    loss = log_loss(y[val_idx], val_prob, labels=[0, 1, 2])
    val_pred, test_pred = model.predict(x[val_idx]), model.predict(x[test_idx])
    val_f1, test_f1, macro_f1 = f1_score(y[val_idx], val_pred, average='micro'), \
                                f1_score(y[test_idx], test_pred, average='micro'), \
                                f1_score(y[test_idx], test_pred, average='macro')
    return val_f1, test_f1, macro_f1, loss


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


class RGCN(nn.Module):
    def __init__(self, G, in_feats, hidden_feats, etypes, embed_ntypes,
                 activation=F.relu, weighted=False, bias=False,
                 residual=False, batchnorm=False, dropout=0., self_loop=True):
        super(RGCN, self).__init__()
        self.G = G
        self.embed_ntypes = embed_ntypes
        self.gcn_layer = HeteroRGCNLayer(in_feats=in_feats, out_feats=hidden_feats, etypes=etypes,
                                         activation=activation, weighted=weighted, bias=bias,
                                         residual=residual, batchnorm=batchnorm, dropout=dropout, self_loop=self_loop)
        if self.embed_ntypes is not None:
            self.embed_layers = nn.ModuleDict({
            ntype: nn.Embedding(self.G.num_nodes(ntype), in_feats) for ntype in embed_ntypes})

    def forward(self, feat_dict):
        if self.embed_ntypes is not None:
            for ntype in self.embed_ntypes:
                feat_dict[ntype] = self.embed_layers[ntype](torch.arange(self.G.num_nodes(ntype)).to(self.G.device))

        return self.gcn_layer(self.G, feat_dict)
#
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
                self._in_src_feats, out_feats * num_heads, bias=True)
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
            nn.init.xavier_normal_(self.fc_v.weight, gain=gain)
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
        graph.srcdata.update({'ft': feat_src, 'el': el})
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
        self.homo_g = None

    def forward(self, bg, feat_dict, ntype):
        if self.all_ntypes is None:
            self.all_ntypes = bg.ntypes
            self.all_ntypes.sort()

        if self.homo_g is None:
            self.homo_g = self.get_FC_graph(bg.num_nodes(ntype)).to(bg.device)

        homo_feat = feat_dict[ntype]
        feat_dict[ntype] = self.layer(self.homo_g, homo_feat)
        return feat_dict

    def get_FC_graph(self, n_node):
        homo_g = mat2graph(np.ones((n_node, n_node)))
        return homo_g



class Model(nn.Module):
    def __init__(self, G, in_feats_list, hidden_feats, etypes, embed_ntypes, ntype, meta_paths, embed_dim=128,
                 activation=F.relu, weighted=False, bias=False, norm=True,
                 residual=False, batchnorm=False, dropout=0., self_loop=True, num_heads=1):
        super(Model, self).__init__()

        self.transformation = nn.ModuleDict({
            ntype: nn.Linear(in_feats, embed_dim) for ntype, in_feats in in_feats_list})

        self.G_1hop = G
        # self.G_2hop = get_2_hop_heterograph(G, ntype)
        self.SAlayer = SALayer(in_feats=hidden_feats, out_feats=hidden_feats // num_heads, num_heads=num_heads,
                               feat_drop=dropout, attn_drop=dropout, alpha=0.2, residual=True,
                               agg_mode='flatten', activation=activation)
        self.ntype = ntype
        self.meta_paths = meta_paths

        self.gcn_1hop = RGCN(G=self.G_1hop, in_feats=embed_dim, hidden_feats=hidden_feats,
                             etypes=etypes, embed_ntypes=embed_ntypes,
                             activation=activation, weighted=weighted, bias=bias,
                             residual=residual, batchnorm=batchnorm, dropout=dropout,
                             self_loop=self_loop)
        # self.gcn_2 = RGCN(G=self.G_1hop, in_feats=hidden_feats, hidden_feats=hidden_feats,
        #                      etypes=etypes, embed_ntypes=None,
        #                      activation=activation, weighted=weighted, bias=bias,
        #                      residual=residual, batchnorm=batchnorm, dropout=dropout,
        #                      self_loop=self_loop)


        # self.gcn_2hop = GCNLayer(in_feats=embed_dim, out_feats=hidden_feats, activation=activation,
        #                          weighted=weighted, bias=bias, norm=norm, residual=residual,
        #                          batchnorm=batchnorm, dropout=dropout)

        self.meta_path_emb = nn.ModuleDict({
            meta_path_name: GCNLayer(in_feats=embed_dim, out_feats=hidden_feats, activation=activation,
                                     weighted=weighted, bias=bias, norm=norm, residual=False,
                                     batchnorm=batchnorm, dropout=dropout) for meta_path_name, _ in self.meta_paths
        })


        self.meta_paths_graph = None

        self.scores_func = Discriminator(hidden_feats, hidden_feats)
        self.h = None
        self.mp_pos_scores, self.mp_neg_scores = {}, {}

    def forward(self, feat_dict=None):
        if self.meta_paths_graph is None:
            self.meta_paths_graph = {meta_path_name: dgl.metapath_reachable_graph(self.G_1hop, meta_path)
                .add_self_loop().to(self.G_1hop.device)
                                     for meta_path_name, meta_path in self.meta_paths}

        if feat_dict is None:
            feat_dict = self.G_1hop.ndata['feat'].copy()
        for ntype in feat_dict.keys():
            feat_dict[ntype] = self.transformation[ntype](feat_dict[ntype].float())


        h_1hop = self.gcn_1hop(feat_dict)
        h_1hop = self.SAlayer(self.G_1hop, h_1hop, self.ntype)
        h_anchor = h_1hop[self.ntype]
        h_pos = [self.meta_path_emb[meta_path_name](self.meta_paths_graph[meta_path_name], feat_dict[self.ntype])
                 for meta_path_name, _ in self.meta_paths]



        shuffle_idx = np.random.permutation(h_anchor.shape[0])
        h_neg = [h[shuffle_idx] for h in h_pos]

        pos_scores, neg_scores = [], []

        for i in range(len(h_pos)):
            pos_score, neg_score = self.scores_func(h_anchor, h_pos[i], h_neg[i])
            pos_scores.append(pos_score)
            neg_scores.append(neg_score)
            self.mp_pos_scores[self.meta_paths[i][0]] = pos_score.cpu().detach().numpy()
            self.mp_neg_scores[self.meta_paths[i][0]] = neg_score.cpu().detach().numpy()

        return torch.cat(pos_scores), torch.cat(neg_scores)

    def get_emb(self, feat_dict=None):
        if self.meta_paths_graph is None:
            self.meta_paths_graph = {meta_path_name: dgl.metapath_reachable_graph(self.G_1hop, meta_path)
                .add_self_loop().to(self.G_1hop.device)
                                     for meta_path_name, meta_path in self.meta_paths}

        if feat_dict is None:
            feat_dict = self.G_1hop.ndata['feat'].copy()
        for ntype in feat_dict.keys():
            feat_dict[ntype] = self.transformation[ntype](feat_dict[ntype].float())

        h_1hop = self.gcn_1hop(feat_dict)
        h_1hop = self.SAlayer(self.G_1hop, h_1hop, self.ntype)

        h_anchor = h_1hop[self.ntype]
        return h_anchor.cpu().detach().numpy()



def train_SSL(args):
    seed = 123
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    dropout = args.dropout
    lr = args.lr
    weight_decay = args.weight_decay
    epochs = args.epochs
    hidden_feats = args.hidden_feats
    embed_dim = args.embed_dim

    activation = F.relu
    weighted = False
    bias = False
    residual = False
    batchnorm = False
    self_loop = True
    norm = True

    (train_idx, val_idx, test_idx) = pickle.load(open('split_idx_ACM.pkl', 'rb'))
    G = load_graphs('data/simple_ACM_G.bin')[0][0]
    device = torch.device('cuda:0')
    y = G.ndata['label']['P'].numpy()
    etypes = G.etypes
    embed_ntypes = ['A', 'S']
    ntype = 'P'
    meta_paths = [('PPP', ['PP']), ('PAP', ['PA', 'AP']), ('PSP', ['PS', 'SP'])]
    in_feats_list = [('P', G.nodes['P'].data['feat'].shape[1])]


    loss_func = nn.BCELoss()
    G = G.to(device)
    model = Model(G=G, in_feats_list=in_feats_list, hidden_feats=hidden_feats, ntype=ntype,
                  norm=norm, etypes=etypes, embed_ntypes=embed_ntypes, embed_dim=embed_dim,
                  activation=activation, weighted=weighted, bias=bias, meta_paths=meta_paths,
                  residual=residual, batchnorm=batchnorm, dropout=dropout, self_loop=self_loop)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    losses = []
    val_losses = []
    val_f1_scores, test_f1_scores = [], []
    test_macro_f1s = []


    for epoch in range(epochs):
        model.train()
        pos_scores, neg_scores = model()
        pos_labels, neg_labels = torch.ones(pos_scores.shape[0]).to(device), torch.zeros(neg_scores.shape[0]).to(device)


        loss = (loss_func(pos_scores, pos_labels) + loss_func(neg_scores, neg_labels)) / 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        losses.append(loss.item())

        with torch.no_grad():
            model.eval()
            emb = model.get_emb()

            val_f1, test_f1, test_macro, val_loss = make_prediction(emb, y, train_idx, val_idx, test_idx)
            test_f1_scores.append(test_f1)
            val_f1_scores.append(val_f1)
            test_macro_f1s.append(test_macro)
            val_losses.append(val_loss)
            print(epoch, 'test_if1 score: ', test_f1_scores[-1] , 'test_af1 score: ', test_macro_f1s[-1], 'val_loss: ', val_loss)


    print(np.argmin(val_losses))
    return test_f1_scores[np.argmin(val_losses)], test_macro_f1s[np.argmin(val_losses)]


parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=300,
                    help="number of total training epochs")
parser.add_argument("--embed_dim", type=int, default=128,
                    help="number of initial projection dimensions")
parser.add_argument("--hidden_feats", type=int, default=128,
                    help="number of hidden dimensions")
parser.add_argument("--dropout", type=float, default=.0,
                    help="dropout")
parser.add_argument("--lr", type=float, default=1e-3,
                    help="learning rate")
parser.add_argument('--weight-decay', type=float, default=1e-4,
                    help="weight decay")
parser.add_argument('--num_heads', type=int, default=1,
                    help="the number of SA heads")
args = parser.parse_args()

print(train_SSL(args))











