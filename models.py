import torch as th
from torch import nn
from torch.nn import init
from dgl import function as fn
import torch.nn.functional as F
from dgl.nn.pytorch.softmax import edge_softmax
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair
import dgl


class GraphConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 norm=True,
                 bias=True,
                 weighted=False,
                 activation=None):
        super(GraphConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm
        self._weighted = weighted

        self.send = fn.copy_src(src='h', out='m')
        self.recv = fn.sum(msg='m', out='h')
        if self._weighted:
            self.send = fn.u_mul_e('h', 'w', 'm')
        self.weight = nn.Parameter(th.Tensor(in_feats, out_feats))
        if bias:
            self.bias = nn.Parameter(th.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self._activation = activation

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, graph, feat):
        r"""Compute graph convolution.

        Notes
        -----
        * Input shape: :math:`(N, *, \text{in_feats})` where * means any number of additional
          dimensions, :math:`N` is the number of nodes.
        * Output shape: :math:`(N, *, \text{out_feats})` where all but the last dimension are
          the same shape as the input.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature

        Returns
        -------
        torch.Tensor
            The output feature
        """
        graph = graph.local_var()
        if self._norm:
            try:
                norm = th.pow(graph.ndata['in_degrees'].float().clamp(min=1), -0.5)
            except KeyError:
                norm = th.pow(graph.in_degrees().float().clamp(min=1), -0.5)
            shp = norm.shape + (1,) * (feat.dim() - 1)
            norm = th.reshape(norm, shp).to(feat.device)
            feat = feat * norm

        if self._in_feats > self._out_feats:
            # mult W first to reduce the feature size for aggregation.
            feat = th.matmul(feat, self.weight)
            graph.ndata['h'] = feat
            graph.update_all(self.send, self.recv)
            rst = graph.ndata['h']
        else:
            # aggregate first then mult W
            graph.ndata['h'] = feat
            graph.update_all(self.send, self.recv)
            rst = graph.ndata['h']
            rst = th.matmul(rst, self.weight)

        if self._norm:
            rst = rst * norm

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


class GCNLayer(nn.Module):
    """Single layer GCN for updating node features

    Parameters
    ----------
    in_feats : int
        Number of input atom features
    out_feats : int
        Number of output atom features
    activation : activation function
        Default to be ReLU
    weighted : weighted graph or not
        Default to be False
    residual : bool
        Whether to use residual connection, default to be True
    batchnorm : bool
        Whether to use batch normalization on the output,
        default to be True
    dropout : float
        The probability for dropout. Default to be 0., i.e. no
        dropout is performed.
    """
    def __init__(self, in_feats, out_feats, activation=F.relu, weighted=False, bias=True,
                 norm=True, residual=True, batchnorm=True, dropout=0.):
        super(GCNLayer, self).__init__()

        self.activation = activation
        self.graph_conv = GraphConv(in_feats=in_feats, out_feats=out_feats, bias=bias,
                                    norm=norm, weighted=weighted, activation=activation)
        self.dropout = nn.Dropout(dropout)

        self.residual = residual
        if residual:
            self.res_connection = nn.Linear(in_feats, out_feats)

        self.bn = batchnorm
        if batchnorm:
            self.bn_layer = nn.BatchNorm1d(out_feats)

    def forward(self, bg, feats):
        """Update atom representations

        Parameters
        ----------
        bg : BatchedDGLGraph
            Batched DGLGraphs for processing multiple molecules in parallel
        feats : FloatTensor of shape (N, M1)
            * N is the total number of atoms in the batched graph
            * M1 is the input atom feature size, must match in_feats in initialization

        Returns
        -------
        new_feats : FloatTensor of shape (N, M2)
            * M2 is the output atom feature size, must match out_feats in initialization
        """
        new_feats = self.graph_conv(bg, feats)

        if self.residual:
            res_feats = self.res_connection(feats)
            if self.activation is not None:
                res_feats = self.activation(res_feats)
            new_feats = new_feats + res_feats
        new_feats = self.dropout(new_feats)

        if self.bn:
            new_feats = self.bn_layer(new_feats)

        return new_feats


class GATConv(nn.Module):
    r"""Apply `Graph Attention Network <https://arxiv.org/pdf/1710.10903.pdf>`__
    over an input signal.

    .. math::
        h_i^{(l+1)} = \sum_{j\in \mathcal{N}(i)} \alpha_{i,j} W^{(l)} h_j^{(l)}

    where :math:`\alpha_{ij}` is the attention score bewteen node :math:`i` and
    node :math:`j`:

    .. math::
        \alpha_{ij}^{l} & = \mathrm{softmax_i} (e_{ij}^{l})

        e_{ij}^{l} & = \mathrm{LeakyReLU}\left(\vec{a}^T [W h_{i} \| W h_{j}]\right)

    Parameters
    ----------
    in_feats : int, or pair of ints
        Input feature size.

        If the layer is to be applied to a unidirectional bipartite graph, ``in_feats``
        specifies the input feature size on both the source and destination nodes.  If
        a scalar is given, the source and destination node feature size would take the
        same value.
    out_feats : int
        Output feature size.
    num_heads : int
        Number of heads in Multi-Head Attention.
    feat_drop : float, optional
        Dropout rate on feature, defaults: ``0``.
    attn_drop : float, optional
        Dropout rate on attention weight, defaults: ``0``.
    negative_slope : float, optional
        LeakyReLU angle of negative slope.
    residual : bool, optional
        If True, use residual connection.
    activation : callable activation function/layer or None, optional.
        If not None, applies an activation function to the updated node features.
        Default: ``None``.
    """
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
        self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
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
        r"""Compute graph attention network layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, H, D_{out})` where :math:`H`
            is the number of heads, and :math:`D_{out}` is size of output feature.
        """
        graph = graph.local_var()
        if isinstance(feat, tuple):
            h_src = self.feat_drop(feat[0])
            h_dst = self.feat_drop(feat[1])
            feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
            feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
        else:
            h_src = h_dst = self.feat_drop(feat)
            feat_src = feat_dst = self.fc(h_src).view(
                -1, self._num_heads, self._out_feats)
        # NOTE: GAT paper uses "first concatenation then linear projection"
        # to compute attention scores, while ours is "first projection then
        # addition", the two approaches are mathematically equivalent:
        # We decompose the weight vector a mentioned in the paper into
        # [a_l || a_r], then
        # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
        # Our implementation is much efficient because we do not need to
        # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
        # addition could be optimized with DGL's built-in function u_add_v,
        # which further speeds up computation and saves memory footprint.
        el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
        er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
        graph.srcdata.update({'ft': feat_src, 'el': el})
        graph.dstdata.update({'er': er})
        # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
        graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
        e = self.leaky_relu(graph.edata.pop('e'))
        # compute softmax
        atte_coef = self.attn_drop(edge_softmax(graph, e))
        graph.edata['a'] = atte_coef
        # message passing
        graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                         fn.sum('m', 'ft'))
        rst = graph.dstdata['ft']
        # residual
        if self.res_fc is not None:
            resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
            rst = rst + resval
        # activation
        if self.activation:
            rst = self.activation(rst)
        return rst


class GATLayer(nn.Module):
    """Single layer GAT for updating node features

    Parameters
    ----------
    in_feats : int
        Number of input atom features
    out_feats : int
        Number of output atom features for each attention head
    num_heads : int
        Number of attention heads
    feat_drop : float
        Dropout applied to the input features
    attn_drop : float
        Dropout applied to attention values of edges
    alpha : float
        Hyperparameter in LeakyReLU, slope for negative values. Default to be 0.2
    residual : bool
        Whether to perform skip connection, default to be False
    agg_mode : str
        The way to aggregate multi-head attention results, can be either
        'flatten' for concatenating all head results or 'mean' for averaging
        all head results
    activation : activation function or None
        Activation function applied to aggregated multi-head results, default to be None.
    """
    def __init__(self, in_feats, out_feats, num_heads, feat_drop, attn_drop,
                 alpha=0.2, residual=True, agg_mode='flatten', activation=None):
        super(GATLayer, self).__init__()
        self.gnn = GATConv(in_feats=in_feats, out_feats=out_feats, num_heads=num_heads,
                           feat_drop=feat_drop, attn_drop=attn_drop,
                           negative_slope=alpha, residual=residual)
        assert agg_mode in ['flatten', 'mean']
        self.agg_mode = agg_mode
        self.activation = activation

    def forward(self, bg, feats):
        """Update atom representations

        Parameters
        ----------
        bg : BatchedDGLGraph
            Batched DGLGraphs for processing multiple molecules in parallel
        feats : FloatTensor of shape (N, M1)
            * N is the total number of atoms in the batched graph
            * M1 is the input atom feature size, must match in_feats in initialization

        Returns
        -------
        new_feats : FloatTensor of shape (N, M2)
            * M2 is the output atom feature size. If self.agg_mode == 'flatten', this would
              be out_feats * num_heads, else it would be just out_feats.
        """
        new_feats = self.gnn(bg, feats)
        if self.agg_mode == 'flatten':
            new_feats = new_feats.flatten(1)
        else:
            new_feats = new_feats.mean(1)

        if self.activation is not None:
            new_feats = self.activation(new_feats)

        return new_feats


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)                     # (M, 1)
        beta = th.softmax(w, dim=0)                     # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)

        return (beta * z).sum(1)                        # (N, D * K)


class HANLayer(nn.Module):
    """
    HAN layer.
    Arguments
    ---------
    meta_paths : list of metapaths, each as a list of edge types
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability
    Inputs
    ------
    g : DGLHeteroGraph
        The heterogeneous graph
    h : tensor
        Input features
    Outputs
    -------
    tensor
        The output feature
    """
    def __init__(self, meta_paths, in_size, out_size, layer_num_heads, dropout):
        super(HANLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(len(meta_paths)):
            self.gat_layers.append(GATConv(in_size, out_size, layer_num_heads,
                                           feat_drop=0.0, attn_drop=dropout, activation=th.sigmoid))
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads, hidden_size=64)
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)

        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self, g, h):
        semantic_embeddings = []

        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path in self.meta_paths:
                self._cached_coalesced_graph[meta_path] = dgl.metapath_reachable_graph(
                        g, meta_path).to(h.device)

        for i, meta_path in enumerate(self.meta_paths):
            new_g = self._cached_coalesced_graph[meta_path]
            semantic_embeddings.append(self.gat_layers[i](new_g, h).flatten(1))
        semantic_embeddings = th.stack(semantic_embeddings, dim=1)                  # (N, M, D * K)

        return self.semantic_attention(semantic_embeddings)                            # (N, D * K)


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
        self.weight = nn.Parameter(th.Tensor(in_feats, out_feats))
        if bias:
            self.bias = nn.Parameter(th.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self._activation = activation

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, graph, feat):
        graph = graph.local_var()
        assert len(graph.canonical_etypes) == 1
        src_type = graph.canonical_etypes[0][0]
        dst_type = graph.canonical_etypes[0][-1]

        if self._in_feats > self._out_feats:
            # mult W first to reduce the feature size for aggregation.
            feat = th.matmul(feat, self.weight)
            graph.nodes[src_type].data['h'] = feat
            graph.update_all(self.send, self.recv)
            rst = graph.nodes[dst_type].data['h']
        else:
            # aggregate first then mult W
            graph.nodes[src_type].data['h'] = feat
            graph.update_all(self.send, self.recv)
            rst = graph.nodes[dst_type].data['h']
            rst = th.matmul(rst, self.weight)

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

    def forward(self, bg, feats):
        new_feats = self.graph_conv(bg, feats)
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
        self.cache_etypes_sub_G = None
        self.etypes = etypes
        self.self_loop = self_loop

        if self_loop:
            self.self_loop_layer = nn.Linear(in_feats, out_feats)

    def forward(self, G, feat_dict, R_aggregate_mode='mean'):
        # The input is a dictionary of node features for each type
        if self.cache_etypes_sub_G is None:
            self.cache_etypes_sub_G = {
                name: dgl.metapath_reachable_graph(G, [name]).to(G.device) for name in self.etypes
            }

        for srctype, etype, dsttype in G.canonical_etypes:
            etype_gcn_layer = self.gcn_layers[etype]
            etype_sub_G = self.cache_etypes_sub_G[etype]
            src_feat = feat_dict[srctype]

            dst_etype_feat = etype_gcn_layer(etype_sub_G, src_feat)
            G.nodes[dsttype].data[etype] = dst_etype_feat

        conv_feature_dict = {}
        for ntype in G.ntypes:
            ntype_feats = [G.nodes[ntype].data[etype] for etype in G.etypes if etype[-1] == ntype]
            if self.self_loop:
                ntype_feats.append(self.self_loop_layer(feat_dict[ntype]))
            ntype_feat = th.mean(th.stack(ntype_feats), 0)
            conv_feature_dict[ntype] = ntype_feat

        return conv_feature_dict


class GAE(nn.Module):
    def __init__(self, in_dim, hidden_dims, dropout=0.0, activation=F.relu, decode_dropout=0.0,
                 weighted=False, bias=True, norm=True, residual=True, batchnorm=True):
        super(GAE, self).__init__()
        layers = [GCNLayer(in_dim, hidden_dims[0], activation=activation, weighted=weighted,
                           bias=bias, norm=norm, residual=residual, batchnorm=batchnorm, dropout=dropout)]
        if len(hidden_dims) >= 2:
            for i in range(1, len(hidden_dims)):
                layers.append(GCNLayer(hidden_dims[i - 1], hidden_dims[i], activation=activation, weighted=weighted,
                                       bias=bias, norm=norm, residual=residual, batchnorm=batchnorm, dropout=dropout))

        self.encoder = nn.ModuleList(layers)
        self.decoder = InnerProductDecoder(dropout=decode_dropout)

    def forward(self, g):
        h = g.ndata['feat'].float()
        # h /= th.sum(h, 1).reshape(-1, 1)
        for conv in self.encoder:
            h = conv(g, h)
        g.ndata['h'] = h
        adj_rec = self.decoder(h)

        return adj_rec

    def encode(self, g):
        h = g.ndata['feat'].float()
        # h /= th.sum(h, 1).reshape(-1, 1)
        for conv in self.encoder:
            h = conv(g, h)
        return h


class InnerProductDecoder(nn.Module):
    def __init__(self, activation=th.sigmoid, dropout=0.3):
        super(InnerProductDecoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, z):
        z = self.dropout(z)
        adj = self.activation(th.mm(z, z.t()))
        return adj

