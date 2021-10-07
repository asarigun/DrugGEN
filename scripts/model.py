import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from layers import GraphConvolution, GraphAggregation, TransformerEncoder



class First_Generator(nn.Module):
    def __init__(self, conv_dims, z_dim, vertexes, edges, nodes, depth, heads, mlp_ratio, drop_rate, dropout):
        super().__init__()
        
        """Adapted from https://github.com/yongqyu/MolGAN-pytorch"""
        
        self.vertexes = vertexes
        self.edges = edges
        self.nodes = nodes

        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_ratio = mlp_ratio
        self.droprate_rate =drop_rate

        layers = []
        for c0, c1 in zip([z_dim]+conv_dims[:-1], conv_dims):
            layers.append(nn.Linear(c0, c1))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(p=dropout, inplace=True))
        self.layers = nn.Sequential(*layers)

        self.edges_layer = nn.Linear(conv_dims[-1], edges * vertexes * vertexes)
        self.nodes_layer = nn.Linear(conv_dims[-1], vertexes * nodes)
        self.dropoout = nn.Dropout(p=dropout)

        self.TransformerEncoder = TransformerEncoder(depth=self.depth, dim=self.dim, heads=self.heads, mlp_ratio=self.mlp_ratio, drop_rate=self.droprate_rate)

    def forward(self, x):
    	
        
        output = self.layers(x)
        edges_logits = self.edges_layer(output)\
                       .view(-1,self.edges,self.vertexes,self.vertexes)
        edges_logits = (edges_logits + edges_logits.permute(0,1,3,2))/2
        edges_logits = self.dropoout(edges_logits.permute(0,2,3,1))

        nodes_logits = self.nodes_layer(output)
        nodes_logits = self.dropoout(nodes_logits.view(-1,self.vertexes,self.nodes))

        edges_logits = self.TransformerEncoder(edges_logits)
        nodes_logits = self.TransformerEncoder(nodes_logits)
        
        return edges_logits, nodes_logits


################################################################
# Discriminator from https://github.com/yongqyu/MolGAN-pytorch #
################################################################

class First_Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, conv_dim, m_dim, b_dim, dropout):
        super(Discriminator, self).__init__()

        graph_conv_dim, aux_dim, linear_dim = conv_dim
        # discriminator
        self.gcn_layer = GraphConvolution(m_dim, graph_conv_dim, b_dim, dropout)
        self.agg_layer = GraphAggregation(graph_conv_dim[-1], aux_dim, b_dim, dropout)

        # multi dense layer
        layers = []
        for c0, c1 in zip([aux_dim]+linear_dim[:-1], linear_dim):
            layers.append(nn.Linear(c0,c1))
            layers.append(nn.Dropout(dropout))
        self.linear_layer = nn.Sequential(*layers)

        self.output_layer = nn.Linear(linear_dim[-1], 1)

    def forward(self, adj, hidden, node, activatation=None):
        adj = adj[:,:,:,1:].permute(0,3,1,2)
        annotations = torch.cat((hidden, node), -1) if hidden is not None else node
        h = self.gcn_layer(annotations, adj)
        annotations = torch.cat((h, hidden, node) if hidden is not None\
                                 else (h, node), -1)
        h = self.agg_layer(annotations, torch.tanh)
        h = self.linear_layer(h)

        # Need to implemente batch discriminator #
        ##########################################

        output = self.output_layer(h)
        output = activatation(output) if activatation is not None else output

        return output, h