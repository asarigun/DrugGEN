# MolecularTransGAN
Official implementation of MolecularTransGAN

<p align="center"><img width="100%" src="https://github.com/asarigun/MolecularTransGAN/blob/main/papers/assets/DrugGEN_MolecularTransGAN.png"></p>

MolecularTransGAN: Model architecture for DrugGEN. **Details coming soon!**

## Generator 1

<p align="center"><img width="100%" src="https://github.com/asarigun/MolecularTransGAN/blob/main/papers/assets/Generator1.png"></p>

For the Generator 1, the [MolGAN](https://arxiv.org/abs/1805.11973)'s generator is adapted and developed with Transformers which can be found at ```scripts/layers.py```. Here the MolGAN's Generator:

```python
class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dims, z_dim, vertexes, edges, nodes, dropout):
        super(Generator, self).__init__()

        self.vertexes = vertexes
        self.edges = edges
        self.nodes = nodes

        layers = []
        for c0, c1 in zip([z_dim]+conv_dims[:-1], conv_dims):
            layers.append(nn.Linear(c0, c1))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(p=dropout, inplace=True))
        self.layers = nn.Sequential(*layers)

        self.edges_layer = nn.Linear(conv_dims[-1], edges * vertexes * vertexes)
        self.nodes_layer = nn.Linear(conv_dims[-1], vertexes * nodes)
        self.dropoout = nn.Dropout(p=dropout)

    def forward(self, x):

        output = self.layers(x)
        edges_logits = self.edges_layer(output)\
                       .view(-1,self.edges,self.vertexes,self.vertexes)
        edges_logits = (edges_logits + edges_logits.permute(0,1,3,2))/2
        edges_logits = self.dropoout(edges_logits.permute(0,2,3,1))
        nodes_logits = self.nodes_layer(output)
        nodes_logits = self.dropoout(nodes_logits.view(-1,self.vertexes,self.nodes))
        return edges_logits, nodes_logits
```
and our developed first Generator at ```scripts/model.py``` :

```python
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
```
## Discriminator 1

The discriminator is the same at MolGAN's discriminators consisting of Graph Convolutional Nets (GCNs). You can look at the details at [here](https://arxiv.org/abs/1805.11973). The implementations details can be founa at  ```scripts/model.py```

## News 

* **Related Paper for MolecularTransGAN** project can be found at [here](https://github.com/asarigun/MolecularTransGAN/blob/main/papers)!

* **Higlighted Libraries:** [[PyTorchGeometric]](https://pytorch-geometric.readthedocs.io/en/latest/#) [[TorchDrug]](https://torchdrug.ai/)  

### Paper Presentations :page_facing_up:
-----------
#### :date: 23.09.2021
:small_orange_diamond: Hayriye Çelikbilek & Elif Candas
* [arXiv 2019] **Deep learning for molecular design—a review of the state of the art** [Paper](https://arxiv.org/abs/1903.04388)
* [Journal of Molecular Modeling 2021] **Generative chemistry: drug discovery with deep learning generative models** [[Paper]](https://arxiv.org/abs/2008.09000)[[PDF]](assets/Generativechemistrydrugdiscoverywithdeeplearninggenerativemodels.pdf)
* [Drug Discovery Today: Technologies 2020] **Graph-based generative models for de Novo drug design** [[Paper]](https://www.sciencedirect.com/science/article/pii/S1740674920300251)[[PDF]](assets/GraphbasedgenerativemodelsfordeNovodrugdesign.pdf)

:small_orange_diamond: Ahmet Sarigun [[PDF version of Slides]](papers/assets/DrugGEN_paper_presentations.pdf)
* [Journal of Cheminformatics 2019] **A de novo molecular generation method using latent vector based generative adversarial network**  [[Paper]](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-019-0397-9)[[Code]](https://github.com/Dierme/latent-gan)[[PDF]](assets/A_denovomoleculargenerationmethodusinglatentvectorbasedgenerativeadversarialnetwork.pdf)
* [RSC 2021] **Attention-based generative models for de novo molecular design**  [[Paper]](https://pubs.rsc.org/en/content/articlehtml/2021/sc/d1sc01050f)[[Code]](https://github.com/oriondollar/TransVAE)

#### Upcoming Weeks 

:date:

* [Journal of Cheminformatics 2018] **Multi-objective de novo drug design with conditional graph generative model** [[Paper]](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-018-0287-6)[[Code]](https://github.com/kevinid/molecule_generator)[[PDF]](assets/Multiobjectivedenovodrugdesignwithconditionalgraphgenerativemodel.pdf)
* [arXiv 2018] **MolGAN: An implicit generative model for small molecular graphs** [[Paper]](https://arxiv.org/abs/1805.11973)[[PapersWithCode]](https://paperswithcode.com/paper/molgan-an-implicit-generative-model-for-small)[[PDF]](assets/MolGANAnimplicitgenerativemodelforsmallmoleculargraphs.pdf)


