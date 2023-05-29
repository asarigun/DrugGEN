# DrugGEN: Target Specific De Novo Design of Drug Candidate Molecules with Graph Transformer-based Generative Adversarial Networks



<p align="center">
  <a href="https://arxiv.org/abs/2302.07868"><img src="https://img.shields.io/badge/arXiv-preprint-B31B1B?style-for-the-badge&logo=arXiv"/></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-Implementation-EE4C2C?style-for-the-badge&logo=PyTorch"/></a>
  <a href="https://pytorch-geometric.readthedocs.io/en/latest/"><img src="https://img.shields.io/badge/PyG-Implementation-3C2179?style-for-the-badge&logo=PyG"/></a>
  <a href=https://github.com/asarigun/DrugGEN/blob/main/LICENSE><img src="https://img.shields.io/badge/See%20-License%20-blue"/></a>
  <!--<a href="Give a link here" alt="license"><img src="https://colab.research.google.com/assets/colab-badge.svg"/></a>-->
</p>

<!--PUT HERE SOME QUALITATIVE RESULTS IN THE ASSETS FOLDER-->
<!--YOU CAN PUT ALSO IN THE GIF OR PNG FORMAT -->
<!--<p float="center">
  <img src="assets/sample1.png" width="49%" />
  <img src="assets/sample2.png" width="49%" />
</p>-->

<!--PUT THE ANIMATED GIF VERSION OF THE DRUGGEN MODEL (Figure 1)-->
<p float="center">
  <img src="assets/druggen_figure1_mod.gif" width="98%" />
</p>

<!--
## Abstract

> Discovering novel drug candidate molecules is one of the most fundamental and critical steps in drug development. Generative deep learning models, which create synthetic data given a probability distribution, have been developed with the purpose of picking completely new samples from a partially known space. Generative models offer high potential for designing de novo molecules; however, in order for them to be useful in real-life drug development pipelines, these models should be able to design target-specific molecules, which is the next step in this field. In this study, we propose a novel generative system, DrugGEN, for de novo design of drug candidate molecules that interact with selected target proteins. The proposed system represents compounds and protein structures as graphs and processes them via serially connected two generative adversarial networks comprising graph transformers. DrugGEN is implemented with five independent models, each with a unique sample generation routine. The system is trained using a large dataset of compounds from ChEMBL and target-specific bioactive molecules, to design effective and specific inhibitory molecules against the AKT1 protein, effective targeting of which has critical importance for developing treatments against various types of cancer. DrugGEN has a competitive or better performance against other methods on fundamental benchmarks. To assess the target-specific generation performance, we conducted further in silico analysis with molecular docking and deep learning-based bioactivity prediction. Their results indicate that de novo molecules have high potential for interacting with the AKT1 protein structure in the level of its native ligand. DrugGEN can be used to design completely novel and effective target-specific drug candidate molecules for any druggable protein, given the target features and a dataset of experimental bioactivities.
-->


Check out our paper below for more details

> [**DrugGEN: Target Centric De Novo Design of Drug Candidate Molecules with Graph Generative Deep Adversarial Networks**](https://arxiv.org/abs/2302.07868),            
> [Atabey Ünlü](https://tr.linkedin.com/in/atabeyunlu), [Elif Çevrim](https://www.linkedin.com/in/elifcevrim/?locale=en_US), [Ahmet Sarıgün](https://asarigun.github.io/), [Heval Ataş](https://www.linkedin.com/in/heval-atas/), [Altay Koyaş](https://www.linkedin.com/in/altay-koya%C5%9F-8a6118a1/?originalSubdomain=tr), [Hayriye Çelikbilek](https://www.linkedin.com/in/hayriye-celikbilek/?originalSubdomain=tr), [Deniz Cansen Kahraman](https://www.linkedin.com/in/deniz-cansen-kahraman-6153894b/?originalSubdomain=tr), [Abdurrahman Olğaç](https://www.linkedin.com/in/aolgac/?originalSubdomain=tr), [Ahmet S. Rifaioğlu](https://saezlab.org/person/ahmet-sureyya-rifaioglu/), [Tunca Doğan](https://yunus.hacettepe.edu.tr/~tuncadogan/)     
> *Arxiv, 2023*



## Features

<!--PUT HERE 1-2 SENTECE FOR METHOD WHICH SHOULD BE SHORT  Pleaser refer to our [arXiv report](link here) for further details. -->

This implementation:

- has the demo and training code for DrugGEN implemented in ![badge](https://img.shields.io/badge/PyTorch--EE4C2C?style-for-the-badge&logo=PyTorch) & ![badge](https://img.shields.io/badge/PyG--3C2179?style-for-the-badge&logo=PyG),
- can design de novo drugs based on their protein interactions,
<!-- - supports both CPU and GPU inference (though GPU is way faster), -->
<!-- ADD HERE SOME FEATURES FOR DRUGGEN & SUMMARIES & BULLET POINTS -->


<!-- ADD THE ANIMATED GIF VERSION OF THE GAN1 AND GAN2 -->
| First Generator                                                                                                | Second Generator                                                                                               |
|------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| ![FirstGAN](assets/generator_1_mod.gif) | ![SecondGAN](assets/generator_2_mod.gif) |


## Preliminary results (generated molecules)

| ChEMBL-25                                                                                                | ChEMBL-45                                                                                               |
|------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| ![ChEMBL_25](assets/molecule_1.png) | ![ChEMBL_45](assets/molecule_2.png) |

## Overview
We provide the implementation of the DrugGEN, along with scripts from PyTorch Geometric framework to generate and run. The repository is organised as follows:

```data``` contains: 
- **Raw dataset files**, which should be text files containing SMILES strings only. Raw datasets preferably should not contain stereoisomeric SMILES to prevent Hydrogen atoms to be included in the final graph data. 
- Constructed **graph datasets** (.pt) will be saved in this folder along with atom and bond encoder/decoder files (.pk).

```experiments``` contains: 
- ```logs``` folder. Model loss and performance metrics will be saved in this directory in seperate files for each model. 
- ```tboard_output``` folder. Tensorboard files will be saved here if TensorBoard is used.
- ```models``` folder. Models will be saved in this directory at last or preferred steps. 
- ```samples``` folder. Molecule samples will be saved in this folder.
- ```inference``` folder. Molecules generated in inference mode will be saved in this folder.

**Python scripts are:**

- ```layers.py``` file contains **Transformer Encoder** and **Transformer Decoder** implementations.  
- ```main.py``` contains arguments and this file is used to run the model.   
- ```models.py``` has the implementation of the **Generators** and **Discriminators** which are used in GAN1 and GAN2.  
- ```new_dataloader.py``` constructs the graph dataset from given raw data. Uses PyG based data classes.  
- ```trainer.py``` is the training and testing file for the model. Workflow is constructed in this file.   
- ```utils.py``` contains performance metrics from several other papers and some unique implementations. (De Cao et al, 2018; Polykovskiy et al., 2020)  

## Datasets
Three different data types (i.e., compound, protein, and bioactivity) were retrieved from various data sources to train our deep generative models. GAN1 module requires only compound data while GAN2 requires all of three data types including compound, protein, and bioactivity.
- **Compound data** includes atomic, physicochemical, and structural properties of real drug and drug candidate molecules. [ChEMBL v29 compound dataset](data/dataset_download.sh) was used for the GAN1 module. It consists of 1,588,865 stable organic molecules with a maximum of 45 atoms and containing  C, O, N, F, Ca, K, Br, B, S, P, Cl, and As heavy atoms. 
- **Protein data** was retrieved from Protein Data Bank (PDB) in biological assembly format, and the coordinates of protein-ligand complexes were used to construct the binding sites of proteins from the bioassembly data. The atoms of protein residues within a maximum distance of 9 A from all ligand atoms were recorded as binding sites. GAN2 was trained for generating compounds specific to the target protein AKT1, which is a member of serine/threonine-protein kinases and involved in many cancer-associated cellular processes including metabolism, proliferation, cell survival, growth and angiogenesis. Binding site of human AKT1 protein was generated from the kinase domain (PDB: 4GV1). 
- **Bioactivity data** of AKT target protein was retrieved from large-scale ChEMBL bioactivity database. It contains ligand interactions of human AKT1 (CHEMBL4282) protein with a pChEMBL value equal to or greater than 6 (IC50 <= 1 µM) as well as SMILES information of these ligands. The dataset was extended by including drug molecules from DrugBank database known to interact with human AKT proteins. Thus, a total of [1,600 bioactivity data](data/filtered_akt_inhibitors.smi) points were obtained for training the AKT-specific generative model. 
<!-- To enhance the size of the bioactivity dataset, we also obtained two alternative versions by incorporating ligand interactions of protein members in non-specific serine/threonine kinase (STK) and kinase families. -->

More details on the construction of datasets can be found in our paper referenced above.

## Updates

- 15/02/2023: Our <a href="https://arxiv.org/abs/2302.07868"><img src="https://img.shields.io/badge/arXiv-preprint-B31B1B?style-for-the-badge&logo=arXiv"/></a> is shared [here](assets/DrugGEN_Arxiv_formatted_submitted.pdf) together with its supplementary material document [link](assets/Druggen_Arxiv_submitted_Supplementary_Materials.pdf).
- 01/01/2023: First version script of DrugGEN is released.

## Getting Started
DrugGEN has been implemented and tested on Ubuntu 18.04 with python >= 3.9. It supports both GPU and CPU inference.

<!--If you don't have a suitable device, try running our Colab demo. -->

Clone the repo:
```bash
git clone https://github.com/asarigun/DrugGEN.git
```
<!--
Install the requirements using `virtualenv` or `conda`:
```bash
# pip
source install/install_pip.sh

# conda
source install/install_conda.sh
```

## Running the Demo
You could try Google Colab if you don't already have a suitable environment for running this project.
It enables cost-free project execution in the cloud. You can use the provided notebook to try out our Colab demo:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](Give a link here)-->

## Training

### Setting up environment

You can set up the environment using either conda or pip.

Here is with conda:

```bash
# set up the environment (installs the requirements):

conda env create -f DrugGEN/dependencies.yml

# activate the environment:

conda activate druggen
```

Here is with pip using virtual environment:

```bash
python -m venv DrugGEN/.venv
./Druggen/.venv/bin/activate
pip install -r DrugGEN/requirements.txt
```

### Starting the training

```bash

# Download the raw files

cd DrugGEN/data

bash dataset_download.sh

# DrugGEN can be trained with a one-liner

python DrugGEN/main.py --mode="train" --device="cuda" --raw_file="DrugGEN/data/chembl_smiles.smi" --dataset_file="chembl45.pt" -- drug_raw_file="drug_smies.smi" --drug_dataset_file="drugs.pt" --max_atom=45
```

** Please find the arguments in the **main.py** file. Explanation of the commands can be found below.

```bash
Model arguments:
  --submodel SUBMODEL       Choose the submodel for training
  --act ACT                 Activation function for the model
  --z_dim Z_DIM             Prior noise for the first GAN
  --max_atom MAX ATOM       Maximum atom number for molecules must be specified
  --lambda_gp LAMBDA_GP     Gradient penalty lambda multiplier for the first GAN
  --dim DIM                 Dimension of the Transformer models for both GANs
  --depth DEPTH             Depth of the Transformer model from the first GAN
  --heads HEADS             Number of heads for the MultiHeadAttention module from the first GAN
  --dec_depth DEC_DEPTH     Depth of the Transformer model from the second GAN
  --dec_heads DEC_HEADS     Number of heads for the MultiHeadAttention module from the second GAN
  --mlp_ratio MLP_RATIO     MLP ratio for the Transformers
  --dis_select DIS_SELECT   Select the discriminator for the first and second GAN
  --init_type INIT_TYPE     Initialization type for the model
  --dropout DROPOUT         Dropout rate for the model
Training arguments:
  --batch_size BATCH_SIZE   Batch size for the training
  --epoch EPOCH             Epoch number for Training
  --warm_up_steps           Warm up steps for the first GAN
  --g_lr G_LR               Learning rate for G
  --g2_lr G2_LR             Learning rate for G2
  --d_lr D_LR               Learning rate for D
  --d2_lr D2_LR             Learning rate for D2      
  --n_critic N_CRITIC       Number of D updates per each G update
  --beta1 BETA1             Beta1 for Adam optimizer
  --beta2 BETA2             Beta2 for Adam optimizer 
  --clipping_value          Clipping value for the gradient clipping process
  --resume_iters            Resume training from this step for fine tuning if desired
Dataset arguments:      
  --features FEATURES       Additional node features (Boolean) (Please check new_dataloader.py Line 102)
```

## Molecule Generation Using Trained DrugGEN Models in the Inference Mode


- First, please download the model weights of trained model, e.g., [DrugGEN-Prot](https://drive.google.com/drive/folders/19knQAtpieSamaxB4L5ft8bFiCVikBFDS?usp=share_link) and place it in the folder: "DrugGEN/experiments/models/".
- After that, please run the code below:


```bash

python DrugGEN/main.py --submodel="Prot" --mode="inference" --inference_model="DrugGEN/experiments/models/{Chosen model name}"
```

- SMILES representation of the generated molecules will be saved into the file: "DrugGEN/experiments/inference/{Chosen submodel name}/denovo_molecules.txt".

## De Novo Generated Molecules and its AKT1 inhibitor subset
- SMILES notations of 50,000 de novo generated molecules from DrugGEN models (10,000 from each) can be downloaded from [here](results/generated_molecules). 
- We first filtered the 50,000 de novo generated molecules by applying Lipinski, Veber and PAINS filters; and 43,000 of them remained in our dataset after this operation ([SMILES notations of filtered de novo molecules](results/generated_molecules/filtered_all_generated_molecules.smi)).
- We run our deep learning-based drug/compound-target protein interaction prediction system [DEEPScreen](https://pubs.rsc.org/en/content/articlehtml/2020/sc/c9sc03414e) on 43,000 filtered molecules. DEEPScreen predicted 18,000 of them as active against AKT1, 301 of which received high confidence scores (> 80%) ([SMILES notations of DeepScreen predicted actives](results/deepscreen)).
- At the same time, we performed a molecular docking analysis on these 43,000 filtered de novo molecules against the crystal structure of [AKT1](https://www.rcsb.org/structure/4gv1), and found that 118 of them had sufficiently low binding free energies (< -9 kcal/mol) ([SMILES notations of de novo molecules with low binding free energies](results/docking/Molecules_th9_docking.smi)).
- Finally, de novo molecules to effectively target AKT1 protein are selected via expert curation from the dataset of molecules with binding free energies lower than -9 kcal/mol. The structural representations of the selected molecules are shown in the figure below ([SMILES notations of the expert selected de novo AKT1 inhibitor molecules](results/docking/Selected_denovo_AKT1_inhibitors.smi)).

![structures](assets/Selected_denovo_AKT1_inhibitors.png)

## Citation

```bibtex
@article{unlu2023target,
  title={Target Specific De Novo Design of Drug Candidate Molecules with Graph Transformer-based Generative Adversarial Networks},
  author={{\"U}nl{\"u}, Atabey and {\c{C}}evrim, Elif and Sar{\i}g{\"u}n, Ahmet and {\c{C}}elikbilek, Hayriye and G{\"u}venilir, Heval Ata{\c{s}} and Koya{\c{s}}, Altay and Kahraman, Deniz Cansen, Ol{\u{g}}a{\c{c}}, Abdurrahman, and Rifaio{\u{g}}lu, Ahmet, and Dogan, Tunca},
  journal={arXiv preprint arXiv:2302.07868},
  year={2023}
}
```
<!--ADD BIBTEX AFTER THE PUBLISHING-->

## License

This code is available for **non-commercial scientific research purposes** as will be defined in the [LICENSE file](LICENSE). By downloading and using this code you agree to the terms in the [LICENSE](LICENSE). Third-party datasets and software are subject to their respective licenses.
<!--ADD LICENSE TERMS AND LICENSE FILE AND GIVE A LINK HERE-->

## References

In each file, we indicate whether a function or script is imported from another source. Here are some excellent sources from which we benefit: 
<!--ADD THE REFERENCES THAT WE USED DURING THE IMPLEMENTATION-->
- First GAN is borrowed from [MolGAN](https://github.com/yongqyu/MolGAN-pytorch)
- [MOSES](https://github.com/molecularsets/moses) was used for performance calculation (MOSES Script are directly embedded to our code due to current installation issues related to the MOSES repo).
- Graph Transformer architecture of generators was inspired from [Dwivedi & Bresson, 2020](https://arxiv.org/abs/2012.09699)
- Several [Pytroch Geometric](https://github.com/pyg-team/pytorch_geometric) scripts were used to generate the datasets.
- Graph Transformer Encoder architecture was taken from [Dwivedi & Bresson (2021)](https://arxiv.org/abs/2012.09699) and [Vignac et al. (2022)](https://github.com/cvignac/DiGress) and modified. 
