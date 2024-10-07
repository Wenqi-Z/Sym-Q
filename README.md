# Sym-Q
![overview](Overview.png)
Inspired by how human experts refine and adapt expressions, we introduce Symbolic Qnetwork (Sym-Q), a novel reinforcement learningbased model that redefines symbolic regression as a sequential decision-making task. Sym-Q leverages supervised demonstrations and refines expressions based on reward signals indicating the quality of fitting precision. Its distinctive ability to manage the complexity of expression trees and perform precise step-wise updates significantly enhances flexibility and efficiency. Our results demonstrate that Sym-Q excels not only in recovering underlying mathematical structures but also uniquely learns to efficiently refine the output expression based on reward signals, thereby discovering underlying expressions. Sym-Q paves the way for more intuitive and impactful discoveries in physical science, marking a substantial advancement in the field of symbolic regression. This repository contains the official Pytorch implementation for the paper ["Sym-Q: Adaptive Symbolic Regression via Sequential Decision-Making"](https://arxiv.org/abs/2402.05306).

## Workspace Setup
To create the symbolic regression workspace with the necessary repositories, follow these steps:

```bash
mkdir symbolic_ws
cd symbolic_ws
git clone https://github.com/SymposiumOrganization/NeuralSymbolicRegressionThatScales.git
git clone https://github.com/Wenqi-Z/Sym-Q.git
```

After running the above commands, your workspace should have the following structure:

```Diff
symbolic_ws
│
└───NeuralSymbolicRegressionThatScales
│ 
└───Sym-Q
```

## Environment Setup
Create and activate a Conda environment for Sym-Q:
```bash
conda create -n SymQ python=3.9
conda activate SymQ
pip install -e NeuralSymbolicRegressionThatScales/src/
pip install -r Sym-Q/requirements.txt
```

## Dataset Creation
Firstly, define the dataset configuration in `NeuralSymbolicRegressionThatScales/dataset_configuration.json`, for this project we use following settings:
```json
{
    "max_len": 20,
    "operators": "add:10,mul:10,sub:5,div:5,sqrt:4,pow2:4,pow3:2,pow4:1,pow5:1,ln:4,exp:4,sin:4,cos:4,tan:4",
    "max_ops": 5,
    "rewrite_functions": "",
    "variables": ["x_1","x_2","x_3"], 
    "eos_index": 1,
    "pad_index": 0
}
```
We use 10 million skeletons as training dataset and 1000 skeletons as validation dataset. Generate the corresponding datasets with following commands:
```bash
python NeuralSymbolicRegressionThatScales/scripts/data_creation/dataset_creation.py --number_of_equations 10000000 --no-debug
python NeuralSymbolicRegressionThatScales/scripts/data_creation/dataset_creation.py --number_of_equations 1000 --no-debug
```
Then you will have your raw dataset under `NeuralSymbolicRegressionThatScales/data/raw_datasets/NumberOfEquations`. We will sample 50 different constants to substitute the ones in the skeltons, you can configure the number in `Sym-Q/cfg.yaml`。 Also set the folder path to place your dataset in the `Sym-Q/cfg.yaml`. Run:
```bash
cd Sym-Q
./run_parallel.sh
```
This will save `.h5` files under the folder you defined and each file contains the data based on 100 skeletons.


## Quick Start
Download the pretrain weights under 10M skeletons from [NeSymReS]https://github.com/SymposiumOrganization/NeuralSymbolicRegressionThatScales/tree/main?tab=readme-ov-file under `Sym-Q/weights`.
Train the Sym-Q model:
```cmd
python train_var3.py
```

This generates a folder `/model/GENERATION_TIME` with the model weights and evaluation data. Generate beamsearch results for evaluation:
```
python beam_search.py --weights_path PATH_TO_WEIGHTS --target ssdnc_r2
```
Run few-shot learning:
```
python few_shot.py --resume_path PATH_TO_WEIGHTS
```


## Acknowledgment

Our implementation is mainly refers to the following codebases. We gratefully thank the authors for their wonderful works.

[SymbolicGPT](https://github.com/mojivalipour/symbolicgpt), [DSO](https://github.com/brendenpetersen/deep-symbolic-optimization), [NeSymReS](https://github.com/SymposiumOrganization/NeuralSymbolicRegressionThatScales), [T-JSL](https://github.com/AILWQ/Joint_Supervised_Learning_for_SR)


## Citing this work

If you found our work useful and used code, please use the following citation:

```
@article{tian2024sym,
  title={Sym-Q: Adaptive Symbolic Regression via Sequential Decision-Making},
  author={Tian, Yuan and Zhou, Wenqi and Dong, Hao and Kammer, David S and Fink, Olga},
  journal={arXiv preprint arXiv:2402.05306},
  year={2024}
}
```
