# DL2_hopfield 
This codebase extends the Hopfield Network based on the paper [Hopfield Network is All you need](https://arxiv.org/abs/2008.02217).

# Environment Setup
Ideally setup a conda environment and install all the requirements. Code for SentEval and Hopfield is bundled as part of the repo, so additional installation is not required.
Use environment.yml or the requirementst.txt to setup your environment.

# Code Setup
We run the pytorch code in train.py for a quick training of the model. For transfer task evaluation(SentEval), use the main.py which is a pytorch lightning implementation that freezes and saves the trained hopfield encoder separately to use it for downstream tasks.

# Downloading Dataset and Setting up Project
We use 3 datasets.
1. SST
2. UDPOS
3. SNLI

All 3 datasets are auto downloaded the first time train.py or main.py is called. 

# Example Training 
An example of how to run the code with `main.py` or `train.py`. 
1. SST Dataset

`python train.py --batch_size 16 --dataset SST --save_every 1` 

This saves the model after each epoch.

2. UDPOS

`python main.py  --batch_size 16 --dataset UDPOS --save_every 1` 

This saves the model after each epoch.

3. SNLI

`python main.py --batch_size 16 --progress_bar` 

This saves the best model in the pl_logs folder. Use the path to run the sentenceEval

## Sentence Evaluation
We use the SentEval toolkit from Facebook - https://github.com/facebookresearch/SentEval

The setup is not required since the code is already part of this repository, however downloading the evaluation dataset is needed. 
To get all the transfer tasks datasets, run the script inside SentEval folder

`./SentEval/data/downstream/get_transfer_data.bash`

Place the SNLI dataset downloaded in the pretrained folder. If downloaded via our code, it should be present in .vector_cache or .data. 
To perform the Sentence Evaluation on all the tasks run

`python SentenceEval.py --model Hopfield`

Evaluated model is saved as *SentenceEvalRes.pt

