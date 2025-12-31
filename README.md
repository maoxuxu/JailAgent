---

# JailAgent

This repository contains codes for the paper “[Stop Fixating on Prompts: Reasoning Hijacking and Constraint Tightening for Red-Teaming LLM Agents]()”.

## 🔮 Abstract

With the widespread application of LLM-based agents across various domains, their complexity has introduced new security threats. Existing red-team methods mostly rely on modifying user prompts, which lack adaptability to new data and may impact the agent's performance. To address this, this paper proposes the JailAgent framework, which completely avoids modifying the user prompt. Instead, it implicitly manipulates the agent's reasoning trajectory and memory retrieval. JailAgent consists of three key stages: Prompt Grouping, Reasoning Hijacking, and Constraint Tightening. Through precise trigger identification, real-time adaptive mechanisms, and an optimized objective function, JailAgent demonstrates outstanding performance in cross-model and cross-scenario environments.

<div align="center">
  <img src="https://github.com/maoxuxu/JailAgent/main/jialagent.jpg" width="100%">
</div>

## Requirements

The following packages are required:

- Python (tested on 3.8.13)
- CUDA (tested on 11.6)
- PyTorch (tested on 1.11.0)
- Transformers (tested on 4.14.1)
- Sentence-Transformers
- FAISS
- numpy (tested on 1.22.4)
- scikit-learn
- tqdm
- NLTK
- spaCy

## LLM Agents
The download link for the Agent used in the paper is:

 ► ReAct-UALA：https://uala-agent.github.io/
 
 ► EhrAgent：https://github.com/wshi83/EhrAgent
 
 ► VideoAgent：https://wxh1996.github.io/VideoAgent-Website/

## Dataset
The download link for the dataset used in the paper is:

 ► EgoSchema：https://proceedings.neurips.cc/paper_files/paper/2023/hash/90ce332aff156b910b002ce4e6880dec-Abstract-Datasets_and_Benchmarks.html
 
 ► NExT-QA：https://openaccess.thecvf.com/content/CVPR2021/html/Xiao_NExT-QA_Next_Phase_of_Question-Answering_to_Explaining_Temporal_Actions_CVPR_2021_paper.html
 
 ► HotpotQA：https://aclanthology.org/D18-1259/

 ► StrategyQA：https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00370/100680/Did-Aristotle-Use-a-Laptop-A-Question-Answering
 
 ► MMLU：https://arxiv.org/abs/2009.03300
 
 ► MIMIC-III：https://www.nature.com/articles/sdata201635

 ► eICU：https://www.nature.com/articles/sdata2018178

 ► TREQS：https://dl.acm.org/doi/abs/10.1145/3366423.3380120
 

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourname/yourproject.git
cd yourproject
```

### 2. Prepare Pre-trained Language Models
This project uses two pre-trained models:

► PLM: bert-base-uncased

► Reranker PLM: all-MiniLM-L6-v2

You can either download them automatically or specify local paths in config.py:
```bash
# config.py
PLM = "bert-base-uncased"
PLM_path = "/path/to/bert-base-uncased"

Reranker_PLM = "all-MiniLM-L6-v2"
Reranker_PLM_path = "/path/to/all-MiniLM-L6-v2"
```
If PLM_path or Reranker_PLM_path is provided, the model will be loaded from the local directory; otherwise, it will be downloaded automatically from HuggingFace.

### 3. Run the Code
You can run `main.py` directly to test the pipeline.  
If you want to adapt the method to different agents, please integrate the `run_poison_pipeline` function into the target agent and perform red-teaming evaluations accordingly.  
Hyperparameters can be adjusted based on different scenarios and application settings.





