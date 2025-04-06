***
This folder is modified by following repository: https://github.com/insuhan/hyper-attn
***

# Efficient Attention via Pre-Scoring: Prioritizing Informative Keys in Transformers

This folder is the PyTorch implementation of pre-scoreing attention paper on Hyperattention:

HyperAttention: Long-context Attention in Near-Linear Time (https://arxiv.org/pdf/2310.05869.pdf)  
Insu Han, Rajesh Jayaram, Amin Karbasi, Vahab Mirrokni, David P. Woodruff, Amir Zandieh ($\alpha$~-~$\beta$)

# Requirements

The requirements of the code is listed in the requirements.txt file.


# Benchmarks

The repository contains two benchmark experiments under the following files:

1. `benchmark_single_attention_layer.py`:  This code is for a benchmark estimating (GPU) runtimes of HyperAttention and FlashAttention exploring the sequence lengths from 1K to 131k. To run, 
    ```shell
    python benchmark_single_attention_layer.py --attn_method hyper 
    ```
    You can choose the computation mode among forward, backward or both. To specify the mode, please add ``--mode fwd`` for forward, ```--mode bwd``` for backward and ```--mode fwd+bwd``` for both. The default is ```fwd+bwd```. Additionally, to simulate attention without causal masking please add ```--no_causal```.


2. `benchmark_patch_llm.py`:  This code is for a benchmark of computing perplexity of pretrained language models where their self-attention is patched with HyperAttention. We choose [chatglm2-6b-32k](https://huggingface.co/THUDM/chatglm2-6b-32k) model and [LongBench](https://huggingface.co/datasets/THUDM/LongBench) datasets. To run with sequence length 32768

    ```shell
    python benchmark_patch_llm.py --attn_method hyper --seq_len 32768
    ```
    You can also override **FlashAttention** by specifying ``--attn_method flash`` and try other sequence lengths by specifying ```--seq_len 65536``` as long as the VRAM allows.

3.`benchmark_patch_llm_chatGLM3.py`: This code is similar to the above code but replacing [chatglm2-6b-32k] with [chatglm3-6b-32k](https://huggingface.co/THUDM/chatglm3-6b-32k). To use this code, you need to change the variable from 'chatglm2-6b-32k' to 'chatglm3-6b-32k' in `replace_llm_attention.py`

We ran all experiments on a single NVIDIA A100 with 40GB VRAM.

# How to use

The impelmentation of Pre-scoring HyperAttention can be found in ``models/attention/hyper_attn.py``. An example of usage:

```python
from models.attention.hyper_attn import HyperAttention

attn = HyperAttention(
    input_dim=64 
    lsh_num_projs=7,
    block_size=256,
    sample_size=256
    min_seq_len=4096,
    top_k=2048,
    score_method="lev",
    use_prescore=1)

attn_output = attn(query, key, value, causal=True)
```

The module has the following parameters:
- ```input_dim```: the dimension of input query and key. (Required)
- ```lsh_num_projs```: the number of dimension in the hashing space. The default is 7.
- ```block_size```: the size of blocks for the block-diagonal approximation. The default is 256.
- ```sample_size```: the number of sampled columns in the attention matrix $A$. The default is 256.
- ```min_seq_len```: minimum sequence length that HyperAttention applies. When the sequence length is smaller than this value we compute exactly using the FlashAttention because additional operations of HyperAttention may not negligble. The default value is ```4096```.
-```top_k```: the number of keys passed from scoring method to HyperAttention. The default value is ```2048```.
-```score_method```: the scoring method used to find important keys before HyperAttention, other methods are identified in the benchmark_patch_llm.py.
-```use_prescore```: the switch to devide using prescore method or not. 1 stands for yes and 0 stands for no.

# License
The code is licensed under the Apache 2.0 license.


# Citation

```bibtex
@article{hyperattention,
  title={Hyperattention: Long-context attention in near-linear time},
  author={Han, Insu and Jarayam, Rajesh and Karbasi, Amin and Mirrokni, Vahab and Woodruff, David and Zandieh, Amir},
  journal={arXiv preprint arXiv:2310.05869},
  year={2023}
}
```
