+++
title = "BERT-base Transformer Forward Pass"
description = "Simple"
date = "2020-02-19"
categories = [ "NLP", "Transfomer" ]
tags = [
    "NLP",
    "Transformer"
]
+++

## Initialize

$W_T \in \mathbb{R}^{\text{vocab size} \times d} = \mathbb{R}^{\text{vocab size} \times 768} ... \text{token embeddings}$ 

$W_P \in \mathbb{R}^{\text{max input length} \times d} = \mathbb{R}^{512 \times 768} ... \text{positional embeddings}$

$h \in \\{1,...,n\_{heads}\\}, n_{heads}=12$

$l \in \\{1,...,n\_{layers}\\}, n_{layers}=12$

$W_{h,l}^Q \in \mathbb{R}^{d \times d_q} = \mathbb{R}^{768 \times 64} ...  \text{query weight matrices}$

$W_{h,l}^K \in \mathbb{R}^{d \times d_k} = \mathbb{R}^{768 \times 64} ...  \text{key weight matrices}$

$W_{h,l}^V \in \mathbb{R}^{d \times d_q} = \mathbb{R}^{768 \times 64} ...  \text{value weight matrices}$

$W\_{l}^{ffnn} \in \mathbb{R}^{d \times d\_{ffnn}} = \mathbb{R}^{768 \times 3072} ... \text{feedforward layer's weight matrix}$

$b\_{l}^{ffnn} \in \mathbb{R}^{1 \times d\_{ffnn}} = \mathbb{R}^{1 \times 3072} ... \text{feedforward layer's bias vector}$

$W_{l}^{out} \in \mathbb{R}^{d\_{ffnn} \times d} = \mathbb{R}^{3072 \times 768} ... \text{output layer's weight matrix}$ 

$b_{l}^{out} \in \mathbb{R}^{1 \times d} = \mathbb{R}^{1 \times 768} ... \text{output layer's bias vector}$

$W^{final} \in \mathbb{R}^{d \times d} = \mathbb{R}^{768 \times 768} ... \text{final layer's weight matrix}
$

$I=(i_{1},...,i\_{512}) \in \mathbb{N}\_{0}^{1 \times \text{max input length}} = \mathbb{N}\_{0}^{1 \times 512} ... \text{input vocab indices}$

$T=\texttt{lookup}(W_T,I) \in \mathbb{R}^{\text{max input length} \times d} = \mathbb{R}^{512 \times 768} ... \text{input token embeddings}$

$X = T + W_P  \in \mathbb{R}^{\text{max input length} \times d} = \mathbb{R}^{512 \times 768} ... \text{input embeddings}$

$Z_0=X$


## Forward algorithmm:

For $l \in \\{1,...,n\_{layers}\\}, n\_{layers}=12$:

&nbsp;&nbsp;&nbsp;&nbsp;For $h \in \\{1,...,n\_{heads}\\}, n\_{heads}=12$: 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$Q\_{h,l}=Z\_{l-1} W\_{h,l}^{Q} \in \mathbb{R}^{\text{max input len} \times d_{q}} = \mathbb{R}^{512 \times 64} ... \text{query matrix}$
        
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$K\_{h,l}=Z\_{l-1} W_\{h,l}^{K} \in \mathbb{R}^{\text{max input len} \times d_k} = \mathbb{R}^{512 \times 64} ... \text{key matrix}$
        
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$V\_{h,l}=Z\_{l-1} W\_{h,l}^{V} \in \mathbb{R}^{\text{max input len} \times d_v} = \mathbb{R}^{512 \times 64} ... \text{value matrix}$
        
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$A\_{h,l} = \texttt{Softmax}(\frac{Q\_{h,l}K\_{h,l}^T}{\sqrt{d_k}}) \in \mathbb{R}^{\text{max input len} \times \text{max input len}} = \mathbb{R}^{512 \times 512}$
        
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$Z\_{h,l} = A\_{h,l}V\_{h,l} \in \mathbb{R}^{\text{max input len} \times d_v} = \mathbb{R}^{512 \times 64}$$
       
&nbsp;&nbsp;&nbsp;&nbsp;$\tilde{Z}_l = \texttt{concat}(Z\_{1,l},...,Z\_{n\_{heads},l}) \in \mathbb{R}^{\text{max input len} \times (d_v \cdot n\_{heads})} = \mathbb{R}^{512 \times (64 \cdot 12)} = \mathbb{R}^{512 \times 768}$
    
&nbsp;&nbsp;&nbsp;&nbsp;$\bar{Z_l} = \texttt{LayerNorm}(X+\tilde{Z_l}) \in \mathbb{R}^{512 \times 768}$
    
&nbsp;&nbsp;&nbsp;&nbsp;$Z\_l^{ffnn}=\max(0, \bar{Z_l}W\_l^{ffnn}+b_l^{ffnn}) \in \mathbb{R}^{\text{max input len} \times d\_{ffnn}} = \mathbb{R}^{512 \times 3072}$
    
&nbsp;&nbsp;&nbsp;&nbsp;$Z\_l^{out} = Z\_l^{ffnn}W_l^{out} + b_l^{out} \in  \mathbb{R}^{\text{max input len} \times d} = \mathbb{R}^{512 \times 768}$
    
&nbsp;&nbsp;&nbsp;&nbsp;$Z_l = \texttt{LayerNorm}(X+Z_l^{out}) \in \mathbb{R}^{512 \times 768}$


Pass $\text{tanh}(W^{final}Z\_{n\_{layers}}[0,:])$ to the final $\texttt{Softmax}$ that predicts the class, where $Z\_{n\_{layers}}[0,:]$ is the hidden state corresponding to the first token.

