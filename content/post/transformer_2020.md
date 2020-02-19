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

__Initialize:__

$$W_T \in \mathbb{R}^{\text{vocab size} \times d} = \mathbb{R}^{\text{vocab size} \times 768} ... \text{token embeddings}$$ 

$$W_P \in \mathbb{R}^{\text{max input length} \times d} = \mathbb{R}^{512 \times 768} ... \text{positional embeddings}$$

$$h \in \\{1,...,n\_{heads}\\}, n_{heads}=12$$

$$l \in \\{1,...,n\_{layers}\\}, n_{layers}=12$$

$$W_{h,l}^Q \in \mathbb{R}^{d \times d_q} = \mathbb{R}^{768 \times 64} ...  \text{query weight matrices}$$

$$W_{h,l}^K \in \mathbb{R}^{d \times d_k} = \mathbb{R}^{768 \times 64} ...  \text{key weight matrices}$$

$$W_{h,l}^V \in \mathbb{R}^{d \times d_q} = \mathbb{R}^{768 \times 64} ...  \text{value weight matrices}$$ 

$$W\_{l}^{ffnn} \in \mathbb{R}^{d \times d\_{ffnn}} = \mathbb{R}^{768 \times 3072} ... \text{feedforward layer's weight matrix}$$

$$b\_{l}^{ffnn} \in \mathbb{R}^{1 \times d\_{ffnn}} = \mathbb{R}^{1 \times 3072} ... \text{feedforward layer's bias vector}$$

$$W_{l}^{out} \in \mathbb{R}^{d\_{ffnn} \times d} = \mathbb{R}^{3072 \times 768} ... \text{output layer's weight matrix}$$ 

$$b_{l}^{out} \in \mathbb{R}^{1 \times d} = \mathbb{R}^{1 \times 768} ... \text{output layer's bias vector}$$

$$W^{final} \in \mathbb{R}^{d \times d} = \mathbb{R}^{768 \times 768} ... \text{final layer's weight matrix}
$$

$$I=(i_{1},...,i\_{512}) \in \mathbb{N}\_{0}^{1 \times \text{max input length}} = \mathbb{N}\_{0}^{1 \times 512} ... \text{input vocab indices}$$

$$T=\texttt{lookup}(W_T,I) \in \mathbb{R}^{\text{max input length} \times d} = \mathbb{R}^{512 \times 768} ... \text{input token embeddings}$$

$$X = T + W_P  \in \mathbb{R}^{\text{max input length} \times d} = \mathbb{R}^{512 \times 768} ... \text{input embeddings}$$

$$Z_0=X$$


__Forward algorithmm:__

For $l \in \\{1,...,n\_{layers}\\}, n\_{layers}=12$:

For $h \in \\{1,...,n\_{heads}\\}, n\_{heads}=12$: 
       
$$Q_{h,l}=Z_{l-1}W_{h,l}^{Q} \in \mathbb{R}^{\text{max input len} \times d_q} = \mathbb{R}^{512 \times 64} \hdots \text{query matrix}$$
        
$$K_{h,l}=Z_{l-1}W_{h,l}^{K} \in \mathbb{R}^{\text{max input len} \times d_k} = \mathbb{R}^{512 \times 64} \hdots \text{key matrix}$$
        
$$V_{h,l}=Z_{l-1}W_{h,l}^{V} \in \mathbb{R}^{\text{max input len} \times d_v} = \mathbb{R}^{512 \times 64} \hdots \text{value matrix}$$
        
$$A_{h,l} = \texttt{Softmax}(\frac{Q_{h,l}K_{h,l}^T}{\sqrt{d_k}}) \in \mathbb{R}^{\text{max input len} \times \text{max input len}} = \mathbb{R}^{512 \times 512}$$
        
$$Z_{h,l} = A_{h,l}V_{h,l} \in \mathbb{R}^{\text{max input len} \times d_v} = \mathbb{R}^{512 \times 64}$$
       

    
$$\tilde{Z}_l = \texttt{concat}(Z_{1,l}, \hdots,Z_{n_{\text{heads}},l}) \in \mathbb{R}^{\text{max input len} \times (d_v \cdot n_{\text{heads}})} = \mathbb{R}^{512 \times (64 \cdot 12)} = \mathbb{R}^{512 \times 768}$$
    
$$\bar{Z_l} = \texttt{LayerNorm}(X+\tilde{Z_l}) \in \mathbb{R}^{512 \times 768}$$
    
$$Z_l^{ffnn}=\max(0, \bar{Z_l}W_l^{ffnn}+b_l^{ffnn}) \in \mathbb{R}^{\text{max input len} \times d_{ffnn}} = \mathbb{R}^{512 \times 3072}$$
    
$$Z_l^{out} = Z_l^{ffnn}W_l^{out} + b_l^{out} \in  \mathbb{R}^{\text{max input len} \times d} = \mathbb{R}^{512 \times 768}$$
    
$$Z_l = \texttt{LayerNorm}(X+Z_l^{out}) \in \mathbb{R}^{512 \times 768}$$


Pass $\text{tanh}(W^{final}Z\_{n\_{layers}}[0,:])$ to the final $\texttt{Softmax}$ that predicts the class, where $Z\_{n\_{layers}}[0,:]$ is the hidden state corresponding to the first token.

