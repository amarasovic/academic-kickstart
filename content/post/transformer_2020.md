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

$$h \in \\{1,..., n_{\text{heads}}\\}, l \in \\{1,..., n_{\text{layers}}\\}, n_{\text{heads}}=12, n_{\text{layers}}=12$$

$$W_{h,l}^Q \in \mathbb{R}^{d \times d_q} = \mathbb{R}^{768 \times 64} ...  query \textit{weight} matrices$$

$$W_{h,l}^K \in \mathbb{R}^{d \times d_k} = \mathbb{R}^{768 \times 64} ...  key \textit{weight} matrices$$

$$W_{h,l}^V \in \mathbb{R}^{d \times d_q} = \mathbb{R}^{768 \times 64} ...  value \textit{weight} matrices$$ 

$$W_l^{ffnn} \in \mathbb{R}^{d \times d_{ffnn}} = \mathbb{R}^{768 \times 3072} ...  feedforward layer's weight matrix$$

$$b_l^{ffnn} \in \mathbb{R}^{1 \times d_{ffnn}} = \mathbb{R}^{1 \times 3072} ...  feedforward layer's bias vector$$

$$W_l^{out} \in \mathbb{R}^{d_{ffnn} \times d} = \mathbb{R}^{3072 \times 768} ...  output layer's weight matrix$$ 

$$&b_l^{out} \in \mathbb{R}^{1 \times d} = \mathbb{R}^{1 \times 768} ...  output layer's bias vector$$

$$&W^{final} \in \mathbb{R}^{d \times d} = \mathbb{R}^{768 \times 768} ... final layer's weight matrix
$$

$$I=(i_1,\hdots,i_{512}) \in \mathbb{N}_0^{1 \times \text{max input length}} = \mathbb{N}_0^{1 \times 512} ...  input vocab indices$$

$$T=\texttt{lookup}(W_T,I) \in \mathbb{R}^{\text{max input length} \times d} = \mathbb{R}^{512 \times 768} ...  input token embeddings$$

$$X = T + W_P  \in \mathbb{R}^{\text{max input length} \times d} = \mathbb{R}^{512 \times 768} ...  input embeddings$$

$$Z_0=X$$


\pagebreak 
\noindent Forward algorithmm: \\

\begin{algorithm}[H]
\SetAlgoLined
  \For{$l=1;\ l \leq n_{\text{layers}}=12;\ l++$}{
       \For{$h=1;\ h \leq n_{\text{heads}}=12;\ h++$}{
        \begin{align*}
        &Q_{h,l}=Z_{l-1}W_{h,l}^{Q} \in \mathbb{R}^{\text{max input len} \times d_q} = \mathbb{R}^{512 \times 64} \hdots \text{query matrix}\\
        &K_{h,l}=Z_{l-1}W_{h,l}^{K} \in \mathbb{R}^{\text{max input len} \times d_k} = \mathbb{R}^{512 \times 64} \hdots \text{key matrix}\\
        &V_{h,l}=Z_{l-1}W_{h,l}^{V} \in \mathbb{R}^{\text{max input len} \times d_v} = \mathbb{R}^{512 \times 64} \hdots \text{value matrix}\\
        &A_{h,l} = \texttt{Softmax}(\frac{Q_{h,l}K_{h,l}^T}{\sqrt{d_k}}) \in \mathbb{R}^{\text{max input len} \times \text{max input len}} = \mathbb{R}^{512 \times 512}\\
        &Z_{h,l} = A_{h,l}V_{h,l} \in \mathbb{R}^{\text{max input len} \times d_v} = \mathbb{R}^{512 \times 64}
        \end{align*}
        \vspace{-0.5cm}
       }
    \vspace{-0.5cm}
    \begin{align*}
    &\tilde{Z}_l = \texttt{concat}(Z_{1,l}, \hdots,Z_{n_{\text{heads}},l}) \in \mathbb{R}^{\text{max input len} \times (d_v \cdot n_{\text{heads}})} = \mathbb{R}^{512 \times (64 \cdot 12)} = \mathbb{R}^{512 \times 768}\\
    &\bar{Z_l} = \texttt{LayerNorm}(X+\tilde{Z_l}) \in \mathbb{R}^{512 \times 768}\\
    &Z_l^{ffnn}=\max(0, \bar{Z_l}W_l^{ffnn}+b_l^{ffnn}) \in \mathbb{R}^{\text{max input len} \times d_{ffnn}} = \mathbb{R}^{512 \times 3072}\\
    &Z_l^{out} = Z_l^{ffnn}W_l^{out} + b_l^{out} \in  \mathbb{R}^{\text{max input len} \times d} = \mathbb{R}^{512 \times 768}\\
    &Z_l = \texttt{LayerNorm}(X+Z_l^{out}) \in \mathbb{R}^{512 \times 768}\\
    \end{align*}
    \vspace{-1cm}
 }
\end{algorithm}

\noindent Pass $\text{tanh}(W^{final}Z_{n_{\text{layers}}}[0,:])$ to the final \texttt{Softmax} that predicts the class, where $Z_{n_{\text{layers}}}[0,:]$ is the hidden state corresponding to the first token.

