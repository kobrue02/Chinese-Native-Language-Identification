Native Language Identification (NLI) is a text classification task with the goal of identifying an author's first language (L1) based on their second language writing (L2). This project investigates NLI methods applied to Chinese texts written by language learners from a variety of first languages.

\section{Research Questions}
How do different computational methods (e.g., traditional feature based approaches, multilingual encoder models, chinese encoder models, ...) perform in identifying the L1 of learners from their Chinese writing?  
Which linguistic features are most discriminative for NLI? Does writing context affect classification performance?

\section{Dataset}
The Jinan Chinese Learner Corpus \cite{wang-etal-2015-jinan} contains Chinese texts from Chinese learners with metadata including document ID, writing circumstance (e.g., Exam), native country/language, and gender. It contains more than 8,000 documents from a variety of L1s and contexts.
    
\section{Methodology}
Three approaches will be implemented and compared:
\begin{enumerate}
\item \textbf{Traditional ML:} Feature-based classification using linguistic features such as n-grams, POS patterns, and syntactic features with SVM classifier (this has worked well for English \cite{zampieri-etal-2017-native}, \cite{kulmizev-etal-2017-power})
\item \textbf{Neural Network:} LSTM/CNN architecture with pre-trained word embeddings, which appear to perform worse than linear SVMs in English \cite{kulmizev-etal-2017-power}
\item \textbf{Transformer-based:} Fine-tuned BERT, XLM-RoBERTa or Chinese encoder models. BERT has shown promising results in English \cite{steinbakken-gamback-2020-native}.
\end{enumerate}

\section{Summary}
In essence, this work will explore whether popular NLI approaches that work well in Western languages also do well in Chinese, where there has been less research on the topic so far. While I do not expect to come up with a super novel approach or beat some SOTA, I think it will be a meaningful contribution to the broader field of NLI, particularly in light of non-Western languages.
