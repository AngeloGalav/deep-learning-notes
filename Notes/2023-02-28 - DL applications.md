# Some applications of DL

## NLP
###  Key Technologies 
- __tokenization__: splitting the input sentence into relevant lexical components (characters/words/subwords), and coding them into numbers. Byte-Pair Encoding, WordPiece, SentencePiece, etc. see this tokenizer summary 
- __transformers__: a feed-forward deep learning model adopting self-attention for weighting the mutual significance of tokens in the sentence 
- __word embeddings__ a semantic embedding of words, mostly used for text similarity, text retrieval, code search, etc. Examples are Word2Vec, Glove. Transformers do not use them: they learn their own embeddigs. see this blog for a comparison of state-of-the-art text embeddings
 
## Generative modeling
Train a _generator_ able to sample original data similar to those in the training set, implicitly learning the _distribution of data_.
![[generator.png]]
- the randomicity of the generator is provided by a _random seed_ (noise) received as input. 
-
