## Cross-Lingual Transfer of Grammatical Gender in multilingual BERT (M-BERT). 


This repository contains code and experiments related to the study of zero-shot cross-lingual transfer of grammatical gender using multilingual BERT (M-BERT). 

#### Citation:

Schröter, A., & Basirat, A. (2025, November). Universal Patterns of Grammatical Gender in Multilingual Large Language Models. In Proceedings of the 5th Workshop on Multilingual Representation Learning (MRL 2025) (pp. 34–46). [PDF](https://aclanthology.org/2025.mrl-main.3/)

#### Project Summary
This study investigates how M-BERT encodes grammatical gender across languages, focusing on both universal and language-specific factors. It explores how gender information is captured in M-BERT's embeddings and how well this transfers between languages, including those with different scripts and linguistic families.

##### Key findings:

- Gender information is encoded in M-BERT's middle-to-upper layers in a largely language-agnostic way.

- Formal and semantic features influence gender prediction performance.

- Isomorphic gender systems (e.g., Arabic–Italian, German–Greek) lead to stronger transfer, while non-isomorphic systems (e.g., German–Danish) pose challenges.

- Animacy matters: In Russian, inanimate nouns were predicted more reliably than animate ones.

- Zero-shot transfer is possible even to languages not explicitly trained on, such as Beja.

#### Usage 

##### Install dependencies:

pip install -r requirements.txt

##### Run the main script:

python scripts/Layer-wise-analysis-mBERT.py


