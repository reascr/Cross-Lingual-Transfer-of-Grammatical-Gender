#### Cross-Lingual Transfer of Grammatical Gender in multilingual BERT (M-BERT). 


This repo contains code to study the transfer of grammatical gender in mBERT.

This study investigates the zero-shot cross-lingual transfer of grammatical gender using multilingual BERT (M-BERT) embeddings, focusing on both universal and languagespecific factors influencing gender prediction. The findings demonstrate that M-BERT
can transfer gender information across languages with diverse scripts and linguistic families, indicating that grammatical gender is encoded within a unified multilingual feature
space. Analysis reveals that M-BERT’s middle to higher layers capture gender in a largely
language-agnostic manner, with both formal and semantic features playing significant roles
in gender prediction. The study further shows that languages with isomorphic gender systems, such as Arabic-Italian and German-Greek, exhibit stronger transfer performance,
while non-isomorphic systems, like German-Danish, present greater challenges. Furthermore, animacy affects gender prediction in Russian, with inanimate nouns outperforming
animate nouns. Notably, the findings also show that cross-lingual transfer can be successful for languages not explicitly trained on by M-BERT, such as Beja, emphasizing
the universality of gender assignment features. Despite these promising results, the study
identifies several limitations, including the absence of a robust baseline and variations in
sample sizes, which should be addressed in future work. Further research should explore
the specific mechanisms by which M-BERT processes gender and expand the study to include more languages and larger models, such as M-GPT, to improve the generalizability
and understanding of cross-lingual gender transfer with LLMs
