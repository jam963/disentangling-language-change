# Disentangling Language Change
Code and data for the paper ["Disentangling language change: sparse autoencoders quantify the semantic evolution of indigeneity in French"](https://aclanthology.org/2025.naacl-long.559/) (_NAACL_ 2025). 

Our dataset is available on Hugging Face [here](https://huggingface.co/datasets/jam963/indigeneity_fr).

# Citation 
```
@inproceedings{matthews-etal-2025-disentangling,
    title = "Disentangling language change: sparse autoencoders quantify the semantic evolution of indigeneity in {F}rench",
    author = "Matthews, Jacob A.  and
      Dubreuil, Laurent  and
      Terhmina, Imane  and
      Sun, Yunci  and
      Wilkens, Matthew  and
      Schijndel, Marten Van",
    editor = "Chiruzzo, Luis  and
      Ritter, Alan  and
      Wang, Lu",
    booktitle = "Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    month = apr,
    year = "2025",
    address = "Albuquerque, New Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.naacl-long.559/",
    pages = "11208--11222",
    ISBN = "979-8-89176-189-6",
    abstract = "This study presents a novel approach to analyzing historical language change, focusing on the evolving semantics of the French term {\textquotedblleft}indig{\`e}ne(s){\textquotedblright} ({\textquotedblleft}indigenous{\textquotedblright}) between 1825 and 1950. While existing approaches to measuring semantic change with contextual word embeddings (CWE) rely primarily on similarity measures or clustering, these methods may not be suitable for highly imbalanced datasets, and pose challenges for interpretation. For this reason, we propose an interpretable, feature-level approach to analyzing language change, which we use to trace the semantic evolution of {\textquotedblleft}indig{\`e}ne(s){\textquotedblright} over a 125-year period. Following recent work on sequence embeddings (O`Neill et al., 2024), we use $k$-sparse autoencoders ($k$-SAE) (Makhzani and Frey, 2013) to interpret over 210,000 CWEs generated using sentences sourced from the French National Library. We demonstrate that $k$-SAEs can learn interpretable features from CWEs, as well as how differences in feature activations across time periods reveal highly specific aspects of language change. In addition, we show that diachronic change in feature activation frequency reflects the evolution of French colonial legal structures during the 19th and 20th centuries."
}
```
