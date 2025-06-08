---
jupytext:
  cell_metadata_filter: -all
  formats: source/datasets///ipynb,jupyterbook/datasets///md:myst,jupyterbook/datasets///ipynb
  main_language: python
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.5
kernelspec:
  display_name: python3
  language: python
  name: python3
---

# Khan Gene Data

The data consists of a number of tissue samples corresponding to
four distinct types of small round blue cell tumors. For each
tissue sample, 2308 gene expression measurements are available.

     
## Format

The format is a dict containing four components: `xtrain`,
`xtest`, `ytrain`, and `ytest`. `xtrain` contains the 2308 gene
expression values for 63 subjects and `ytrain` records the
corresponding tumor type. `ytrain` and `ytest` contain the
corresponding testing sample information for a further 20
subjects.

## Notes

This data were originally reported in:

- Khan J, Wei J, Ringner M, Saal L, Ladanyi M, Westermann F,
Berthold F, Schwab M, Antonescu C, Peterson C, and Meltzer P.
Classification and diagnostic prediction of cancers using gene
expression profiling and artificial neural networks. Nature
Medicine, v.7, pp.673-679, 2001.

The data were also used in:

- Tibshirani RJ, Hastie T, Narasimhan B, and G. Chu. Diagnosis of
Multiple Cancer Types by Shrunken Centroids of Gene Expression.
Proceedings of the National Academy of Sciences of the United
States of America, v.99(10), pp.6567-6572, May 14, 2002.

```{code-cell}
from ISLP import load_data
Khan = load_data('Khan')
Khan.keys()
```

```{code-cell}
for X in ['xtest', 'xtrain']:
    print(Khan[X].shape)
```

```{code-cell}
for Y in ['ytest', 'ytrain']:
    print(Khan[Y].value_counts())
```
