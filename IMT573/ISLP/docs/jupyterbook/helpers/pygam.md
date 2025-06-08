---
jupytext:
  cell_metadata_filter: -all
  formats: source/helpers///ipynb,jupyterbook/helpers///md:myst,jupyterbook/helpers///ipynb
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

# Generalized Additive Models

This module has helper functions to help
compute the degrees of freedom of a GAM and to create a partial dependence plot of a
fitted `pygam` model.

```{code-cell}
import numpy as np
from pygam import LinearGAM, s
from ISLP.pygam import (plot, 
                        approx_lam, 
                        degrees_of_freedom)
```

## Make a toy dataset

We create a simple dataset with 5 features.
We'll have a cubic effect for our first feature, and linear for the remaining 4 features.

By construction, all the "action" in our GAM will be in the first feature. This will have our 
scatter plot look like the partial residuals from our fit. Usually, the scatter plot will not
look so nice on a partial dependence plot. One should use partial residuals instead. We take this liberty
here while demonstrating the `plot` function.

```{code-cell}
rng = np.random.default_rng(1)
N = 100
X = rng.normal(size=(N, 3))
Y = X[:,0] + 0.3 * X[:,0]**3 + rng.normal(size=N)
```

## Create a GAM

Let's start of fitting a GAM with a relatively small amount of smoothing.

```{code-cell}
terms = [s(f, lam=0.01) for f in range(3)]
gam = LinearGAM(terms[0] + 
                terms[1] + 
                terms[2])
gam.fit(X, Y)
```

## Plot the partial dependence plot for first feature

```{code-cell}
ax = plot(gam, 0)
```

Including a scatter plot of

```{code-cell}
ax.scatter(X[:,0], 
           Y - Y.mean(),
          facecolor='k',
          alpha=0.4)
ax.get_figure()
```

Let's take a look at (approximately) how many degrees of freedom we've used:

```{code-cell}
[degrees_of_freedom(X,
                   terms[i]) for i in range(X.shape[1])]
```

## Fixing degrees of freedom

Suppose we want to use 5 degrees of freedom for each feature. 
We compute a value of `lam` for each that fixes the degrees of freedom at 5.

```{code-cell}
lam_vals = [approx_lam(X,
                       terms[i],
                       df=5) for i in range(X.shape[1])]
lam_vals
```

### Create a new GAM with the correctly fixed terms

```{code-cell}
fixed_terms = [s(f, lam=l) for 
               f, l in zip(range(3), lam_vals)]
fixed_gam = LinearGAM(fixed_terms[0] + 
                      fixed_terms[1] + 
                      fixed_terms[2])
fixed_gam.fit(X, Y)
```

```{code-cell}
ax = plot(fixed_gam, 0)
ax.scatter(X[:,0], 
           Y - Y.mean(),
          facecolor='k',
          alpha=0.4);
```
