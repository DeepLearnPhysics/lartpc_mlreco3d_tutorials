---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.3
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
execution:
  timeout: 180
---


# Shared metrics

These pages are meant to share **reproducible metrics** for each stage of the
full chain. Feel free to open them in Binder to customize the visualization.

There are currently two types of shared metrics:
* [Raw](./Raw.md) means that we look directly at the output of the full chain, without
any intermediate (which can introduce biases and bugs of their own)
* [Analysis](./Analysis.md) means that we look at the metrics using the so-called analysis
tools of the full chain, which are meant to make its output more digestible
and user-friendly. They can have their own underlying assumptions (e.g.
minimum particle voxel count, etc) so this should be considered separately
from the "raw" metrics.

```{note} What if my results disagree with the metrics shown here?

Please contact us if your analysis leads to a very different performance!
Feedback is welcome, it's an opportunity to understand what might have
gone wrong in the reconstruction chain, potentially.
```
