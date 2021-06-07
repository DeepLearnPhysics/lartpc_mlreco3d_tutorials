# Reconstruction outputs

If you want to visualize the predictions made by the reconstruction chain,
you are at the right place. Select which specific stage of the chain you are interested
in the menu on the left, or keep reading.

## Visualizing with Plotly
A typical way of visualizing either inputs or outputs of the chain is using Plotly.

```
import plotly
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode(connected=False)
```

`lartpc_mlreco3d` has a few helpers to facilitate the visualization of common
quantities. For example, the `scatter_points` function will display a 3D sparse tensor
and `plotly_layout3d` provides a default layout for the Plotly 3D visualization.

```
from mlreco.visualization import scatter_points, plotly_layout3d
```

You can then have just a few lines of code to visualize your tensor:

```
trace = []

trace+= scatter_points(input_data,markersize=1,color=input_data[:, -1], cmin=0, cmax=10, colorscale=plotly.colors.qualitative.D3)
trace[-1].name = 'True semantic labels (true no-ghost mask)'

fig = go.Figure(data=trace,layout=plotly_layout3d())
fig.update_layout(legend=dict(x=1.0, y=0.8))

iplot(fig)
```

## Helper functions for Plotly visualization in `lartpc_mlreco3d`

```{note} TODO
```