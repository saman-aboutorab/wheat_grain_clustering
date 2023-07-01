# Wheat Grain clustering
:::

::: {.cell .code execution_count="14" id="Hj7YmKZsrhtI"}
``` python
# Perform the necessary imports
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Import pandas
import pandas as pd

import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.cluster.hierarchy import fcluster

# Import TSNE
from sklearn.manifold import TSNE

```
:::

::: {.cell .markdown}
![Wheat](vertopal_73d1a1d69379457abb6d22c57aa8a118/fd940dbb6731fe1a1b540673476a781a5ac1f0fd.jpg)
:::

::: {.cell .markdown id="xX20AVWmtQjC"}
## setup dataset
:::

::: {.cell .code execution_count="22" id="aIO2XMW0r8yN"}
``` python
seeds_df = pd.read_csv('seeds.csv')

column_names = ['measure1', 'measure2','measure3','measure4','measure5','measure6','measure7','type']

seeds_df.columns = column_names

X_seeds_df = seeds_df.drop('type', axis=1)
samples = X_seeds_df.to_numpy()

grain_type = seeds_df['type']

seeds_df2 = pd.read_csv('seeds-width-vs-length.csv', header=None)

samples2 = seeds_df2.to_numpy()
```
:::

::: {.cell .markdown id="myPi87-0tM4m"}
## Find best cluster number
:::

::: {.cell .code execution_count="8" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":648}" id="u-AWdDKirnss" outputId="5501aae3-0b28-41b8-ccc3-81f8457048f4"}
``` python
ks = range(1, 6)
inertias = []

for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)

    # Fit model to samples
    model.fit(samples)

    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)

# Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()
```

::: {.output .stream .stderr}
    /usr/local/lib/python3.10/dist-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    /usr/local/lib/python3.10/dist-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    /usr/local/lib/python3.10/dist-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    /usr/local/lib/python3.10/dist-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    /usr/local/lib/python3.10/dist-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
:::

::: {.output .display_data}
![](vertopal_73d1a1d69379457abb6d22c57aa8a118/47d228d53631ee5472f7b7fa1db243ea3a13c125.png)
:::
:::

::: {.cell .markdown id="qotPJJ3UtKW4"}
## Evaluating the grain clustering
:::

::: {.cell .code execution_count="10" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="0YdnN1SEs5BP" outputId="e01d30fa-2cba-458a-901c-79a17ad82888"}
``` python
# Create a KMeans model with 3 clusters: model
model = KMeans(n_clusters=3)

# Use fit_predict to fit model and obtain cluster labels: labels
labels = model.fit_predict(samples)

# Create a DataFrame with labels and varieties as columns: df
df = pd.DataFrame({'labels': labels, 'type': grain_type})

# Create crosstab: ct
ct = pd.crosstab(df['labels'], df['type'])

# Display ct
print(ct)
```

::: {.output .stream .stdout}
    type     1   2   3
    labels            
    0        1  60   0
    1        9   0  68
    2       59  10   2
:::

::: {.output .stream .stderr}
    /usr/local/lib/python3.10/dist-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
:::
:::

::: {.cell .markdown id="KlzJMNjn5vAd"}
## Hierarchical clustering of the grain data
:::

::: {.cell .code execution_count="10" id="sDSKdUs96qHA"}
``` python
varieties = ['Kama wheat', 'Kama wheat', 'Kama wheat', 'Kama wheat', 'Kama wheat', 'Kama wheat', 'Kama wheat', 'Kama wheat', 'Kama wheat', 'Kama wheat', 'Kama wheat', 'Kama wheat', 'Kama wheat', 'Kama wheat', 'Rosa wheat', 'Rosa wheat', 'Rosa wheat', 'Rosa wheat', 'Rosa wheat', 'Rosa wheat', 'Rosa wheat', 'Rosa wheat', 'Rosa wheat', 'Rosa wheat', 'Rosa wheat', 'Rosa wheat', 'Rosa wheat', 'Rosa wheat', 'Canadian wheat', 'Canadian wheat', 'Canadian wheat', 'Canadian wheat', 'Canadian wheat', 'Canadian wheat', 'Canadian wheat', 'Canadian wheat', 'Canadian wheat', 'Canadian wheat', 'Canadian wheat', 'Canadian wheat', 'Canadian wheat', 'Canadian wheat']

samples_variety = samples[:42]
```
:::

::: {.cell .code execution_count="11" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":484}" id="qTxdA2s-5vby" outputId="ed222f06-aca6-496c-a46c-8f7b0908f8e2"}
``` python
# Calculate the linkage: mergings
mergings = linkage(samples_variety, method='complete')

# Plot the dendrogram, using varieties as labels
dendrogram(mergings,
           labels=varieties,
           leaf_rotation=90,
           leaf_font_size=6,
)
plt.show()
```

::: {.output .display_data}
![](vertopal_73d1a1d69379457abb6d22c57aa8a118/067586d03736c26fc8eb6c0a66dbb5948cf25c7e.png)
:::
:::

::: {.cell .code execution_count="28" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="OEwAgBBw9V7b" outputId="edc0c695-3859-4fda-a861-d4aceca161b9"}
``` python
# Use fcluster to extract labels: labels
labels = fcluster(mergings, 6, criterion='distance')

# Create a DataFrame with labels and varieties as columns: df
df = pd.DataFrame({'labels': labels, 'varieties': varieties})

# Create crosstab: ct
ct = pd.crosstab(df['labels'], df['varieties'])

# Display ct
print(ct)
```

::: {.output .stream .stdout}
    varieties  Canadian wheat  Kama wheat  Rosa wheat
    labels                                           
    1                       4           3           2
    2                      10          11          12
:::
:::

::: {.cell .markdown id="hzBfGSW8tmot"}
## Result

The cross-tabulation shows that the 3 varieties of grain separate really
well into 3 clusters.
:::

::: {.cell .code id="JZE2PkButbLW"}
``` python
```
:::
