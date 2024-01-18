# Classification Excel Naming Rule

## old_format

    {3CLS_BY_SurfStDev}_data.xlsx
    {4CLS_BY_SurfStDev}_data.xlsx
    ...

Note: One xlsx can contains different `StDev` sheet for classification.

## new_format ( After adding the KMeans classification strategy )

    {3CLS_SURF_050STDEV}_data.xlsx
    {3CLS_SURF_075STDEV}_data.xlsx
    {3CLS_SURF_100STDEV}_data.xlsx
    {4CLS_SURF_050STDEV}_data.xlsx
    {4CLS_SURF_075STDEV}_data.xlsx
    {4CLS_SURF_100STDEV}_data.xlsx
    {3CLS_SURF_KMeansLOG10_RND2022}_data.xlsx
    {3CLS_SURF_KMeansORIG_RND2022}_data.xlsx
    {4CLS_SURF_KMeansLOG10_RND2022}_data.xlsx
    {4CLS_SURF_KMeansORIG_RND2022}_data.xlsx
    ...

Note:

1. Each xlsx file can only represent one classification strategy.
2. The `old_format` parsing is deprecated.
