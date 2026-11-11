## New features and improvements

- The large-row EXL3 linear (the Flash-Next shared expert's three projections during prefill) runs the column strips of a 64-row group together when a layer has at most 24 strips of 128 columns, instead of the row groups of one strip. A narrow layer's whole weight set is a few MB, so it stays in L2 while the rotated input streams from DRAM once; the old order read the input once a strip. Which block runs when never changes a bit. On 8,192-row chunks the shared expert's launches average 726.7 us instead of 1,150.4 us each.
