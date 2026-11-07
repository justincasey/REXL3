## New features and improvements

- GLM P2 prefill (`REXL3_GLM_PREFILL_CONTRACT=p2`) now runs the RED-SNOW 4.91bpw pack, on one node and at TP=2. The P2 expert GEMMs take 6-bit experts. A tensor-parallel rank whose share of a mixed-width layer is several subbanks runs the P2 pairs stage once per subbank, then does the usual part swap. A split gate/up subbank (gate and up of different widths) runs its pairs through the decode pairs stage in windows. The prefix namespace and the TP Hello name this as `split-gu=p1`.
