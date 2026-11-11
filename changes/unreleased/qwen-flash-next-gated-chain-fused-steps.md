## New features and improvements

- The gated delta-net prompt chain (one block per value head, 8,192-row chunks) runs each step's decay and key product inside the previous step's state update, so the state's registers are swept once a step instead of twice and the key chunk is loaded once for both uses. Every operation on every state element is the previous kernel's, in the same order: outputs, final state and the conv history are bit for bit the same. One layer's prefill step (pre-processing, chain, gated norm and conv commit) at 8,192 rows takes 5.81 ms instead of 6.55 ms.
