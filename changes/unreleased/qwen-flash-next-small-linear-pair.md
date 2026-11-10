## New features and improvements

- Qwen3.8-Flash-Next's decode window launches the router and shared-expert gate, and the GDN `a` and `b` projections, as one two-weight kernel instead of two launches. The outputs are the bits of two separate calls (checked by a new test).
