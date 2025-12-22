## Changes from original -> custom MambaVision

- in_dims -> patch_embed_dim (default to 256)
- dims in list form for each level instead of single integer for the first level
- resolution 224 -> 256 to make it in line with CD patches
- MambaVision forward() will return outputs at multiple levels
