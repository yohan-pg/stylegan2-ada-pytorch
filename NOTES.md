IDEAS
-----
- try injecting more or less strongly
- log syle std thru training 
- Try training without the noise latents. 
- try training with a fixed instead of a learnable bias for the affine layer
- Try injecting at only one place.
- blockwise diag

PROBLEMS
-----
- PPL doesn't seems OK with adaconv.
- Style mixing doesn't work. Letting is go though is what causes in-place errors, and then cloing ws at the end of the mapper tensor seems to have caused the memory leak!?! (I hope)
- Can't train on liszt - only tch works.

QUESTIONS
-----
what do the torgb layers do in sgan2?

TODO
-----
- we got a performance regression in adain64? (it's the batch size?)
- Measure training times
- Plug in churches
- What do the
- Plug in FFHQ
- Autoscreen in the training script
- Train some comparative models
    - 64 for 5m images
    - 64 for 25m images 
    - 64 with W+
    - 64 without PPL
    - 64 without style mixing 
    - 64 without noise injection?
    - 64 without the 2nd moment normalization?
- Build the optimization experiments
    - Optimize on Z 
    - Optimize on W+
    - Optimize on our style, Z/W/W+
    - Understand why MSE fails 