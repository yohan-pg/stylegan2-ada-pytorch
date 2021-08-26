IDEAS
-----
- double check that it works
    - Try with and without skip on AdaIN 
    - Double check that our AdaIN gives the same thing as theirs
    - Add a learnable parameter for the skip (mimics their adain)
    - Try training without style injection... Can we still learn something using only the noise latents?
    - Try training without the noise latents. 
    - Try injecting at only one place.
    - Try to add a normalization layer. In principle we should still be able to train (first try their code, then our adain, then their code).
    - Try adding the bias back? This is what made the difference with genforce.
    - Try injecting in a StyleGAN1?

PROBLEMS
-----
- stupid wasteful cloning in the mapper
- Can't train on liszt - only tch works.
- There seems to be a leaking memory problem. Is it the eval? -> no, we had it disabled...
- PPL doesn't seems OK with adaconv.
- Style mixing doesn't work.

TODO
-----
- Measure training times
- Plug in churches
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