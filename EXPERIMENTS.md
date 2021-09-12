# Sep 10
- Reduced batch size to 4 for all subsequent experiments
- With a frozen mapper, and without the toRGB layers (resnet architecture + removed injection in the last ToRGB), tried to inject a different # of style vectors
    -> it keeps training up to 16 (so far! trying to get as high as possible). However, some weird "wobly" artifacts appear, and take longer and

Questions:
- Does increasing the resolution help at all? (dilute out the style in comparatively more pixels) -> NO
- Does adding noise back in help? -> NO, it kills it again (total collapse)
- Does training the mapper help? -> a bit but not much, at least it still trains
- Does attenuating the style or amplifying it help? -> amplifying (multiply (1 - s) by a factor, sqrt(D) or K) does seem to help, but we still collapse to less images.


#  Sep 11
THE PROBLEM IS WITH THE GAMMA FACTOR

*) Make sure removing it also kills the training performance -> IT DOES
*) Try style size 512 -> IT WORKS
*) Bring back ToRGB
    *) Try injecting in the final toRGB -> IT WORKS, although there is more variety (color hue) early in training. Should compare quality.
    *) Try going back to old config (skip) with our injection -> IT WORKS, same comment applies
*) Try different batch sizes
    *) Batch size 4 (Can we run tests very efficiently?) -> IT WORKS, but doesn't seem much faster than 8 for the same kimg. I will keep 8.
    *) Batch size 32 (Will it still work in our final tests?)
*) Bring back PPL
-) Try baking it as a convolution! 
    -) First make it a separate conv
    -) Then try baking it

-) Bring back optimizable noise -> IT FAILS
    ... The issue seems to be: the noise learns faster... we can try to reduce the LR on the noise

-) Bring back style mixing
-) Try on a higher resolution
-) Double check the SG2 configuration (which will be our final config)
-) Try different gamma factors
-) If quality is poorl, double check quality with/without injection in ToRGB 