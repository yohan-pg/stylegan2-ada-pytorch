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

# High prio

*) Make sure removing it also kills the training performance -> IT DOES

*) Try style size 512 -> IT WORKS

*) Bring back ToRGB
    *) Try injecting in the final toRGB -> IT WORKS, although there is more variety (color hue) early in training. Should compare quality.
    *) Try going back to old config (skip) with our injection -> IT WORKS, same comment applies

*) Try different batch sizes
    *) Batch size 4 (Can we run tests very efficiently?) -> IT WORKS, but doesn't seem much faster than 8 for the same kimg. I will keep 8.
    *) Batch size 32 (Will it still work in our final tests?)

*) Try different gamma factors
    *) It fails when gamma is 1. Let's just keep 10 for now.
    -) Can we make the noise work when the gamma factor is OK?

=) Bring back PPL -> It seems to work, but will need to review that it performs well
    *) Train with it -> It doesn't damage the training whatsoever
    -) Combine with style mixing
    -) Verify that the levels are similar
    -) Double check that the math being done makes sense. Are the values comparable to the old?

=) Bring back style mixing 
    *) Train with it -> It very trains, but looks worse early on?
    -) Turn it off, and double check that the performance didn't regress because of some other detail
    -) Make it work combined with the PPL

=) Bring back optimizable noise. The issue seems to be, the network ignores the style, and only uses the noise. -> IT FAILS
    *) It seems to get out of the collapse past a certain point! Is the performance still as good?
        -) Comapre performance with and without.
        -) Compare without the style mixing. Was it the mixing that solved everything?
    -) Try with the whole stylegan2 config
    -) Try to train with adaconv

-) Try baking it as a convolution! 
    -) Train with it 
    -) Try to combine with the noise naïvely. Does it make a difference?
    -) Work out the math. Why are there transpositions?

-) Make sure the metrics are still good.
    -) Recall seems low. What gives?
        -) Maybe it's the perceptual distance that is falling appart on 32x32. But then, why doesn't it suck with the baseline?
            -) Try on a higher resolution. 
            -) Try with MSE instead of perceptual?
        -) What is the impact of the gamma factor on the recall, when using our baseline?


# Low prio
-) Once finished, abalate every change (noise, ppl, gamma, mixing, etc.) to understand their effect better
-) Try on a higher resolution
-) Hyperparameter search on the ideal gamma factor, but only once everything else is in place. 
-) Double check the SG2 configuration (which will be our final config)
-) Can we train on W+ with our thing?
-) Double check quality with/without injection in ToRGB 
