# Sep 10
- Reduced batch size to 8 for all subsequent experiments
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

*) Try number of style vectors 512 -> IT WORKS

*) Bring back ToRGB
    *) Try injecting in the final toRGB -> IT WORKS, although there is more variety (color hue) early in training. Should compare quality.
    *) Try going back to old config (skip) with our injection -> IT WORKS, same comment applies

*) Try different batch sizes
    *) Batch size 4 (Can we run tests very efficiently?) -> IT WORKS, but doesn't seem much faster than 8 for the same kimg. I will keep 8.
    *) Batch size 32 (Will it still work in our final tests?)

=) Try different gamma factors
    *) It fails when gamma is 1. Let's just keep 10 for now, as this is what stylegan uses.
    -) Try gamma 100: it's what they use for church!

=) Bring back style mixing 
    *) Fixed the division to be exactly between groups of 512
    *) Train with it -> It trained without the groups of 512, it will surely work with it as well...
    -) Make it work for wplus sampling, both for adain and adaconv

=) Try baking it as a convolution! 
    *) Implement it.
    -) Make it train.
    -) How to make it worked unbaked?
    -) What is the cost of 3x3 vs 1x1?
    -) Try to combine with the noise naïvely. Does it make a difference?
    -) Work out the math. Why are there transpositions?

=) Bring back PPL regularization -> It seems to work, but will need to review that it performs well
    *) Fix problem with the gradient -> It's their custom op. They train WITHOUT fusion, and then add the `conv2d_gradfix.no_weight_gradients()` flag when measuring the PPL? I'm so confused.
    *) Train with it.
    -) Train AdaIN fused and unfused, to compare that our performance deosn't drop.
    -) The levels are not similar, because we are averaging. Train with and without, to compare. Train our adaconv, and look at the PPL loss
    -) Double check that the math being done makes sense. Are the values comparable to the old?

=) Bring back optimizable noise. The issue seems to be, the network ignores the style, and only uses the noise. -> IT FAILS
    *) It seems to get out of the collapse past a certain point! Is the performance still as good?
    -) Try with the whole stylegan2 config
    *) Gamma 100 doesn't save us

-) Retry freezing the mapper? Make sure it doesn't help.

-) They don't use fused modconv during training! But the result is different from during inference!? WTF is going on?

-) Make sure the metrics are still good.
    -) Recall seems low. What gives?
        -) Maybe it's the perceptual distance that is falling appart on 32x32. But then, why doesn't it suck with the baseline?
            *) Try on a higher resolution. 
            *) We get a decent gamma score when the style collapses and all the training happens in the noise (0.17 after just 200k imgs). Maybe we just need the noise?
            -) Try with MSE in the eval code instead of perceptual?
        -) What is the impact of the gamma factor on the recall, when using our baseline?

*) Bring back data augmentation.

-) Get 16 bit training working again! Provided it is easy.

-) Measure optimisation speed on 128 
    *) Train 128 for 1M
    -) Train a baseline for 1M
    -) Update the optim script and compare

# Real Conv
- With the SG2 normalization applies to the style matrix, it does not seem to train. noise/no-noise, plus id/no plus id (centered at 0), gamma 10/100.
- Without it, it does seem to train. So far I tried without plus id, just centered at 0. With plus ID it seems to work negligably better. Mul by sqrt(512) (before the +I) also works, but appears to slow down convergence a bit... Div by sqrt(512) performs a bit better than both. 
- Next up, try with the noise again, then with baking (without plus ID, different factors, etc.). 

# Low prio
-) Once finished, abalate every change (noise, ppl, gamma, mixing, etc.) to understand their effect better
-) Hyperparameter search on the ideal gamma factor, but only once everything else is in place. 
-) Double check the SG2 configuration (which will be our final config)
-) Can we train on W+ with our thing?
-) Double check quality with/without injection in ToRGB 



