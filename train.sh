
    # To compile the sources, the default system compiler (``c++``) is used,
    # which can be overridden by setting the ``CXX`` environment variable. To pass
    # additional arguments to the compilation process, ``extra_cflags`` or
    # ``extra_ldflags`` can be provided. For example, to compile your extension
    # with optimizations, pass ``extra_cflags=['-O3']``. You can also use
    # ``extra_cflags`` to pass further include directories.

export TORCH_EXTENSIONS_DIR=extensions
python train.py --outdir=./training-runs --data=./datasets/afhq32cat.zip --snap 10 --gpus=1 --metrics=none

