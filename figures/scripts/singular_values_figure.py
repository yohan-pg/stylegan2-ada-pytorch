from prelude import * 

GENERATORS = {
    ("no_slowdown", "Regular Learning Rate"): "pretrained/old/cond-no-slowdown.pkl",
    ("slowdown", "Reduced Learning Rate"): "pretrained/old/cond.pkl",
}

# -----------------------------

OUT_PATH = "figures/out/singular_values_figure"

fresh_dir(OUT_PATH)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))

colors = iter(axes[0]._get_lines.prop_cycler)

for i, ((training_name, title), generator_path) in enumerate(GENERATORS.items()):
    G = open_generator(generator_path)
    color = next(colors)['color']

    layers = [
        G.synthesis.b16.conv0.affine,
        G.synthesis.b16.conv1.affine,
        G.synthesis.b32.conv0.affine,
        G.synthesis.b32.conv1.affine,
        G.synthesis.b64.conv0.affine,
        G.synthesis.b64.conv1.affine,
    ]

    Ss = []
    for layer in layers:
        U, S, V = layer.weight.svd()
        Ss.append(S)
    
    axes[i].set_title(title)
    axes[i].plot(torch.stack(Ss).mean(dim=0).detach().cpu(), c=color)
    axes[i].set_ylabel("Singular Value")
    
plt.suptitle("Singular Values")

plt.tight_layout()
plt.savefig(f"{OUT_PATH}/singular_values.png", pad_inches=0)
plt.savefig(f"{OUT_PATH}/singular_values.pdf", pad_inches=0)

