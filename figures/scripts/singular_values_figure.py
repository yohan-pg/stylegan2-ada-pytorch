 class log_singular_values:
        layers = [
            ("mapping_fc0", G.mapping.fc0),
            ("mapping_fc1", G.mapping.fc1),
            ("synthesis_b16_affine", G.synthesis.b16.conv1.affine),
            ("synthesis_b64_affine", G.synthesis.b64.conv1.affine),
        ]

        plt.figure()
        
        for name, layer in layers:
            U, S, V = layer.weight.svd()
            plt.plot(S.detach().cpu())
            plt.title(f"Singular values in layer {name}")
        
        plt.legend([name for name, _ in layers])
        plt.ylabel("Singular Value")
        plt.savefig(f"{OUT_DIR}/singular_values.png")