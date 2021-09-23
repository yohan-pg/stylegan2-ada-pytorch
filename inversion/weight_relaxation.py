    # with torch.no_grad():
    #     for i in range(7):
    #         layer = getattr(G.mapping, f"fc{i}")
    #         old = layer.weight.clone()
    #         U, S, V = layer.weight.svd()
    #         layer.we
    # ight.copy_(U @ torch.diag_embed(S + S.max() * 0.1) @ V.t())
