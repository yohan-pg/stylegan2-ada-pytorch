import torch 

W = torch.randn(512, 512, 3, 3)
S = torch.randn(4, 512, 512)
x = torch.randn(512)

W2 = torch.zeros_like(W).repeat(4, 1, 1, 1, 1)

for k in range(len(S)):
    for i in range(3):
        for j in range(3):
            W2[k, :, :, i, j] = W[:, :, i, j] @ S[k]

W3 = (W.reshape(512, 512, -1).transpose(2, 1).transpose(1, 0).unsqueeze(0) @ S.unsqueeze(1)).transpose(2, 1).transpose(3, 2).reshape(S.shape[0], *W.shape)

target = W[:, :, 0, 0] @ (S[0] @ x)
(target - W2[0, :, :, 0, 0] @ x).abs().max()
(target - W3[0, :, :, 0, 0] @ x).abs().max()