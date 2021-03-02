from torchsummary import summary

#pour Out: torch.Size([1, 2, 512, 512])

summary = summary(model, (1, 512, 512))