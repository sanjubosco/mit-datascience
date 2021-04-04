import torch 
import torch.nn as nn

input = torch.randn(1,1,28,28)
print (input.shape)
c1 = nn.Conv2d(1,32,3)
p1 = nn.MaxPool2d(2)
print (c1(input).shape)

out1 = p1(c1(input))
print (out1.shape)

c2 = nn.Conv2d(32,64,3)
print (c2(out1).shape)
p2 = nn.MaxPool2d(2)
out2 = p2(c2(out1))

print (out2.shape)

fl = nn.Flatten(out2)

print (fl.shape)