import torch
from LeNet5 import LeNet


model = LeNet()
model.cuda()
model.load_state_dict(torch.load('../../train/LeNet5_MNIST_parameter.pth'))


"""
for parameters in model.parameters():
    print(parameters)

import cv2
from torch.autograd import Variable
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

img = cv2.imread("a8.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = np.array(img).astype(np.float32)
img = np.expand_dims(img, 0)
img = np.expand_dims(img, 0)
img = torch.from_numpy(img)
img = img.to(device)
output = model(Variable(img))
prob = torch.nn.functional.softmax(output, dim=1)
prob = Variable(prob)
prob = prob.cpu().numpy()
pred = np.argmax(prob)
print(pred.item())
"""


example = torch.rand(64, 1, 28, 28).cuda()
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("trace_model.pt")
