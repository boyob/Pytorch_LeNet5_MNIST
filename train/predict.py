import torch
import cv2
from torch.autograd import Variable
import numpy as np


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load('LeNet5_MNIST_parameter_and_model.pth')
    model = model.to(device)
    model.eval()

    img = cv2.imread("digits/a8.jpg")
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
    #print(prob)
    pred = np.argmax(prob)
    print(pred.item())
