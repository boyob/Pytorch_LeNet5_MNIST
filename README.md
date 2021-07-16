# Pytorch_LeNet5_MNIST

# 1. 准备数据

1. 下载二进制数据集：在 prepareData 目录内运行 down_mnist_dataset.py，将会在当前目录创建 mnist 文件夹并下载二进制数据集。
2. 转换二进制数据集为图片格式：在 prepareData 目录运行 convert_mnist_to_image.py，将会在 prepareData 的同级目录创建 mnistImgs 文件夹存放转换的图片。

# 2. 训练

&emsp;&emsp;在 train 目录在 中运行 train.py 会在当前文件夹生成 LeNet5_MNIST_parameter.pth 和 LeNet5_MNIST_parameter_and_model.pth。

