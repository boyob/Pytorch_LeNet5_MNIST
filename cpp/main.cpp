#include <iostream>
#include <string.h>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torch/script.h>
#include <vector>

using namespace std;
using namespace cv;


int main(int argc, char** argv) {
    //string model_path = "../../torchScript/script/script_model.pt";
    string model_path = "../../torchScript/trace/trace_model.pt";
    string image_path = "../../train/digits/" + (string)argv[1] + ".jpg";
    torch::jit::script::Module module = torch::jit::load(model_path);
    module.to(at::kCUDA);

    Mat img = imread(image_path);
    cvtColor(img, img, CV_BGR2GRAY);
    img.convertTo(img, CV_32F);

    auto options = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);
    torch::Tensor img_tensor = torch::from_blob(img.data, {1, img.rows, img.cols, 1}, options);
    img_tensor = img_tensor.permute({0, 3, 1, 2}).to(torch::kCUDA);

    vector<torch::jit::IValue> inputs;
    inputs.emplace_back(img_tensor.to(at::kCUDA));
    torch::Tensor output = module.forward(inputs).toTensor();
    inputs.pop_back();

    //cout << output << endl << endl;
    tuple<at::Tensor, at::Tensor> result = output.max(1, true);
    int max_index = get<1>(result).item<float>();
    cout<<max_index<<endl;
}
