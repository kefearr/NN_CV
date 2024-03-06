import onnx
import onnxruntime
import torch
import torch.nn as nn
from torchvision import models
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
import time
from torchvision.transforms import transforms
import csv

train_path = './archive/train/train'


def test_onnx(torch_model, samples, batch_size):
    torch_model.eval()
    dummy_input = torch.randn(batch_size, 3, 255, 255)

    export_output = torch.onnx.export(torch_model, dummy_input,"model.onnx", verbose=False)

    ort_session = onnxruntime.InferenceSession("./model.onnx", providers=['CPUExecutionProvider'])



    sum_torch_time = 0
    sum_torch_qua_time = 0
    sum_onnx_time = 0


    torch_model_qua = torch.ao.quantization.quantize_dynamic(
        torch_model,
        {torch.nn.Linear},
        dtype=torch.qint8)

    with torch.no_grad():
        i = 0
        for sample, labels in samples:
            if i > 0:
                break
                
            input_name = ort_session.get_inputs()[0].name
            output_name = ort_session.get_outputs()[0].name
            onnx_input = {input_name: dummy_input.numpy()}

            onnx_start = time.process_time()
            onnxruntime_outputs = ort_session.run([output_name],onnx_input)

            onnx_time = time.process_time() - onnx_start

            _, predicted = torch.max(torch.FloatTensor(onnxruntime_outputs[0]), 1)

            torch_start = time.process_time()
            torch_outputs = torch_model(sample)
            torch_time = time.process_time() - torch_start

            _, predicted = torch.max(torch_outputs.data, 1)


            torch_q_start = time.process_time()
            torch_q_outputs = torch_model_qua(sample)
            torch_q_time = time.process_time() - torch_q_start

            _, predicted = torch.max(torch_q_outputs.data, 1)


            sum_torch_time += torch_time
            sum_onnx_time += onnx_time
            sum_torch_qua_time += torch_q_time
            i += 1

    return sum_torch_time, sum_torch_qua_time, sum_onnx_time



def make_model():
    train_data = ImageFolder('./archive/train/train')
    model = models.efficientnet_b1(weights='IMAGENET1K_V1')
    model.classifier[1] = nn.Linear(1280, len(train_data.classes), bias=True)
    return model


def load_test_data(batch_size):
    transform1 = transforms.Compose([
        transforms.Resize(255),
        transforms.ToTensor()
    ])
    
    test_set= ImageFolder(train_path,transform = transform1)
    return torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

def main():
    model = make_model()
    model.load_state_dict(torch.load('./model/model.pth', map_location=torch.device('cpu')))

    result = []
    for batch_size in [2, 4, 8, 16]:
        test_data = load_test_data(batch_size)
        sum_t, sum_q, sum_o = test_onnx(model, test_data, batch_size)
        result.append({
            "batch_size": batch_size,
            "torch_time": sum_t,
            "torch_q_time": sum_q,
            "onxx_time": sum_o,
        })
    print(result)

    with open("./output/output.csv", "w", newline="") as f:
        print("Saving results to file")
        title = "batch_size,torch_time,torch_q_time,onxx_time".split(",")
        cw = csv.DictWriter(f, title, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        cw.writeheader()
        cw.writerows(result)


if __name__ == '__main__':
    main()
