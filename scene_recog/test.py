import statistics
import time

import onnx
import onnxruntime
import torch


def do_test(model_path, device):
    onnx_model = onnx.load(model_path)
    session_options = onnxruntime.SessionOptions()
    session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

    if device == 'cuda' and not torch.cuda.is_available():
        return "none"

    if device == 'cpu':
        inference_session = onnxruntime.InferenceSession(
            onnx_model.SerializeToString(), session_options, providers=['CPUExecutionProvider']
        )
    elif device == 'cuda':
        session_options.intra_op_num_threads = 1
        session_options.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL
        inference_session = onnxruntime.InferenceSession(
            onnx_model.SerializeToString(), session_options, providers=['CUDAExecutionProvider']
        )
    else:
        raise ValueError(f"Unsupported device type: {device}")

    samples_count = 100
    input_data = torch.randn(1, 3, 224, 224).to(device)
    input_data = {'input': input_data.cpu().numpy()}

    results = []
    for i in range(samples_count):
        start_time = time.time()
        inference_session.run([], input_data)
        results.append(time.time() - start_time)

    return results


if __name__ == '__main__':
    models = ['18_resnet.onnx', '152_resnet.onnx',
              'no_18_resnet.onnx',]
    devices = ['cpu', 'cuda']

    for model in models:
        for device in devices:
            result = do_test(model, device)
            if (result == 'none'):
                print(model, device)
                print("Cuda device not detected")
                print()
            else:
                print(model, device)
                print(f'Avg Time: {statistics.mean(result)}')
                print()