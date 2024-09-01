import argparse  # Import argparse để phân tích các tham số từ dòng lệnh

import coremltools as ct  # Import coremltools để làm việc với mô hình CoreML
import torch  # Import torch để làm việc với PyTorch
from pytorch_lightning import LightningModule  # Import LightningModule từ pytorch_lightning để sử dụng trong mô hình
from torch.utils.mobile_optimizer import optimize_for_mobile  # Import hàm optimize_for_mobile để tối ưu hóa mô hình cho Torch Mobile

from AnimeGANv2 import AnimeGANv2  # Import lớp AnimeGANv2 từ file AnimeGANv2

# Hàm export_to_onnx để xuất mô hình sang định dạng ONNX
def export_to_onnx(model: LightningModule, input_sample):
    """
    Export the model to ONNX format
    Args:
        model: The model to be exported
        input_sample: The input sample to the model

    Returns:
        None
    """
    model.to_onnx('save_model/onnx/animeGan.onnx', input_sample=input_sample,
                  input_names=['input'], output_names=['output'])
    # Xuất mô hình sang ONNX và lưu tại 'save_model/onnx/animeGan.onnx'

# Hàm export_to_onnx_with_dynamic_input để xuất mô hình ONNX với đầu vào động
def export_to_onnx_with_dynamic_input(model: LightningModule, input_sample):
    """
    Export the model to ONNX format with dynamic input
    Args:
        model: The model to be exported

    Returns:
        None
    """
    # Định nghĩa các trục động cho đầu vào (NCHW)
    dynamic_axes = {
        'input': {2: "height", 3: 'width'},
    }
    # Xuất mô hình sang ONNX với đầu vào động và lưu tại 'save_model/onnx/animeGan_dynamic.onnx'
    model.to_onnx('save_model/onnx/animeGan_dynamic.onnx', input_sample=input_sample, dynamic_axes=dynamic_axes,
                  input_names=['input'], output_names=['output'])

# Hàm export_to_pytorch_model để xuất mô hình sang định dạng PyTorch
def export_to_pytorch_model(model: AnimeGANv2):
    """
    Export the model to PyTorch model format
    Args:
        model: The model to be exported

    Returns:
        None
    """
    torch.save(model.generated, 'save_model/pytorch/animeGan.pth')
    # Lưu mô hình PyTorch tại 'save_model/pytorch/animeGan.pth'

# Hàm export_to_torchscript_model để xuất mô hình sang định dạng TorchScript
def export_to_torchscript_model(model: AnimeGANv2):
    """
    Export the model to TorchScript model format
    Args:
        model: The model to be exported

    Returns:
        None
    """
    traced_script_module = torch.jit.trace(model.generated, example_inputs=torch.randn(1, 3, 256, 256))
    # Truy vết mô hình bằng TorchScript
    traced_script_module.save('save_model/torchscript/animeGan.pt')
    # Lưu mô hình TorchScript tại 'save_model/torchscript/animeGan.pt'

# Hàm export_to_coreml_model để xuất mô hình sang định dạng CoreML
def export_to_coreml_model(model: AnimeGANv2):
    """
    Export the model to CoreML model format
    Args:
        model: The model to be exported

    Returns:
        None
    """
    model = model.generated
    model.eval()  # Chuyển mô hình sang chế độ đánh giá
    trace_model = torch.jit.trace(model, example_inputs=torch.randn(1, 3, 256, 256))
    # Truy vết mô hình để chuẩn bị xuất sang CoreML

    # Định nghĩa scale và bias để chuẩn bị cho quá trình chuẩn hóa dữ liệu đầu vào
    scale = 1 / (0.226 * 255.0)
    bias = [- 0.485 / (0.229), - 0.456 / (0.224), - 0.406 / (0.225)]
    # Định nghĩa hình dạng đầu vào của mô hình
    input_shape = ct.Shape(shape=(1, 3,
                                  ct.RangeDim(lower_bound=0, upper_bound=-1, default=256),
                                  ct.RangeDim(lower_bound=0, upper_bound=-1, default=256)))
    image_input = ct.ImageType(name="input",
                               shape=input_shape,
                               scale=scale, bias=bias)

    # Chuyển mô hình sang định dạng CoreML
    mlmodel = ct.convert(trace_model, inputs=[image_input],
                         outputs=[ct.TensorType(name='output')],
                         debug=True)
    mlmodel.save('save_model/coreml/animeGan.mlmodel')
    # Lưu mô hình CoreML tại 'save_model/coreml/animeGan.mlmodel'

# Hàm export_to_torch_mobile_model để xuất mô hình sang định dạng Torch Mobile
def export_to_torch_mobile_model(model: AnimeGANv2):
    """
    Export the model to Torch Mobile model format
    Args:
        model: The model to be exported

    Returns:
        None
    """
    model = model.generated
    model.eval()  # Chuyển mô hình sang chế độ đánh giá
    example_inputs = torch.randn(1, 3, 256, 256)
    traced_script_module = torch.jit.trace(model, example_inputs=example_inputs)
    # Truy vết mô hình để chuẩn bị xuất sang Torch Mobile
    optimized_model = optimize_for_mobile(traced_script_module, backend='metal')
    # Tối ưu hóa mô hình cho Torch Mobile với backend là 'metal'
    print(torch.jit.export_opnames(optimized_model))
    optimized_model._save_for_lite_interpreter('save_model/torch_mobile/animeGan_metal.pt')
    # Lưu mô hình Torch Mobile tại 'save_model/torch_mobile/animeGan_metal.pt'

# Hàm pase_args để nhận các tham số từ dòng lệnh
def pase_args():
    """
    Parse the arguments
    Returns:
        The parsed arguments
    """
    desc = "Export the model to ONNX or PyTorch model format"  # Mô tả ngắn về chương trình
    parser = argparse.ArgumentParser(description=desc)  # Tạo đối tượng ArgumentParser để nhận tham số từ dòng lệnh
    parser.add_argument('--checkpoint_path', type=str, required=True, help='The path to the checkpoint')
    # Thêm tham số `--checkpoint_path` để chỉ định đường dẫn tới checkpoint, bắt buộc phải có
    parser.add_argument('--onnx', action='store_true', help='Export to ONNX format')
    # Thêm tùy chọn `--onnx` để xuất mô hình sang định dạng ONNX
    parser.add_argument('--pytorch', action='store_true', help='Export to PyTorch model format')
    # Thêm tùy chọn `--pytorch` để xuất mô hình sang định dạng PyTorch
    parser.add_argument('--dynamic', action='store_true', help='Export to ONNX format with dynamic input')
    # Thêm tùy chọn `--dynamic` để xuất mô hình ONNX với đầu vào động
    parser.add_argument('--torchscript', action='store_true', help='Export to TorchScript model format')
    # Thêm tùy chọn `--torchscript` để xuất mô hình sang định dạng TorchScript
    parser.add_argument('--coreml', action='store_true', help='Export to CoreML model format')
    # Thêm tùy chọn `--coreml` để xuất mô hình sang định dạng CoreML
    parser.add_argument('--torch_mobile', action='store_true', help='Export to Torch Mobile model format')
    # Thêm tùy chọn `--torch_mobile` để xuất mô hình sang định dạng Torch Mobile
    args = parser.parse_args()  # Phân tích các tham số từ dòng lệnh
    return args  # Trả về các tham số đã phân tích

# Kiểm tra nếu chương trình được chạy từ dòng lệnh
if __name__ == '__main__':
    args = pase_args()  # Nhận các tham số từ dòng lệnh
    model = AnimeGANv2.load_from_checkpoint(args.checkpoint_path, strict=False)
    # Tải mô hình từ checkpoint đã cung cấp
    input_sample = torch.randn(1, 3, 256, 256)  # Tạo mẫu đầu vào để xuất mô hình
    if args.onnx:
        export_to_onnx(model, input_sample)
        print('Export to ONNX format successfully')  # In thông báo nếu xuất sang ONNX thành công
    if args.pytorch:
        export_to_pytorch_model(model)
        print('Export to PyTorch model format successfully')  # In thông báo nếu xuất sang PyTorch thành công
    if args.dynamic:
        export_to_onnx_with_dynamic_input(model, input_sample)
        print('Export to ONNX format with dynamic input successfully')  # In thông báo nếu xuất ONNX động thành công
    if args.torchscript:
        export_to_torchscript_model(model)
        print('Export to TorchScript model format successfully')  # In thông báo nếu xuất sang TorchScript thành công
    if args.coreml:
        export_to_coreml_model(model)
        print('Export to CoreML model format successfully')  # In thông báo nếu xuất sang CoreML thành công
    if args.torch_mobile:
        export_to_torch_mobile_model(model)
        print('Export to Torch Mobile model format successfully')  # In thông báo nếu xuất sang Torch Mobile thành công
