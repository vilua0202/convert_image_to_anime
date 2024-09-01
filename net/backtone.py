import os
from collections import OrderedDict

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn


# Định nghĩa một lớp có tên là VGGCaffePreTrained, lớp này kế thừa từ pl.LightningModule của PyTorch Lightning.
class VGGCaffePreTrained(pl.LightningModule):
    # Cấu hình các lớp của mạng VGG, với 'M' đại diện cho lớp Max Pooling và các số khác đại diện cho số lượng filter trong các lớp Convolution.
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256,
           'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']

    # Hàm khởi tạo của lớp. weights_path là đường dẫn tới file chứa trọng số của mô hình đã được huấn luyện trước, output_index là vị trí lớp mà ta muốn lấy kết quả đầu ra.
    def __init__(self, weights_path: str = os.path.dirname(os.path.abspath(__file__)) + '/models/vgg19.npy',
                 output_index: int = 26) -> None:
        super().__init__()
        try:
            # Tải dữ liệu trọng số từ file npy. Tạo các layer của mô hình sử dụng make_layers và truyền vào data_dict chứa các trọng số.
            data_dict: dict = np.load(weights_path, encoding='latin1', allow_pickle=True).item()
            self.features = self.make_layers(self.cfg, data_dict)
            del data_dict  # Xóa data_dict khỏi bộ nhớ sau khi không cần dùng nữa.
        except FileNotFoundError as e:
            # Nếu không tìm thấy file trọng số, hiển thị thông báo lỗi.
            print("weights_path:", weights_path,
                  'does not exits!, if you want to training must download pretrained weights')
        self.output_index = output_index  # Lưu trữ vị trí lớp đầu ra.
        self.vgg_normalize = None  # Sẽ được thiết lập trong hàm setup.

    # Hàm xử lý ảnh trước khi đưa vào mô hình, bao gồm việc chuyển đổi từ RGB sang BGR và chuẩn hóa theo chuẩn VGG.
    def _process(self, x):
        # NOTE: Phạm vi giá trị ảnh đầu vào [-1~1], ta cần chuyển về 0-1 trước khi chuẩn hóa.
        rgb = (x * 0.5 + 0.5) * 255  # Chuyển đổi về khoảng 0-255.
        bgr = rgb[:, [2, 1, 0], :, :]  # Chuyển từ RGB sang BGR.
        return self.vgg_normalize(bgr)  # Chuẩn hóa theo chuẩn VGG.

    # Hàm setup được gọi khi mô hình được chuyển lên một thiết bị (ví dụ: GPU), để thiết lập mean cho việc chuẩn hóa.
    def setup(self, device: torch.device):
        mean: torch.Tensor = torch.tensor([103.939, 116.779, 123.68], device=device)
        mean = mean[None, :, None, None]  # Thêm các chiều cho mean để phù hợp với kích thước của input.
        self.vgg_normalize = lambda x: x - mean  # Đặt hàm chuẩn hóa VGG.
        self.freeze()  # Đóng băng mô hình (tức là không cho phép cập nhật trọng số).

    # Hàm thực hiện việc forward qua mô hình mà không áp dụng hàm kích hoạt ReLU cuối cùng.
    def _forward_impl(self, x):
        x = self._process(x)  # Xử lý ảnh trước.
        # NOTE: Lấy đầu ra mà không áp dụng hàm kích hoạt ReLU.
        x = self.features[:self.output_index](x)  # Chạy qua các layer của mô hình đến lớp có chỉ số output_index.
        return x

    # Hàm forward, gọi _forward_impl để thực hiện forward.
    def forward(self, x):
        return self._forward_impl(x)

    # Ghi đè hàm train để đảm bảo mô hình luôn ở chế độ đánh giá (evaluation mode).
    def train(self, mode: bool):
        return super().train(False)

    # Ghi đè hàm state_dict để không lưu trọng số của mô hình.
    def state_dict(self, destination, prefix, keep_vars):
        destination = OrderedDict()
        destination._metadata = OrderedDict()
        return destination

    # Các hàm tiện ích để lấy các trọng số từ data_dict.
    @staticmethod
    def get_conv_filter(data_dict, name):
        return data_dict[name][0]

    @staticmethod
    def get_bias(data_dict, name):
        return data_dict[name][1]

    @staticmethod
    def get_fc_weight(data_dict, name):
        return data_dict[name][0]

    # Hàm tạo các layer của mô hình từ cấu hình cfg, sử dụng trọng số từ data_dict.
    def make_layers(self, cfg, data_dict, batch_norm=False) -> nn.Sequential:
        layers = []
        in_channels = 3  # Số kênh đầu vào, RGB nên là 3.
        block = 1  # Số thứ tự của block convolution.
        number = 1  # Số thứ tự của lớp trong mỗi block.
        for v in cfg:
            if v == 'M':  # Nếu gặp 'M' trong cấu hình, thêm một lớp MaxPooling.
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                block += 1  # Tăng số block.
                number = 1  # Reset số thứ tự của lớp trong block.
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)  # Tạo một lớp Conv2D.
                with torch.no_grad():
                    """ Đặt giá trị trọng số """
                    weight = torch.FloatTensor(self.get_conv_filter(data_dict, f'conv{block}_{number}'))
                    weight = weight.permute((3, 2, 0, 1))  # Chuyển đổi thứ tự chiều của trọng số từ HWC sang CHW.
                    bias = torch.FloatTensor(self.get_bias(data_dict, f'conv{block}_{number}'))
                    conv2d.weight.set_(weight)  # Đặt trọng số cho lớp convolution.
                    conv2d.bias.set_(bias)  # Đặt bias cho lớp convolution.
                number += 1  # Tăng số thứ tự lớp trong block.
                if batch_norm:  # Nếu có batch_norm, thêm batch normalization và ReLU vào sau Conv2D.
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:  # Nếu không có batch_norm, chỉ thêm ReLU vào sau Conv2D.
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v  # Cập nhật số kênh đầu vào cho lớp tiếp theo.

        return nn.Sequential(*layers)  # Trả về một chuỗi các layer dưới dạng nn.Sequential.
