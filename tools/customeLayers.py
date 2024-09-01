import torch.nn as nn  # Import các lớp mạng thần kinh từ PyTorch
from torch.nn.utils import spectral_norm  # Import spectral normalization để chuẩn hóa các trọng số của lớp Conv2d


# Lớp ConvNormLReLU là một lớp kết hợp giữa các lớp convolution, normalization, và activation (LeakyReLU)
class ConvNormLReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, pad_mode="reflect", groups=1, bias=False):
        super(ConvNormLReLU, self).__init__()

        # Chọn lớp padding dựa trên chế độ pad_mode
        pad_layer = {
            "zero": nn.ZeroPad2d,  # Zero padding
            "same": nn.ReplicationPad2d,  # Replication padding
            "reflect": nn.ReflectionPad2d,  # Reflection padding
        }
        if pad_mode not in pad_layer:
            raise NotImplementedError  # Nếu pad_mode không hợp lệ, thông báo lỗi

        # Khởi tạo các lớp thành phần
        self.pad = pad_layer[pad_mode](padding)  # Lớp padding
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=0, groups=groups, bias=bias)
        # Lớp convolution với các tham số được truyền vào
        self.norm = nn.InstanceNorm2d(out_ch, affine=True)  # Lớp normalization (InstanceNorm2d)
        self.relu = nn.LeakyReLU(0.2, inplace=True)  # Lớp activation (LeakyReLU)

    def forward(self, input):
        # Định nghĩa quá trình forward qua lớp này
        out = self.pad(input)  # Áp dụng padding
        out = self.conv(out)  # Áp dụng convolution
        out = self.norm(out)  # Áp dụng normalization
        out = self.relu(out)  # Áp dụng activation (LeakyReLU)
        return out  # Trả về kết quả đầu ra


# Lớp InvertedResBlock là một khối residual với cấu trúc inverted bottleneck
class InvertedResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, expansion_ratio=2):
        super(InvertedResBlock, self).__init__()

        self.use_res_connect = in_ch == out_ch  # Xác định xem có sử dụng kết nối residual hay không (nếu đầu vào và đầu ra có cùng kích thước)
        bottleneck = int(round(in_ch * expansion_ratio))  # Tính toán kích thước bottleneck
        layers = []  # Tạo danh sách các lớp

        if expansion_ratio != 1:
            layers.append(ConvNormLReLU(in_ch, bottleneck, kernel_size=1, padding=0))  # Lớp ConvNormLReLU với kernel 1x1 để mở rộng kênh

        # Depthwise convolution (dw)
        layers.append(ConvNormLReLU(bottleneck, bottleneck, groups=bottleneck, bias=True))
        # Pointwise convolution (pw)
        layers.append(nn.Conv2d(bottleneck, out_ch, kernel_size=1, padding=0, bias=False))  # Convolution với kernel 1x1 để giảm số kênh
        layers.append(nn.InstanceNorm2d(out_ch, affine=True))  # Lớp normalization

        self.layers = nn.Sequential(*layers)  # Gộp các lớp vào nn.Sequential

    def forward(self, input):
        # Định nghĩa quá trình forward qua khối này
        out = self.layers(input)  # Áp dụng các lớp đã được định nghĩa
        if self.use_res_connect:
            out = input + out  # Nếu sử dụng kết nối residual, cộng đầu vào với đầu ra
        return out  # Trả về kết quả đầu ra


# Lớp ConvSN là một lớp convolution với tùy chọn spectral normalization
class ConvSN(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=4, padding=0, stride=2, pad_mode='zero', use_bias=True, sn=False):
        super(ConvSN, self).__init__()

        # Chọn lớp padding dựa trên chế độ pad_mode
        pad_layer = {
            "zero": nn.ZeroPad2d,  # Zero padding
            "same": nn.ReplicationPad2d,  # Replication padding
            "reflect": nn.ReflectionPad2d,  # Reflection padding
        }
        if pad_mode not in pad_layer:
            raise NotImplementedError  # Nếu pad_mode không hợp lệ, thông báo lỗi

        self.pad_layer = pad_layer[pad_mode](padding)  # Lớp padding

        # Tạo lớp convolution với hoặc không có spectral normalization
        if sn:
            self.conv = spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=0, bias=use_bias))
        else:
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=0, bias=use_bias)

    def forward(self, input):
        # Định nghĩa quá trình forward qua lớp này
        out = self.pad_layer(input)  # Áp dụng padding
        out = self.conv(out)  # Áp dụng convolution
        return out  # Trả về kết quả đầu ra
