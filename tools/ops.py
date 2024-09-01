import torch  # Import thư viện PyTorch
from torch import nn  # Import các lớp mạng thần kinh từ PyTorch


def generator_loss(fake):
    # Tính toán mất mát (loss) cho Generator
    # Mục tiêu của Generator là tạo ra ảnh mà Discriminator sẽ đánh giá là "thật" (gần 1.0)
    fake_loss = torch.mean(torch.square(fake - 1.0))
    return fake_loss


def discriminator_loss(real, gray, fake, real_blur):
    # Tính toán mất mát (loss) cho Discriminator
    # Discriminator cố gắng phân biệt giữa ảnh thật (real), ảnh xám (gray), ảnh giả (fake), và ảnh làm mờ (real_blur)

    real_loss = torch.mean(torch.square(real - 1.0))  # Mất mát cho ảnh thật
    gray_loss = torch.mean(torch.square(gray))  # Mất mát cho ảnh xám
    fake_loss = torch.mean(torch.square(fake))  # Mất mát cho ảnh giả
    real_blur_loss = torch.mean(torch.square(real_blur))  # Mất mát cho ảnh làm mờ

    # Các hệ số trong chú thích được sử dụng cho các phong cách anime khác nhau:
    # Hayao : 1.2, 1.2, 1.2, 0.8
    # Paprika : 1.0, 1.0, 1.0, 0.005
    # Shinkai: 1.7, 1.7, 1.7, 1.0
    return real_loss, fake_loss, gray_loss, real_blur_loss


def gram_matrix(input):
    # Tính toán ma trận Gram từ các đặc trưng của ảnh để đo độ tương đồng giữa các kiểu ảnh
    b, c, h, w = input.size()  # Lấy kích thước batch, kênh, chiều cao, chiều rộng của tensor
    reshape_input = input.view(b * c, h * w)  # Reshape tensor thành dạng 2 chiều
    G = torch.mm(reshape_input, reshape_input.t())  # Tính ma trận Gram
    return G.div(b * c * h * w)  # Chia cho tổng số phần tử để chuẩn hóa


def con_loss(pre_train_model: nn.Module, real, fake):
    # Tính toán content loss giữa ảnh thật và ảnh giả dựa trên các đặc trưng được trích xuất từ mô hình VGG đã huấn luyện trước
    real_feature_map = pre_train_model(real)  # Trích xuất đặc trưng từ ảnh thật
    fake_feature_map = pre_train_model(fake)  # Trích xuất đặc trưng từ ảnh giả
    loss = nn.L1Loss()(real_feature_map, fake_feature_map)  # Tính L1 loss giữa hai đặc trưng
    return loss


def style_loss(style, fake):
    # Tính toán style loss giữa ảnh phong cách và ảnh giả dựa trên ma trận Gram của chúng
    return nn.L1Loss()(gram_matrix(style), gram_matrix(fake))


def con_sty_loss(pre_train_model: nn.Module, real, anime, fake):
    # Tính toán cả content loss và style loss
    real_feature_map = pre_train_model(real)  # Trích xuất đặc trưng từ ảnh thật
    fake_feature_map = pre_train_model(fake)  # Trích xuất đặc trưng từ ảnh giả
    anime_feature_map = pre_train_model(anime)  # Trích xuất đặc trưng từ ảnh phong cách anime

    c_loss = nn.L1Loss()(real_feature_map, fake_feature_map)  # Tính content loss giữa ảnh thật và ảnh giả
    s_loss = style_loss(anime_feature_map, fake_feature_map)  # Tính style loss giữa ảnh anime và ảnh giả

    return c_loss, s_loss  # Trả về content loss và style loss


def color_loss(real, fake):
    # Tính toán color loss giữa ảnh thật và ảnh giả trong không gian màu YUV
    real_yuv = rgb2yuv(real)  # Chuyển đổi ảnh thật từ RGB sang YUV
    fake_yuv = rgb2yuv(fake)  # Chuyển đổi ảnh giả từ RGB sang YUV
    loss = nn.L1Loss()(real_yuv[:, :, :, 0], fake_yuv[:, :, :, 0]) + \
           nn.SmoothL1Loss()(real_yuv[:, :, :, 1], fake_yuv[:, :, :, 1]) + \
           nn.SmoothL1Loss()(real_yuv[:, :, :, 2], fake_yuv[:, :, :, 2])  # Tính L1 và Smooth L1 loss cho từng kênh YUV
    return loss


def total_variation_loss(inputs):
    # Tính toán total variation loss để giảm độ nhiễu trong ảnh giả
    dh = inputs[:, :-1, ...] - inputs[:, 1:, ...]  # Sự khác biệt theo chiều dọc
    dw = inputs[:, :, :-1, ...] - inputs[:, :, 1:, ...]  # Sự khác biệt theo chiều ngang
    return torch.mean(torch.abs(dh)) + torch.mean(torch.abs(dw))  # Tổng hợp total variation loss


def rgb2yuv(rgb):
    # Chuyển đổi ảnh từ không gian màu RGB sang YUV
    rgb_ = (rgb + 1.0) / 2.0  # Chuẩn hóa RGB từ [-1, 1] về [0, 1]
    # Ma trận chuyển đổi từ Wikipedia
    A = torch.tensor([[0.299, -0.14714119, 0.61497538],
                      [0.587, -0.28886916, -0.51496512],
                      [0.114, 0.43601035, -0.10001026]])
    A = A.type_as(rgb)  # Chuyển ma trận A sang cùng kiểu dữ liệu với rgb
    yuv = torch.tensordot(rgb_, A, dims=([rgb.ndim - 3], [0]))  # Tính tích vô hướng giữa rgb và ma trận A để chuyển đổi
    return yuv  # Trả về ảnh trong không gian màu YUV
