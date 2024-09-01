import torch
import argparse
from tools.utils import *  # Import các hàm tiện ích từ file utils
import os
from tqdm import tqdm  # Import tqdm để hiển thị tiến trình (progress bar)
from glob import glob  # Import glob để tìm kiếm các file theo pattern
import time
import numpy as np  # Import numpy để xử lý dữ liệu mảng
from net.generator import Generator  # Import lớp Generator từ file generator

# Kiểm tra nếu có GPU thì sử dụng, nếu không thì sử dụng CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hàm parse_args để nhận các tham số từ dòng lệnh
def parse_args():
    desc = "AnimeGANv2"  # Mô tả ngắn về chương trình
    parser = argparse.ArgumentParser(description=desc)  # Tạo đối tượng ArgumentParser để nhận tham số từ dòng lệnh

    # Thêm tham số `--model_dir` để chỉ định đường dẫn tới mô hình đã lưu, mặc định là 'save_model/generated_Hayao.pth'
    parser.add_argument('--model_dir', type=str, default='save_model/' + 'generated_Hayao.pth',
                        help='Directory name to save the checkpoints')

    # Thêm tham số `--if_adjust_brightness` để bật/tắt tùy chọn điều chỉnh độ sáng của ảnh dựa trên ảnh thật
    parser.add_argument('--if_adjust_brightness', action='store_true',
                        help='adjust brightness by the real photo')

    # Thêm tham số `--test_file_path` để chỉ định đường dẫn file ảnh cần kiểm tra
    parser.add_argument('--test_file_path', type=str, default=None,
                        help='test file path')

    """checking arguments"""

    return parser.parse_args()  # Trả về các tham số đã phân tích

# Hàm load_model để tải mô hình từ checkpoint
def load_model(model_dir):
    """
    load model from checkpoint
    Args:
        model_dir: checkpoint directory

    Returns: model

    """
    ckpt = torch.load(model_dir, map_location=device)  # Tải checkpoint từ đường dẫn đã cung cấp
    generated = Generator()  # Khởi tạo đối tượng Generator từ mô hình đã lưu
    # Lọc các trọng số liên quan tới generator từ checkpoint
    generatordict = dict(filter(lambda k: 'generated' in k[0], ckpt['state_dict'].items()))
    # Loại bỏ tiền tố 'generated.' từ các tên trọng số
    generatordict = {k.split('.', 1)[1]: v for k, v in generatordict.items()}
    generated.load_state_dict(generatordict, True)  # Tải trọng số vào mô hình generator
    generated.eval()  # Chuyển mô hình sang chế độ đánh giá (evaluation mode)
    del generatordict  # Giải phóng bộ nhớ
    del ckpt  # Giải phóng bộ nhớ
    return generated  # Trả về mô hình đã tải

# Hàm test để thực hiện kiểm tra mô hình trên một ảnh đầu vào
def test(model_dir, test_file_path, if_adjust_brightness):
    # tf.reset_default_graph()
    result_dir = 'results'  # Thư mục để lưu kết quả ảnh sau khi xử lý
    check_folder(result_dir)  # Kiểm tra và tạo thư mục nếu chưa tồn tại

    generated = load_model(model_dir)  # Tải mô hình từ checkpoint

    sample_image = np.asarray(load_test_data(test_file_path))  # Đọc ảnh đầu vào và chuyển thành mảng numpy
    sample_image = torch.Tensor(sample_image)  # Chuyển ảnh thành đối tượng Tensor của PyTorch
    image_path = os.path.join(result_dir, '{0}'.format(os.path.basename(test_file_path)))  # Đường dẫn lưu ảnh kết quả
    fake_img = generated(sample_image).detach().numpy()  # Chạy mô hình để tạo ảnh giả (phong cách anime)
    fake_img = np.squeeze(fake_img, axis=0)  # Loại bỏ chiều không cần thiết
    fake_img = np.transpose(fake_img, (1, 2, 0))  # Chuyển đổi trục để ảnh có định dạng [H, W, C]

    # Nếu tùy chọn điều chỉnh độ sáng được bật, lưu ảnh với điều chỉnh độ sáng
    if if_adjust_brightness:
        save_images(fake_img, image_path, test_file_path)
    else:  # Nếu không, lưu ảnh mà không điều chỉnh độ sáng
        save_images(fake_img, image_path, None)

    print('Saved image: ' + image_path)  # In ra thông báo đã lưu ảnh

# Kiểm tra nếu chương trình được chạy từ dòng lệnh
if __name__ == '__main__':
    arg = parse_args()  # Nhận các tham số từ dòng lệnh
    print(arg.model_dir)  # In ra đường dẫn mô hình để kiểm tra
    test(arg.model_dir, arg.test_file_path, arg.if_adjust_brightness)  # Gọi hàm test để xử lý ảnh đầu vào
