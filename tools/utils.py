import cv2  # Import OpenCV để xử lý hình ảnh
import os  # Import os để làm việc với hệ thống file

import numpy as np  # Import numpy để làm việc với mảng số học

from tools.adjustBrightness import adjust_brightness_from_src_to_dst, read_img  # Import các hàm tùy chỉnh để điều chỉnh độ sáng


def load_test_data(image_path):
    # Hàm để tải và tiền xử lý dữ liệu ảnh để sử dụng cho mô hình
    img = cv2.imread(image_path).astype(np.float32)  # Đọc ảnh từ đường dẫn và chuyển sang kiểu float32
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Chuyển đổi ảnh từ không gian màu BGR sang RGB
    img = preprocessing(img)  # Tiền xử lý ảnh
    img = np.transpose(img, (2, 0, 1))  # Chuyển đổi ảnh từ (H, W, C) sang (C, H, W)
    img = np.expand_dims(img, axis=0)  # Thêm một chiều vào tensor để tạo batch (1, C, H, W)
    return img  # Trả về ảnh đã được tiền xử lý


def preprocessing(img):
    # Hàm tiền xử lý đơn giản, chuẩn hóa giá trị pixel từ [0, 255] về [-1, 1]
    # (Một số đoạn code khác đã bị bình luận)
    # H, W là kích thước mong muốn của ảnh đầu vào
    return img / 127.5 - 1.0  # Chuẩn hóa ảnh


def save_images(images, image_path, photo_path=None):
    # Hàm lưu trữ ảnh sau khi xử lý
    fake = inverse_transform(images.squeeze())  # Biến đổi ngược ảnh từ [-1, 1] về [0, 255]
    if photo_path:
        # Nếu có đường dẫn tới ảnh thật, điều chỉnh độ sáng trước khi lưu
        return imsave(adjust_brightness_from_src_to_dst(fake, read_img(photo_path)), image_path)
    else:
        # Nếu không, chỉ cần lưu ảnh
        return imsave(fake, image_path)


def inverse_transform(images):
    # Hàm biến đổi ngược ảnh từ [-1, 1] về [0, 255]
    images = (images + 1.) / 2 * 255
    # Do tính toán số thực không chính xác, cần giới hạn giá trị pixel trong khoảng [0, 255]
    # Nếu không, ảnh có thể bị biến dạng hoặc xuất hiện artifact khi hiển thị
    images = np.clip(images, 0, 255)
    return images.astype(np.uint8)  # Trả về ảnh dưới dạng uint8


def imsave(images, path):
    # Hàm lưu ảnh, chuyển đổi từ RGB về BGR trước khi lưu để phù hợp với OpenCV
    return cv2.imwrite(path, cv2.cvtColor(images, cv2.COLOR_BGR2RGB))


crop_image = lambda img, x0, y0, w, h: img[y0:y0 + h, x0:x0 + w]  # Hàm lambda để cắt ảnh theo tọa độ x0, y0 và kích thước w, h


def random_crop(img1, img2, crop_H, crop_W):
    # Hàm cắt ngẫu nhiên hai ảnh cùng vị trí và kích thước
    assert img1.shape == img2.shape  # Đảm bảo rằng hai ảnh có cùng kích thước
    h, w = img1.shape[:2]

    # Chiều rộng cắt không thể vượt quá chiều rộng của ảnh gốc
    if crop_W > w:
        crop_W = w

    # Chiều cao cắt
    if crop_H > h:
        crop_H = h

    # Tạo ngẫu nhiên vị trí góc trên bên trái để cắt ảnh
    x0 = np.random.randint(0, w - crop_W + 1)
    y0 = np.random.randint(0, h - crop_H + 1)

    crop_1 = crop_image(img1, x0, y0, crop_W, crop_H)  # Cắt ảnh đầu tiên
    crop_2 = crop_image(img2, x0, y0, crop_W, crop_H)  # Cắt ảnh thứ hai
    return crop_1, crop_2  # Trả về hai ảnh đã được cắt


def check_folder(log_dir):
    # Hàm kiểm tra và tạo thư mục nếu chưa tồn tại
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir  # Trả về đường dẫn thư mục


def str2bool(x):
    # Hàm chuyển đổi chuỗi thành giá trị boolean
    return x.lower() in ('true')
