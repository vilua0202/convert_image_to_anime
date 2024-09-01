import os  # Import os để làm việc với các tác vụ liên quan đến hệ thống file

import cv2  # Import OpenCV để xử lý hình ảnh
import numpy as np  # Import numpy để làm việc với mảng số học
import torch  # Import torch để sử dụng với PyTorch
from torch.utils.data import Dataset  # Import lớp Dataset để tạo dataset tùy chỉnh


def _transform(image):
    # Hàm này để chuẩn hóa ảnh bằng cách chuyển các giá trị pixel từ [0, 255] về [-1, 1]
    processing_image = image / 127.5 - 1.0
    return processing_image


class AnimeDataSet(Dataset):
    def __init__(self, dataset, data_dir):
        """
        Khởi tạo lớp AnimeDataSet
        Cấu trúc thư mục:
            - {data_dir}
                - train_photo
                    1.jpg, ..., n.jpg
                - {dataset}  # Ví dụ: Hayao
                    smooth
                        1.jpg, ..., n.jpg
                    style
                        1.jpg, ..., n.jpg
        """

        anime_dir = os.path.join(data_dir, dataset)  # Đường dẫn tới thư mục chứa ảnh anime
        if not os.path.exists(data_dir):  # Kiểm tra xem thư mục data_dir có tồn tại không
            raise FileNotFoundError(f'Folder {data_dir} does not exist')

        if not os.path.exists(anime_dir):  # Kiểm tra xem thư mục anime_dir có tồn tại không
            raise FileNotFoundError(f'Folder {anime_dir} does not exist')

        self.data_dir = data_dir
        self.image_files = {}  # Tạo một từ điển để lưu đường dẫn tới các file ảnh
        self.photo = 'train_photo'  # Thư mục chứa ảnh thật
        self.style = f'{anime_dir}/style'  # Thư mục chứa ảnh phong cách anime
        self.smooth = f'{anime_dir}/smooth'  # Thư mục chứa ảnh anime làm mịn
        self.dummy = torch.zeros(3, 256, 256)  # Tạo một tensor dummy để sử dụng khi cần

        for opt in [self.photo, self.style, self.smooth]:
            if 'photo' in opt:
                folder = os.path.join(data_dir, opt)  # Nếu là photo thì ghép thêm với data_dir
            else:
                folder = opt  # Nếu không thì giữ nguyên
            files = os.listdir(folder)  # Lấy danh sách các file trong thư mục

            self.image_files[opt] = [os.path.join(folder, fi) for fi in files]  # Lưu đường dẫn đầy đủ tới các file

        print(f'Dataset: real {len(self.image_files[self.photo])} style {self.len_anime}, smooth {self.len_smooth}')
        # In ra số lượng ảnh trong từng thư mục

    def __len__(self):
        return len(self.image_files[self.photo])  # Trả về số lượng ảnh thật

    @property
    def len_anime(self):
        return len(self.image_files[self.style])  # Trả về số lượng ảnh anime

    @property
    def len_smooth(self):
        return len(self.image_files[self.smooth])  # Trả về số lượng ảnh anime làm mịn

    def __getitem__(self, index):
        image = self.load_photo(index)  # Tải ảnh thật tại chỉ số index
        anm_idx = index  # Khởi tạo chỉ số cho ảnh anime
        if anm_idx > self.len_anime - 1:
            anm_idx -= self.len_anime * (index // self.len_anime)
            # Điều chỉnh chỉ số nếu nó vượt quá số lượng ảnh anime

        anime, anime_gray = self.load_anime(anm_idx)  # Tải ảnh anime và ảnh xám của nó
        smooth_gray = self.load_anime_smooth(anm_idx)  # Tải ảnh anime làm mịn xám

        return image, anime, anime_gray, smooth_gray  # Trả về một bộ gồm ảnh thật, ảnh anime, ảnh anime xám, và ảnh làm mịn xám

    def load_photo(self, index):
        fpath = self.image_files[self.photo][index]  # Lấy đường dẫn tới ảnh thật tại chỉ số index
        image = cv2.imread(fpath).astype(np.float32)  # Đọc ảnh từ file và chuyển sang kiểu float32
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Chuyển ảnh từ không gian màu BGR sang RGB

        image = _transform(image)  # Chuẩn hóa ảnh
        image = np.transpose(image, (2, 0, 1))  # Chuyển đổi ảnh từ (H, W, C) sang (C, H, W)
        return torch.tensor(image)  # Trả về ảnh dưới dạng tensor của PyTorch

    def load_anime(self, index):
        fpath = self.image_files[self.style][index]  # Lấy đường dẫn tới ảnh anime tại chỉ số index
        # Ảnh màu
        image = cv2.imread(fpath).astype(np.float32)  # Đọc ảnh từ file và chuyển sang kiểu float32
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Chuyển ảnh từ không gian màu BGR sang RGB

        # Ảnh xám
        image_gray = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE).astype(np.float32)  # Đọc ảnh dưới dạng ảnh xám
        image_gray = np.asarray([image_gray, image_gray, image_gray])  # Tạo ra 3 kênh xám để khớp với kênh màu

        image = np.transpose(image, (2, 0, 1))  # Chuyển đổi ảnh màu từ (H, W, C) sang (C, H, W)
        image = _transform(image)  # Chuẩn hóa ảnh màu
        image_gray = _transform(image_gray)  # Chuẩn hóa ảnh xám

        return torch.tensor(image), torch.tensor(image_gray)  # Trả về ảnh màu và ảnh xám dưới dạng tensor của PyTorch

    def load_anime_smooth(self, index):
        fpath = self.image_files[self.smooth][index]  # Lấy đường dẫn tới ảnh anime làm mịn tại chỉ số index

        # Ảnh xám
        image_gray = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE).astype(np.float32)  # Đọc ảnh dưới dạng ảnh xám
        image_gray = np.asarray([image_gray, image_gray, image_gray])  # Tạo ra 3 kênh xám để khớp với kênh màu

        image_gray = _transform(image_gray)  # Chuẩn hóa ảnh xám

        return torch.tensor(image_gray)  # Trả về ảnh làm mịn xám dưới dạng tensor của PyTorch


if __name__ == '__main__':
    import matplotlib.pyplot as plt  # Import matplotlib để hiển thị hình ảnh
    from torch.utils.data import DataLoader  # Import DataLoader để load dữ liệu

    # Tạo DataLoader cho dataset AnimeDataSet
    anime_loader = DataLoader(AnimeDataSet(data_dir='../dataset', dataset='Hayao'), batch_size=2, shuffle=True)

    # Lấy một mẫu từ dataset
    image, anime, anime_gray, smooth_gray = anime_loader.dataset[0]
    plt.imshow(image.numpy().transpose(1, 2, 0))  # Hiển thị ảnh thật sau khi đã chuẩn hóa
    plt.show()  # Hiển thị cửa sổ hình ảnh
