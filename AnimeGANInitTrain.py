from collections import OrderedDict  # Import OrderedDict để duy trì thứ tự của các phần tử trong từ điển
from glob import glob  # Import glob để tìm kiếm các file theo pattern

import pytorch_lightning as pl  # Import pytorch_lightning để sử dụng mô hình huấn luyện linh hoạt
import torch.nn  # Import torch.nn để làm việc với các lớp mạng thần kinh
from torch.optim import Adam  # Import Adam từ torch.optim để tối ưu hóa mô hình

import wandb  # Import wandb để log quá trình huấn luyện và đánh giá mô hình
from net.backtone import VGGCaffePreTrained  # Import mô hình VGG đã được huấn luyện trước
from net.generator import Generator  # Import lớp Generator
from tools.ops import *  # Import các hàm hỗ trợ từ file ops
from tools.utils import *  # Import các hàm tiện ích từ file utils


##################################################################################
# Model
##################################################################################
class AnimeGANInitTrain(pl.LightningModule):
    def __init__(self, img_size=None, dataset_name=None, **kwargs):
        super().__init__()
        self.save_hyperparameters()  # Lưu các tham số hyperparameters được truyền vào

        if img_size is None:
            img_size = [256, 256]  # Mặc định kích thước ảnh nếu không được cung cấp
        self.img_size = img_size
        self.p_model = VGGCaffePreTrained().eval()  # Khởi tạo mô hình VGG đã được huấn luyện trước và chuyển sang chế độ đánh giá
        """ Define Generator """
        self.generated = Generator()  # Khởi tạo Generator

    def on_fit_start(self):
        self.p_model.setup(self.device)  # Thiết lập mô hình VGG để phù hợp với thiết bị (GPU/CPU)

    def forward(self, img):
        return self.generated(img)  # Forward qua Generator để tạo ra ảnh giả

    def training_step(self, batch, batch_idx):
        real, anime, anime_gray, anime_smooth = batch  # Lấy các batch dữ liệu đầu vào
        generator_images = self.generated(real)  # Tạo ảnh giả từ ảnh thật
        # Giai đoạn huấn luyện khởi tạo
        init_c_loss = con_loss(self.p_model, real, generator_images)  # Tính content loss dựa trên VGG
        init_loss = self.hparams.con_weight * init_c_loss  # Tính tổng loss với trọng số content loss

        self.log('init_loss', init_loss, on_step=True, prog_bar=True, logger=True)  # Log giá trị loss khởi tạo
        return init_loss  # Trả về giá trị loss để tối ưu hóa

    def on_fit_end(self) -> None:
        # log các ảnh trong tập val lên wandb sau khi huấn luyện kết thúc
        val_files = glob('./dataset/{}/*.*'.format('val'))  # Lấy danh sách các file trong thư mục val
        val_images = []
        for i, sample_file in enumerate(val_files):
            print('val: ' + str(i) + sample_file)
            self.generated.eval()  # Chuyển Generator sang chế độ đánh giá
            if i == 0 or i == 26 or i == 5:  # Chỉ log một số lượng ảnh nhất định
                with torch.no_grad():
                    sample_image = np.asarray(load_test_data(sample_file))  # Đọc ảnh từ file
                    test_real = torch.from_numpy(sample_image).type_as(self.generated.out_layer[0].weight)  # Chuẩn bị dữ liệu đầu vào
                    test_generated_predict = self.generated(test_real)  # Tạo ảnh giả
                    test_generated_predict = test_generated_predict.permute(0, 2, 3, 1).cpu().detach().numpy()  # Chuyển đổi định dạng ảnh
                    test_generated_predict = np.squeeze(test_generated_predict, axis=0)  # Loại bỏ chiều không cần thiết
                    val_images.append(
                        wandb.Image(test_generated_predict, caption="Name:{}, epoch:{}".format(i, self.current_epoch)))
        wandb.log({"val_images": val_images})  # Log các ảnh đánh giá lên wandb

    def configure_optimizers(self):
        G_optim = Adam(self.generated.parameters(), lr=self.hparams.init_lr, betas=(0.5, 0.999))  # Cấu hình Adam optimizer cho Generator
        return G_optim  # Trả về optimizer cho Generator
