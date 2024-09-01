import argparse
import time

import torch
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger

from AnimeGANInitTrain import AnimeGANInitTrain
from AnimeGANv2 import AnimeGANv2
from tools.AnimeGanDataModel import AnimeGANDataModel
from tools.utils import *

"""parsing and configuration"""
# Hàm để phân tích và lấy các tham số từ dòng lệnh

def parse_args():
    desc = "AnimeGANv2"  # Mô tả ngắn về chương trình
    parser = argparse.ArgumentParser(description=desc)  # Tạo đối tượng ArgumentParser để nhận tham số từ dòng lệnh
    parser.add_argument('--config_path', type=str, help='hyper params config path', required=True)
    # Thêm tham số `--config_path` bắt buộc phải có, dùng để chỉ đường dẫn tới file cấu hình
    parser.add_argument('--img_size', type=list, default=[256, 256], help='The size of image: H and W')
    # Thêm tham số `--img_size` để chỉ định kích thước của ảnh (chiều cao và chiều rộng), mặc định là [256, 256]
    parser.add_argument('--img_ch', type=int, default=3, help='The size of image channel')
    # Thêm tham số `--img_ch` để chỉ định số kênh màu của ảnh, mặc định là 3 (RGB)
    parser.add_argument('--ch', type=int, default=64, help='base channel number per layer')
    # Thêm tham số `--ch` để chỉ định số kênh cơ bản cho mỗi lớp của mô hình, mặc định là 64
    parser.add_argument('--n_dis', type=int, default=3, help='The number of discriminator layer')
    # Thêm tham số `--n_dis` để chỉ định số lượng lớp phân biệt trong mạng phân biệt, mặc định là 3
    parser.add_argument('--pre_train_weight', type=str, required=False, help='pre-trained weight path, tensorflow checkpoint directory')
    # Thêm tham số `--pre_train_weight` để chỉ định đường dẫn tới trọng số đã được huấn luyện trước, tùy chọn
    parser.add_argument('--resume_ckpt_path', type=str, required=False, help='resume checkpoint path')
    # Thêm tham số `--resume_ckpt_path` để chỉ định đường dẫn tới checkpoint để tiếp tục huấn luyện từ giữa chừng, tùy chọn
    parser.add_argument('--init_train_flag', type=str, required=True, default='False')
    # Thêm tham số `--init_train_flag` để xác định xem có thực hiện huấn luyện khởi tạo hay không, bắt buộc phải có

    return check_args(parser.parse_args())  # Trả về kết quả sau khi kiểm tra các tham số đã nhập

"""checking arguments"""
# Hàm để kiểm tra các tham số đã nhập có hợp lệ không

def check_args(args):
    # --epoch
    try:
        assert args.config_path  # Kiểm tra xem tham số `config_path` có tồn tại không
    except:
        print('config_path is required')  # Nếu không tồn tại, in ra thông báo lỗi
    return args  # Trả về các tham số đã kiểm tra

"""main"""
# Hàm chính của chương trình

def main():
    # parse arguments
    args = parse_args()  # Gọi hàm parse_args để lấy các tham số từ dòng lệnh
    if args is None:
        exit()  # Nếu không có tham số, thoát chương trình
    config_dict = yaml.safe_load(open(args.config_path, 'r'))  # Đọc file cấu hình YAML từ đường dẫn được cung cấp

    # Nếu cờ `init_train_flag` được bật
    if args.init_train_flag.lower() == 'true':
        # Khởi tạo mô hình AnimeGANInitTrain với các tham số đã cung cấp
        model = AnimeGANInitTrain(args.img_size, config_dict['dataset']['name'], **config_dict['model'])
        # Tạo thư mục lưu checkpoint cho quá trình huấn luyện khởi tạo
        check_folder(os.path.join('checkpoint/initAnimeGan', config_dict['dataset']['name']))
        # Định nghĩa callback để lưu checkpoint mỗi epoch
        checkpoint_callback = ModelCheckpoint(dirpath=os.path.join('checkpoint/initAnimeGan', config_dict['dataset']['name']),
                                              monitor='epoch',
                                              mode='max',
                                              save_top_k=-1)
        # Khởi tạo logger cho TensorBoard để theo dõi quá trình huấn luyện
        tensorboard_logger = TensorBoardLogger(save_dir='logs/initAnimeGan')
        # Khởi tạo logger cho Wandb để theo dõi quá trình huấn luyện trên Wandb
        wandb_logger = WandbLogger(project='AnimeGanV2_init_pytorch',
                                   name='initAnimeGan_{}_{}'.format(config_dict['dataset']['name'],
                                                                    time.strftime("%Y-%m-%d_%H:%M", time.localtime())))
        # Khởi tạo Trainer của PyTorch Lightning với các tham số đã định nghĩa
        trainer = Trainer(
            accelerator='auto',
            max_epochs=config_dict['trainer']['epoch'],
            callbacks=[checkpoint_callback],
            logger=[tensorboard_logger, wandb_logger]
        )
        # In ra các thông tin về quá trình huấn luyện
        print()
        print("##### Information #####")
        print("# dataset : ", config_dict['dataset']['name'])
        print("# batch_size : ", config_dict['dataset']['batch_size'])
        print("# epoch : ", config_dict['trainer']['epoch'])
        print("# training image size [H, W] : ", args.img_size)
        print("#con_weight,sty_weight : ", config_dict['model']['con_weight'])
        print("#init_lr: ", config_dict['model']['init_lr'])
        print()
    else:  # Nếu không phải huấn luyện khởi tạo
        # Khởi tạo mô hình AnimeGANv2 với các tham số đã cung cấp
        model = AnimeGANv2(args.ch, args.n_dis, args.img_size, config_dict['dataset']['name'], args.pre_train_weight,
                           **config_dict['model'])
        # Định nghĩa callback để lưu checkpoint mỗi epoch
        checkpoint_callback = ModelCheckpoint(dirpath=os.path.join('checkpoint/animeGan', config_dict['dataset']['name']),
                                              save_top_k=-1,
                                              monitor='epoch', mode='max')
        # Khởi tạo logger cho TensorBoard để theo dõi quá trình huấn luyện
        tensorboard_logger = TensorBoardLogger(save_dir='logs/animeGan')
        # Khởi tạo logger cho Wandb để theo dõi quá trình huấn luyện trên Wandb
        wandb_logger = WandbLogger(project='AnimeGanV2_pytorch',
                                   name='animeGan_{}_{}'.format(config_dict['dataset']['name'],
                                                                time.strftime("%Y-%m-%d_%H:%M", time.localtime())))
        # Khởi tạo Trainer của PyTorch Lightning với các tham số đã định nghĩa
        trainer = Trainer(
            accelerator='auto',
            max_epochs=config_dict['trainer']['epoch'],
            callbacks=[checkpoint_callback],
            logger=[tensorboard_logger, wandb_logger]
        )
        # In ra các thông tin về quá trình huấn luyện
        print()
        print("##### Information #####")
        print("# dataset : ", config_dict['dataset']['name'])
        print("# batch_size : ", config_dict['dataset']['batch_size'])
        print("# epoch : ", config_dict['trainer']['epoch'])
        print("# training image size [H, W] : ", args.img_size)
        print("# g_adv_weight,d_adv_weight,con_weight,sty_weight,color_weight,tv_weight : ",
              config_dict['model']['g_adv_weight'],
              config_dict['model']['d_adv_weight'],
              config_dict['model']['con_weight'],
              config_dict['model']['sty_weight'],
              config_dict['model']['color_weight'],
              config_dict['model']['tv_weight'])
        print("#g_lr,d_lr : ", config_dict['model']['g_lr'], config_dict['model']['d_lr'])
        print()

    # Khởi tạo đối tượng AnimeGANDataModel để quản lý dữ liệu đầu vào
    dataModel = AnimeGANDataModel(data_dir=config_dict['dataset']['path'],
                                  dataset=config_dict['dataset']['name'],
                                  batch_size=config_dict['dataset']['batch_size'],
                                  num_workers=config_dict['dataset']['num_workers'])
    # Nếu có đường dẫn checkpoint để tiếp tục huấn luyện từ giữa chừng
    if args.resume_ckpt_path:
        print("resume from checkpoint:", args.resume_ckpt_path)
        # Huấn luyện mô hình với checkpoint đã cho
        trainer.fit(model, dataModel, ckpt_path=args.resume_ckpt_path)
    else:  # Nếu không có checkpoint, bắt đầu huấn luyện từ đầu
        trainer.fit(model, dataModel)

    # Chuyển mô hình sang định dạng ONNX để dễ dàng triển khai trên các nền tảng khác
    model.to_onnx(file_path='animeGan.onnx', input_sample=torch.randn(1, 3, 256, 256))

    # Lưu trạng thái của mô hình dưới dạng file `.pth`
    # `state_dict()` lấy tất cả các tham số và buffer của mô hình
    # `torch.save()` lưu trữ trạng thái này vào file 'animeGan.pth'
    torch.save(model.generated.state_dict(), 'animeGan.pth')

    # In thông báo kết thúc quá trình huấn luyện
    print("[*] Training finished!")


if __name__ == '__main__':
    main()  # Gọi hàm main để chạy chương trình
