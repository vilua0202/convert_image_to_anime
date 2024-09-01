# Chuyển Đổi Ảnh Sang Anime

## Tổng Quan

Chuyển Đổi Ảnh Webcam Sang Anime là một ứng dụng Python cho phép bạn chụp ảnh từ webcam hoặc tải ảnh từ máy tính, thực
hiện các chỉnh sửa cơ bản (như cắt ảnh) và sau đó chuyển đổi ảnh thành phong cách anime sử dụng mô hình AnimeGAN đã được
huấn luyện trước.

## Tính Năng

- **Chụp Ảnh:** Chụp ảnh trực tiếp từ webcam của bạn.
- **Cắt Ảnh:** Chọn và cắt một vùng cụ thể của ảnh đã chụp hoặc ảnh đã tải lên.
- **Chuyển Đổi Sang Anime:** Chuyển đổi ảnh đã chụp hoặc đã chọn sang phong cách anime sử dụng mô hình AnimeGAN.
- **Lưu Ảnh:** Lưu ảnh đã chụp, đã cắt, hoặc ảnh đã chuyển đổi sang phong cách anime vào hệ thống của bạn.
- **Hoàn Tác:** Khôi phục lại ảnh gốc đã chụp.
- **Chọn Ảnh:** Tải một ảnh từ hệ thống của bạn để chuyển đổi.

## Phụ Thuộc

Ứng dụng yêu cầu các gói Python sau:

- python 3.10.0
- pip 23.0.1
- torch==1.11.0 pytorch-lightning==1.7.7
- torchmetrics==0.7.0
- tqdm==4.64.0 opencv-python==4.5.5.64 wandb pyyaml
- torchtext==0.12.0
- onnx
- onnx-simplifier
- numpy<2
- pybind11>=2.12

## Huấn luyện

1. Tải xuống mô hình vgg19 va mô hình đã huấn luyện trước
    - [vgg19.npy](https://drive.google.com/drive/folders/1Yc0lj5qy1RRSdyMb_AX6bEiJtriK_Cj3)
    - [checkpoint](https://drive.google.com/drive/folders/1Yc0lj5qy1RRSdyMb_AX6bEiJtriK_Cj3)

## Sử Dụng

1. Chạy ứng dụng:
   ```bash
   python WebcamApp.py
   ```

2. Sử dụng giao diện để tương tác với webcam, chụp ảnh, cắt ảnh và chuyển đổi ảnh sang phong cách anime.

3. Lưu ảnh theo nhu cầu.

## Lưu Ý

- Đảm bảo webcam của bạn hoạt động tốt trước khi chạy ứng dụng.
- Mô hình AnimeGAN cần được thiết lập chính xác và hàm `test` cần được triển khai trong `test.py`.


