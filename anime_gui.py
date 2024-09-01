import cv2
import tkinter as tk
from tkinter import Button, Label
from PIL import Image, ImageTk
import torch
from torchvision import transforms
from AnimeGANv2 import AnimeGANv2  # Giả sử model của bạn nằm trong AnimeGANv2.py

# Đường dẫn tương đối đến checkpoint
CHECKPOINT_PATH = './checkpoint/animeGan/Hayao/epoch=8-step=15012.ckpt'

# Load model AnimeGAN từ checkpoint
model = AnimeGANv2(sn=True)  # Khởi tạo model từ AnimeGANv2
checkpoint = torch.load(CHECKPOINT_PATH, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# Chuẩn bị transform cho ảnh đầu vào
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),  # Kích thước đầu vào của model
])


def apply_anime_filter(frame):
    # Chuyển ảnh từ BGR sang RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb_frame)
    img = transform(img).unsqueeze(0)

    # Dự đoán ảnh anime
    with torch.no_grad():
        anime_img = model(img)

    # Chuyển ngược tensor về ảnh PIL
    anime_img = anime_img.squeeze(0).permute(1, 2, 0).numpy()
    anime_img = (anime_img * 255).astype('uint8')
    return anime_img


class AnimeApp:
    def __init__(self, window):
        self.window = window
        self.window.title("AnimeGAN App")

        # Thiết lập webcam
        self.cap = cv2.VideoCapture(0)
        self.width, self.height = 640, 480
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # Thiết lập giao diện
        self.canvas = tk.Canvas(window, width=self.width, height=self.height)
        self.canvas.pack()

        self.button_capture = Button(window, text="Chụp ảnh", command=self.capture_image)
        self.button_capture.pack()

        self.button_convert = Button(window, text="Chuyển đổi sang Anime", command=self.convert_to_anime)
        self.button_convert.pack()

        self.label_info = Label(window, text="")
        self.label_info.pack()

        self.update_frame()

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame
            # Hiển thị ảnh lên canvas
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.window.after(10, self.update_frame)

    def capture_image(self):
        # Lưu ảnh hiện tại
        cv2.imwrite('captured_image.jpg', self.current_frame)
        self.label_info.config(text="Ảnh đã được chụp và lưu thành công!")

    def convert_to_anime(self):
        # Áp dụng AnimeGAN model
        anime_frame = apply_anime_filter(self.current_frame)
        # Hiển thị ảnh Anime lên canvas
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(anime_frame))
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        # Lưu ảnh Anime
        cv2.imwrite('anime_image.jpg', cv2.cvtColor(anime_frame, cv2.COLOR_RGB2BGR))
        self.label_info.config(text="Ảnh đã được chuyển đổi và lưu thành công!")


if __name__ == "__main__":
    root = tk.Tk()
    app = AnimeApp(root)
    root.mainloop()
