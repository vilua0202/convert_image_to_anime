import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import os
import io
import tempfile
from test import test


class WebcamApp:
    def __init__(self, root):
        self.crop_image_to_save = None
        self.temp_image_io = None
        self.anime_image_name = None
        self.original_image_name = None
        self.root = root
        self.root.title("Webcam Anime Converter")

        # Trạng thái
        self.captured_image = None
        self.display_image = None
        self.start_x = self.start_y = self.end_x = self.end_y = 0
        self.is_capturing = True

        self.if_adjust_brightness = False

        # Trạng thái thông báo
        self.status_message = tk.StringVar()
        self.status_message.set("Sẵn sàng")

        self.model_dir = "checkpoint/animeGan/Hayao/epoch=8-step=15012.ckpt"

        # Thiết lập giao diện
        self.setup_ui()

        # Mở webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Lỗi", "Không thể mở webcam.")
            self.root.destroy()
            return

        # Bắt đầu cập nhật khung hình từ webcam
        self.update_frame()

    def setup_ui(self):
        # Khung hiển thị ảnh
        self.img_label = tk.Label(self.root)
        self.img_label.pack(pady=20)

        # Bắt sự kiện chuột để chọn vùng cắt
        self.img_label.bind("<ButtonPress-1>", self.on_mouse_down)
        self.img_label.bind("<B1-Motion>", self.on_mouse_drag)
        self.img_label.bind("<ButtonRelease-1>", self.on_mouse_up)

        # Tạo Frame chứa các nút
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=20)

        # Nút chụp ảnh
        self.capture_button = tk.Button(button_frame, text="Chụp ảnh", command=self.capture_image)
        self.capture_button.grid(row=0, column=0, padx=5, pady=5)

        # Nút cắt ảnh
        self.crop_button = tk.Button(button_frame, text="Cắt ảnh", command=self.crop_image, state=tk.DISABLED)
        self.crop_button.grid(row=0, column=1, padx=5, pady=5)

        # Nút chuyển thành anime
        self.anime_button = tk.Button(button_frame, text="Chuyển thành anime", command=self.convert_to_anime,
                                      state=tk.DISABLED)
        self.anime_button.grid(row=0, column=2, padx=5, pady=5)

        # Nút lưu ảnh
        self.save_button = tk.Button(button_frame, text="Lưu ảnh", command=self.save_image, state=tk.DISABLED)
        self.save_button.grid(row=0, column=3, padx=5, pady=5)

        # Nút tiếp tục quay
        self.continue_button = tk.Button(button_frame, text="Tiếp tục quay", command=self.continue_capture)
        self.continue_button.grid(row=0, column=4, padx=5, pady=5)

        # Nút hoàn tác
        self.undo_button = tk.Button(button_frame, text="Hoàn tác", command=self.undo, state=tk.DISABLED)
        self.undo_button.grid(row=0, column=5, padx=5, pady=5)

        # Nút tắt camera
        self.close_button = tk.Button(button_frame, text="Tắt cam", command=self.close_camera)
        self.close_button.grid(row=0, column=6, padx=5, pady=5)

        # Nút chọn ảnh từ hệ thống
        self.select_button = tk.Button(button_frame, text="Chọn ảnh", command=self.select_image)
        self.select_button.grid(row=0, column=7, padx=5, pady=5)

        # Thanh trạng thái
        self.status_label = tk.Label(self.root, textvariable=self.status_message, relief=tk.SUNKEN, anchor='w')
        self.status_label.pack(fill=tk.X, side=tk.BOTTOM, ipady=2)

    def update_frame(self):
        if self.is_capturing:
            ret, frame = self.cap.read()
            if ret:
                # Hiển thị ảnh bằng cách chuyển đổi từ BGR sang RGB (OpenCV -> PIL)
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image=img)
                self.img_label.imgtk = imgtk
                self.img_label.configure(image=imgtk)
        self.img_label.after(30, self.update_frame)  # Gọi lại hàm update_frame sau 30ms

    def capture_image(self):
        ret, frame = self.cap.read()
        if not ret:
            messagebox.showerror("Lỗi", "Không thể chụp ảnh từ webcam")
            return

        self.captured_image = frame
        self.is_capturing = False

        # Hiển thị ảnh đã chụp
        self.display_image = Image.fromarray(cv2.cvtColor(self.captured_image, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=self.display_image)
        self.img_label.config(image=imgtk)
        self.img_label.image = imgtk

        # Kích hoạt các nút chức năng
        self.crop_button.config(state=tk.NORMAL)
        self.save_button.config(state=tk.NORMAL)
        self.undo_button.config(state=tk.NORMAL)
        self.anime_button.config(state=tk.NORMAL)

        # Đặt tên gốc là 'captured_image.jpg' khi chụp ảnh từ webcam
        self.original_image_name = "captured_image.jpg"

    def save_image(self):
        image_to_save = self.crop_image_to_save if self.crop_image_to_save is not None else self.captured_image

        if image_to_save is None:
            messagebox.showwarning("Cảnh báo", "Không có ảnh nào để lưu")
            return

        # Hỏi tên file để lưu
        file_name = simpledialog.askstring("Lưu ảnh", "Nhập tên file để lưu:", initialvalue=self.original_image_name)
        if not file_name:
            return

        # Đường dẫn lưu ảnh
        save_directory = r"D:\python-project\animeGanv2_pytorch\dataset\real"
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Lưu ảnh
        file_path = os.path.join(save_directory, f"{file_name}.jpg")

        # Chuyển đổi từ PIL Image sang định dạng numpy array để lưu bằng OpenCV
        image_to_save_cv = np.array(image_to_save)

        # Nếu ảnh là ảnh cắt từ ảnh gốc, thì cần chuyển đổi từ RGB sang BGR trước khi lưu bằng OpenCV
        if image_to_save is self.crop_image_to_save:
            image_to_save_cv = cv2.cvtColor(image_to_save_cv, cv2.COLOR_RGB2BGR)

        cv2.imwrite(file_path, image_to_save_cv)  # Lưu ảnh một lần duy nhất trên đĩa

        self.original_image_name = f"{file_name}.jpg"  # Cập nhật tên ảnh gốc sau khi lưu
        messagebox.showinfo("Thành công", f"Ảnh đã được lưu tại: {file_path}")


    def select_image(self):
        self.is_capturing = False
        file_path = filedialog.askopenfilename(title="Chọn ảnh để hiển thị",
                                               filetypes=[("Image files", "*.jpg *.jpeg *.png *.tiff")])
        if file_path:
            self.display_image = Image.open(file_path)
            self.temp_image_io = None  # Đặt lại temp_image_io khi chọn ảnh mới
            imgtk = ImageTk.PhotoImage(image=self.display_image)
            self.img_label.config(image=imgtk)
            self.img_label.image = imgtk

            # Lưu lại tên ảnh gốc
            self.original_image_name = os.path.basename(file_path)

            # Kích hoạt các nút chức năng
            self.anime_button.config(state=tk.NORMAL)
            self.crop_button.config(state=tk.NORMAL)
            self.save_button.config(state=tk.NORMAL)
            self.undo_button.config(state=tk.NORMAL)

    def convert_to_anime(self):
        if self.display_image is None:
            messagebox.showwarning("Cảnh báo", "Không có ảnh nào để chuyển thành anime")
            return
        try:
            self.status_message.set("Đang chuyển đổi ảnh thành anime...")
            self.root.update_idletasks()

            # Nếu có ảnh tạm trong BytesIO, lưu nó ra file với tên gốc
            if self.temp_image_io is not None:
                with open(self.original_image_name, 'wb') as f:
                    f.write(self.temp_image_io.getvalue())

            # Đảm bảo ảnh được lưu trên đĩa trước khi chuyển đổi
            image_input_path = os.path.join(r"D:\python-project\animeGanv2_pytorch\dataset\real",
                                            self.original_image_name)
            self.display_image.save(image_input_path)

            # Gọi hàm test để chuyển đổi ảnh thành anime, truyền vào đúng đường dẫn ảnh
            test(self.model_dir, image_input_path, self.if_adjust_brightness)

            # Xác định đường dẫn của ảnh anime đã được lưu tự động
            anime_image_path = os.path.join(r"D:\python-project\animeGanv2_pytorch\results", self.original_image_name)

            # Kiểm tra xem ảnh anime có tồn tại không
            if os.path.exists(anime_image_path):
                # Hiển thị ảnh anime sau khi xử lý
                anime_image = Image.open(anime_image_path)
                imgtk_anime = ImageTk.PhotoImage(image=anime_image)
                self.img_label.config(image=imgtk_anime)
                self.img_label.image = imgtk_anime
                self.status_message.set(f"Ảnh đã được chuyển thành anime và lưu tại: {anime_image_path}")
                messagebox.showinfo("Thành công", f"Ảnh đã được chuyển thành anime và lưu tại: {anime_image_path}")
            else:
                messagebox.showerror("Lỗi", "Không thể tìm thấy ảnh anime sau khi chuyển đổi")
                self.status_message.set("Lỗi khi chuyển đổi ảnh thành anime.")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi chuyển đổi ảnh thành anime: {e}")
            self.status_message.set("Lỗi khi chuyển đổi ảnh thành anime.")

    def on_mouse_down(self, event):
        self.start_x, self.start_y = event.x, event.y

    def on_mouse_drag(self, event):
        self.end_x, self.end_y = event.x, event.y
        if self.display_image:
            img_copy = self.display_image.copy()
            draw = ImageDraw.Draw(img_copy)
            draw.rectangle([self.start_x, self.start_y, self.end_x, self.end_y], outline="red", width=2)
            imgtk = ImageTk.PhotoImage(image=img_copy)
            self.img_label.config(image=imgtk)
            self.img_label.image = imgtk

    def on_mouse_up(self, event):
        self.end_x, self.end_y = event.x, event.y

    def crop_image(self):
        if not self.display_image or (self.start_x == self.end_x or self.start_y == self.end_y):
            messagebox.showwarning("Cảnh báo", "Vùng cắt không hợp lệ")
            return

        # Cắt ảnh theo vùng được chọn
        crop_box = (self.start_x, self.start_y, self.end_x, self.end_y)
        cropped_image = self.display_image.crop(crop_box)

        # Hiển thị ảnh sau khi cắt
        self.crop_image_to_save = cropped_image
        self.display_image = cropped_image
        self.temp_image_io = None  # Đặt lại temp_image_io sau khi cắt ảnh
        imgtk = ImageTk.PhotoImage(image=self.display_image)
        self.img_label.config(image=imgtk)
        self.img_label.image = imgtk

    def continue_capture(self):
        self.is_capturing = True
        self.update_frame()

    def close_camera(self):
        self.is_capturing = False
        self.cap.release()
        self.img_label.config(image='')  # Xóa hình ảnh đang hiển thị
        self.root.destroy()

    def undo(self):
        if self.captured_image is not None:
            self.display_image = Image.fromarray(cv2.cvtColor(self.captured_image, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=self.display_image)
            self.img_label.config(image=imgtk)
            self.img_label.image = imgtk
            self.status_message.set("Hoàn tác thành công.")
        else:
            messagebox.showwarning("Cảnh báo", "Không có ảnh nào để hoàn tác.")


# Khởi tạo giao diện với Tkinter
root = tk.Tk()
app = WebcamApp(root)
# Khởi động vòng lặp giao diện
root.mainloop()

# Giải phóng camera khi đóng giao diện
if app.cap.isOpened():
    app.cap.release()
cv2.destroyAllWindows()
