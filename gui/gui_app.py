import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2

from video_processor import VideoProcessor

class UAVApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("UAV Evasion System")
        self.geometry("1400x900")
        self.processor = VideoProcessor()
        self.setup_ui()
        self.camera_active = False
        self.update_video()

    def setup_ui(self):

        # === LAYOUT ===
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Sidebar
        self.sidebar = ctk.CTkFrame(self, width=300, corner_radius=0)
        self.sidebar.grid(row=0, column=0, rowspan=2, sticky="nsew")
        self.sidebar.grid_rowconfigure(10, weight=1)

        # Main area
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        # === SIDEBAR CONTENT ===

        # Logo/Title
        self.logo_label = ctk.CTkLabel(
            self.sidebar,
            text="UAV EVASION",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # === VIDEO SECTION ===
        self.video_label = ctk.CTkLabel(
            self.sidebar,
            text="Video",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.video_label.grid(row=1, column=0, padx=20, pady=(20, 5))

        self.load_btn = ctk.CTkButton(
            self.sidebar,
            text="Load Video",
            command=self.load_video
        )
        self.load_btn.grid(row=2, column=0, padx=20, pady=5)
        # === VIDEO ===
        self.video_label = ctk.CTkLabel(
            self.sidebar,
            text="📹 Video Source",
            font=("Arial", 16, "bold")
        )
        self.video_label.grid(row=1, column=0, padx=20, pady=(20, 10))

        # Load Video
        self.load_btn = ctk.CTkButton(
            self.sidebar,
            text="📁 Load Video",
            command=self.load_video,
            height=35
        )
        self.load_btn.grid(row=2, column=0, padx=20, pady=5)

        self.camera_btn = ctk.CTkButton(
            self.sidebar,
            text="Start Camera",
            command=self.start_camera,
            height=35
        )
        self.camera_btn.grid(row=3, column=0, padx=20, pady=5)

        self.stop_camera_btn = ctk.CTkButton(
            self.sidebar,
            text="Stop Camera",
            command=self.stop_camera,
            height=35,
            fg_color="red",
        )
        self.stop_camera_btn.grid(row=4, column=0, padx=20, pady=5)
        # === DETECTION SECTION ===
        self.detection_label = ctk.CTkLabel(
            self.sidebar,
            text="Detection",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.detection_label.grid(row=5, column=0, padx=20, pady=(20, 5))

        # Model
        self.model_var = ctk.StringVar(value="yolov8s_trained.pt")

        self.model_small = ctk.CTkRadioButton(
            self.sidebar,
            text="YOLOv8s",
            variable=self.model_var,
            value="yolov8s_trained.pt",
        )
        self.model_small.grid(row=6, column=0, padx=20, pady=2, sticky="w")

        # === DISPLAY SECTION ===
        self.display_label = ctk.CTkLabel(
            self.sidebar,
            text="Display",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.display_label.grid(row=9, column=0, padx=20, pady=(20, 5))

        self.show_trails_var = ctk.BooleanVar(value=True)
        self.trails_check = ctk.CTkCheckBox(
            self.sidebar,
            text="Show Trajectories",
            variable=self.show_trails_var,
            command=self.update_display
        )
        self.trails_check.grid(row=10, column=0, padx=20, pady=2, sticky="w")

        self.show_velocity_var = ctk.BooleanVar(value=True)
        self.velocity_check = ctk.CTkCheckBox(
            self.sidebar,
            text="Show Velocities",
            variable=self.show_velocity_var,
            command=self.update_display
        )
        self.velocity_check.grid(row=11, column=0, padx=20, pady=2, sticky="w")

        self.show_cone_var = ctk.BooleanVar(value=True)
        self.cone_check = ctk.CTkCheckBox(
            self.sidebar,
            text="Show Uncertainty Cone",
            variable=self.show_cone_var,
            command=self.update_display
        )
        self.cone_check.grid(row=12, column=0, padx=20, pady=2, sticky="w")

        # === GUIDANCE SECTION ===
        self.guidance_label = ctk.CTkLabel(
            self.sidebar,
            text="Guidance",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.guidance_label.grid(row=13, column=0, padx=20, pady=(20, 5))

        # === MAIN AREA ===

        # Video display
        self.video_canvas = ctk.CTkLabel(
            self.main_frame,
            text="No video loaded\nClick 'Load Video' to start",
            font=ctk.CTkFont(size=20)
        )
        self.video_canvas.grid(row=0, column=0, sticky="nsew")

        # Controls
        self.controls_frame = ctk.CTkFrame(self.main_frame)
        self.controls_frame.grid(row=1, column=0, sticky="ew", pady=(10, 0))

        self.play_btn = ctk.CTkButton(
            self.controls_frame,
            text="▶ Play",
            command=self.play_video,
            width=100
        )
        self.play_btn.grid(row=0, column=0, padx=5, pady=10)

        self.pause_btn = ctk.CTkButton(
            self.controls_frame,
            text="⏸ Pause",
            command=self.pause_video,
            width=100
        )
        self.pause_btn.grid(row=0, column=1, padx=5, pady=10)


        self.screenshot_btn = ctk.CTkButton(
            self.controls_frame,
            text="Screenshot",
            command=self.take_screenshot,
            width=120
        )
        self.screenshot_btn.grid(row=0, column=3, padx=5, pady=10)

        # Stats
        self.stats_label = ctk.CTkLabel(
            self.controls_frame,
            text="Frame: 0/0 (0%) | FPS: 0.0 | Tracks: 0"
        )
        self.stats_label.grid(row=0, column=4, padx=20, pady=10)

    # === CALLBACKS ===

    def load_video(self):
        filepath = filedialog.askopenfilename(
            title="Select video",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*")
            ]
        )

        if not filepath:
            return

        try:
            info = self.processor.load_video(filepath)
            messagebox.showinfo(
                "Video Loaded",
                f"Resolution: {info['width']}x{info['height']}\n"
                f"FPS: {info['fps']}\n"
                f"Frames: {info['total_frames']}"
            )
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def play_video(self):
        try:
            self.processor.start()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def pause_video(self):
        self.processor.pause()

    def start_camera(self):
        try:
            camera_id = 0

            info = self.processor.load_camera(camera_id)
            self.camera_id = True
            messagebox.showinfo(
                "Camera Started",
                f"Camera {camera_id} connected\n"
                f"Resolution: {info['width']}x{info['height']}\n"
                f"FPS: {info['fps']}"
            )

            # Автоматично запускаємо
            self.processor.start()

        except Exception as e:
            messagebox.showerror("Camera Error", str(e))

    def stop_camera(self):
        if self.processor.is_running:
            self.processor.is_running = False

            if self.processor.thread:
                self.processor.thread.join(timeout=1.0)

            if self.processor.cap:
                self.processor.cap.release()
            self.camera_active = False
            self.processor.current_frame = None
            self.processor.processed_frame = None
            if hasattr(self.video_canvas, 'image'):
                delattr(self.video_canvas, 'image')

            self.video_canvas.configure(
                text="No video loaded\nClick 'Load Video' or 'Start Camera' to start",
                image=""
            )
            messagebox.showinfo("Camera", "Camera stopped")

    def take_screenshot(self):
        frame = self.processor.get_frame()
        if frame is not None:
            filepath = filedialog.asksaveasfilename(
                defaultextension=".jpg",
                filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")]
            )
            if filepath:
                cv2.imwrite(filepath, frame)
                messagebox.showinfo("Saved", f"Screenshot saved to {filepath}")


    def update_display(self):
        self.processor.update_settings({
            'show_trails': self.show_trails_var.get(),
            'show_velocity': self.show_velocity_var.get(),
            'show_cone': self.show_cone_var.get()
        })

    def update_video(self):

        if not self.camera_active and not self.processor.is_running:
            self.after(30, self.update_video)
            return
        frame = self.processor.get_frame()

        if frame is not None:
            # Конвертація OpenCV -> PIL -> ImageTk
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize для відображення
            h, w = frame_rgb.shape[:2]
            max_w = 1000
            max_h = 700

            scale = min(max_w / w, max_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)

            frame_resized = cv2.resize(frame_rgb, (new_w, new_h))

            img = Image.fromarray(frame_resized)
            imgtk = ImageTk.PhotoImage(image=img)

            self.video_canvas.configure(image=imgtk, text="")
            self.video_canvas.image = imgtk

        stats = self.processor.get_stats()
        self.stats_label.configure(
            text=f"Frame: {stats['frame_count']}/{stats['total_frames']} "
                 f"({stats['frame_count'] / stats['total_frames'] * 100 if stats['total_frames'] > 0 else 0:.1f}%) | "
                 f"FPS: {stats['fps']:.1f}"
        )

        self.after(30, self.update_video)

    def on_closing(self):
        if self.processor.is_running:
            self.processor.is_running = False
            if self.processor.thread:
                self.processor.thread.join(timeout=1)
        if self.processor.cap:
            self.processor.cap.release()

        self.destroy()


if __name__ == "__main__":
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")

    app = UAVApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()