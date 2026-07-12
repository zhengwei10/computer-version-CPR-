import os
import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import time
import threading
from collections import deque
import matplotlib.animation as animation
import math
from tkinter import filedialog
from queue import Empty
from ultralytics import YOLO  # YOLO
from PIL import Image, ImageTk
from tkinter import messagebox
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
# 在程序最开头添加这段代码
import ctypes
winmm = ctypes.WinDLL('winmm')
winmm.timeBeginPeriod(1)  # 将定时器分辨率设置为1ms
os.environ['ULTRALYTICS_VERBOSE'] = '0'
os.environ['YOLO_VERBOSE'] = '0'
class ChestCompressionEvaluator:
    def __init__(self, root):
        self.root = root
        self.root.title("胸外按压操作评估系统v1.0 - YOLO双臂显示版)")
        self.root.geometry("1200x800")
        self.out = None  # 视频写入对象
        self.save_video = True  # 是否保存视频
        self.min_loop_interval = 16   # 强制主线程60fps更新
        self.save_fps = None  # 会被真实帧率覆盖
        self.real_camera_fps = None
        self.frame_interval_ms = None
        self.inference_lock = threading.Lock()
        self.stop_capture_event = threading.Event()   # 用于停止帧捕获线程
        self.capture_thread = None                    # 线程对象
        self.is_recording = False                     # 录制状态标志
        # 🔥 双帧率架构变量
        self.last_inference_result = None  # 缓存最新推理结果（仅关键点）
        self.inference_fps = 0.0  # 单独统计推理帧率
        self.display_fps = 0.0    # 单独统计显示帧率
        
        # 初始化变量
        self.is_running = False
        self.cap = None
        self.compression_count = 0
        self.correct_depth_count = 0
        self.correct_frequency_count = 0
        self.start_time = time.time()
        self.compression_times = deque(maxlen=120)
        self.compression_depths = deque(maxlen=120)
        self.current_depth = 0
        self.current_frequency = 0
        self.score = 0
        self.canvas_image_id = None
        # 上肢角度相关变量 - 现在记录双臂
        self.right_elbow_angle = 0
        self.left_elbow_angle = 0
        self.right_press_vertical_angle = 0
        self.left_press_vertical_angle = 0
        self.correct_elbow_count = 0
        self.correct_vertical_count = 0
        self.elbow_angle_threshold = 10
        self.vertical_angle_threshold = 15

        # YOLO初始化
        # 直接加载 TensorRT 引擎，不需要 .cuda()
        self.model = YOLO('yolo26s-pose.engine')
        #self.model = YOLO('yolo26s-pose.pt').cuda()
    

        
        # 图表变量
        self.time_data = deque(maxlen=120)
        self.depth_data = deque(maxlen=120)
        self.chart_start_time = time.time()

        # 深度相关参数 - 修复核心
        self.reference_y = None  # 参考点（按压起始位置）
        self.min_y = None        # 按压最低点
        self.max_y = None        # 按压最高点
        self.pixel_per_cm = None  # 修复：初始化为None，强制先标定
        self.cm_per_pixel = None  # 新增：厘米/像素，更直观的转换系数
        self.depth_threshold_cm = 2.0
        self.depth_threshold = 20
        self.is_pressing = False
        self.last_wrist_y = None
        self.press_max_depth = 0
        self.press_min_y = None
        self.correct_recoil_count = 0
        # 按压周期位置记录
        self.press_start_y = None  # 按压起始点y坐标（最高点）
        self.press_start_depth = 0  # 按压起始点深度
        self.start_depth_list = []  # 存储所有按压起始点深度
        self.min_depth_list = []    # 存储所有按压最低点深度
        self.cycle_highest_y = None   # 一个周期的最高点（回弹）
        self.cycle_lowest_y = None    # 一个周期的最低点（按压深度）
                # 手动校准相关变量
        self.calibrating = False
        self.is_calibrating_video = False  # 新增：标定专用视频状态
        self.calibration_points = []
        self.actual_length_cm = 16.0  # 标定物实际长度（厘米）
        self.current_frame = None
        self.original_frame_size = None
        self.display_frame_size = None

        # 运动追踪变量
        self.prev_operator_landmarks = None
        self.prev_operator_wrist_y = None
        self.operator_movement_score = 0
        self.operator_stability_frames = 0
        self.operator_id = None
        self.tracking_history = []
        self.tracking_history_size = 5
        self.current_detected_people = []
        self.last_tracking_time = time.time()
        self.min_movement_threshold = 0.01
        self.operator_confidence = 0
        self.max_operator_confidence = 10
        
        # 多人检测跟踪
        self.prev_frame_people = []
        self.person_tracking_ids = []
        self.next_tracking_id = 0
        self.tracking_data = {}
        # 多线程共享帧（避免 cap.read() 竞争）
        self.frame_lock = threading.Lock()
        self.latest_frame = None
        
        # 🔥 优化3：增大队列缓冲，解决推理线程饥饿问题
        from queue import Queue
        self.frame_queue = Queue(maxsize=10)  # 从2改为10，足够缓冲
        self.result_queue = Queue(maxsize=10) # 推理结果队列
        
        # 推理线程控制
        self.inference_thread = None
        self.inference_running = False
        
        # 帧率统计变量
        self.fps_start_time = time.perf_counter()
        self.fps_frame_count = 0
        self.current_fps = 0.0
        
        # 创建界面
        self.create_widgets()

        # 初始化摄像头
        self.init_camera()

        # 图表更新频率改为300ms（3fps足够，人眼看不出区别）
        self.ani = animation.FuncAnimation(
            self.fig, self.animate_chart, interval=200, blit=True, cache_frame_data=False
        )
    # 修改inference_worker函数，去掉timeout，用非阻塞方式

    def inference_worker(self):
        """🔥 极致优化的推理线程：只做推理，零IO，零冗余"""
        inference_frame_count = 0
        inference_start_time = time.perf_counter()
        
        while self.inference_running:
            try:
                # 🔥 非阻塞取帧，没有就立即循环，避免线程挂起
                frame = self.frame_queue.get_nowait()
                h, w = frame.shape[:2]
                
                # YOLO 姿态检测（纯推理，无任何输出）
                results = self.model(frame, verbose=False)
                current_keypoints_list = []

                if results:
                    for r in results:
                        if r.keypoints is not None:
                            kpts = r.keypoints.xy.cpu().numpy()
                            for person_kpts in kpts:
                                current_keypoints_list.append(person_kpts)
                
                # 识别操作者
                operator_landmarks = self.identify_operator_by_motion(current_keypoints_list)
                with self.inference_lock:
                    self.last_inference_result = operator_landmarks
                self.result_queue.put((operator_landmarks, h, w))
                
                # 统计推理帧率（无打印，仅更新变量）
                inference_frame_count += 1
                elapsed = time.perf_counter() - inference_start_time
                if elapsed >= 1.0:
                    self.inference_fps = inference_frame_count / elapsed
                    inference_frame_count = 0
                    inference_start_time = time.perf_counter()
                
            except Empty:
                # 队列空，直接重试，不要打印任何东西
                continue
            except Exception as e:
                print(f"推理线程错误: {type(e).__name__}: {e}")
                continue
        
        print("✅ 推理线程已退出")

                
    def create_widgets(self):
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

        # 左侧数据面板
        data_frame = ttk.LabelFrame(main_frame, text="按压数据", padding="10")
        data_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        data_frame.rowconfigure(14, weight=1)  # 让图表行可以扩展

        # 得分显示
        score_frame = ttk.Frame(data_frame)
        score_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)

        self.score_label = ttk.Label(score_frame, text="得分: 0", font=("Arial", 16, "bold"))
        self.score_label.grid(row=0, column=0, sticky=tk.W)

        self.score_percentage = ttk.Label(score_frame, text="0%", font=("Arial", 24, "bold"))
        self.score_percentage.grid(row=0, column=1, sticky=tk.E, padx=(10, 0))
        score_frame.columnconfigure(1, weight=1)

        # 控制按钮框架
        button_frame = ttk.Frame(data_frame)
        button_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)

        self.start_button = ttk.Button(button_frame, text="开始评估", command=self.toggle_evaluation)
        self.start_button.grid(row=0, column=0, padx=2)

        self.reset_button = ttk.Button(button_frame, text="重置数据", command=self.reset_data)
        self.reset_button.grid(row=0, column=1, padx=2)

        self.manual_calibrate_button = ttk.Button(button_frame, text="手动校准", command=self.start_manual_calibration)
        self.manual_calibrate_button.grid(row=0, column=2, padx=2)

        self.cancel_calibrate_button = ttk.Button(button_frame, text="取消校准", command=self.cancel_calibration)
        self.cancel_calibrate_button.grid(row=0, column=3, padx=2)
        self.cancel_calibrate_button.config(state="disabled")

        # 操作者跟踪状态
        self.tracking_status = ttk.Label(button_frame, text="状态: 等待检测", font=("Arial", 10))
        self.tracking_status.grid(row=0, column=4, padx=10)
        self.upload_button = ttk.Button(button_frame, text="选择视频文件", command=self.upload_video)
        self.upload_button.grid(row=0, column=5, padx=2)
        # 在按钮框架的最后（例如 upload_button 之后）添加
        self.record_button = ttk.Button(button_frame, text="🔴 开始录制", command=self.toggle_recording)
        self.record_button.grid(row=0, column=6, padx=2)
        # 数据标签
        self.frequency_label = ttk.Label(data_frame, text="按压频率: -- 次/分钟", font=("Arial", 14))
        self.frequency_label.grid(row=2, column=0, sticky=tk.W, pady=5)

        self.depth_label = ttk.Label(data_frame, text="当前深度: -- cm", font=("Arial", 14))
        self.depth_label.grid(row=3, column=0, sticky=tk.W, pady=5)

        self.count_label = ttk.Label(data_frame, text="按压次数: 0", font=("Arial", 14))
        self.count_label.grid(row=4, column=0, sticky=tk.W, pady=5)

        # 校准信息显示
        calibration_info_frame = ttk.Frame(data_frame)
        calibration_info_frame.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=5)

        self.calibration_label = ttk.Label(calibration_info_frame, text="像素/厘米: 未标定", font=("Arial", 12))
        self.calibration_label.grid(row=0, column=0, sticky=tk.W)

        self.threshold_label = ttk.Label(calibration_info_frame, text="按压阈值: 20 px (2.0 cm)", font=("Arial", 12))
        self.threshold_label.grid(row=1, column=0, sticky=tk.W)

        self.calibration_instruction = ttk.Label(data_frame, text="", font=("Arial", 10), foreground="blue")
        self.calibration_instruction.grid(row=6, column=0, sticky=tk.W, pady=2)

        # 正确次数统计
        stats_frame = ttk.Frame(data_frame)
        stats_frame.grid(row=7, column=0, sticky=(tk.W, tk.E), pady=5)

        self.correct_depth_label = ttk.Label(stats_frame, text="正确深度: 0/0 (0%)", font=("Arial", 12))
        self.correct_depth_label.grid(row=0, column=0, sticky=tk.W)
        self.correct_recoil_label = ttk.Label(stats_frame, text="正确回弹: 0/0 (0%)", font=("Arial", 12))
        self.correct_recoil_label.grid(row=0, column=1, sticky=tk.W,padx=20)
        self.correct_frequency_label = ttk.Label(stats_frame, text="正确频率: 0/0 (0%)", font=("Arial", 12))
        self.correct_frequency_label.grid(row=1, column=0, sticky=tk.W, pady=2)

        # 上肢角度统计
        self.correct_elbow_label = ttk.Label(stats_frame, text="正确肘角: 0/0 (0%)", font=("Arial", 12))
        self.correct_elbow_label.grid(row=2, column=0, sticky=tk.W, pady=2)

        self.correct_vertical_label = ttk.Label(stats_frame, text="正确垂直: 0/0 (0%)", font=("Arial", 12))
        self.correct_vertical_label.grid(row=3, column=0, sticky=tk.W, pady=2)

        # 评估结果
                
        evaluate_frame = ttk.Frame(data_frame)
        evaluate_frame.grid(row=8, column=0, sticky=(tk.W, tk.E), pady=5)
        evaluate_frame.grid_columnconfigure(0, minsize=260)
        evaluate_frame.grid_columnconfigure(1, uniform='group1', minsize=180)
        evaluate_frame.grid_columnconfigure(2, uniform='group1', minsize=180)
        evaluate_frame.grid_columnconfigure(3, uniform='group1', minsize=180)
        self.frequency_status = ttk.Label(evaluate_frame, text="频率评估: --", font=("Arial", 12))
        self.frequency_status.grid(row=0, column=0, sticky=tk.W, pady=5)
        
        self.depth_status = ttk.Label(evaluate_frame, text="深度评估: --", font=("Arial", 12))
        self.depth_status.grid(row=1, column=0, sticky=tk.W, pady=5)

        # 双臂角度评估 - 紧凑布局
        # 右上肢角度和垂直度放在一行
        right_arm_frame = ttk.Frame(data_frame)
        right_arm_frame.grid(row=9, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.right_elbow_angle_label = ttk.Label(right_arm_frame, text="右肘关节角度: --°", font=("Arial", 12))
        self.right_elbow_angle_label.grid(row=0, column=0, sticky=tk.W, padx=(0, 20))
        
        self.right_vertical_angle_label = ttk.Label(right_arm_frame, text="右按压垂直角度: --°", font=("Arial", 12))
        self.right_vertical_angle_label.grid(row=0, column=1, sticky=tk.W)
        
        # 左上肢角度和垂直度放在一行
        left_arm_frame = ttk.Frame(data_frame)
        left_arm_frame.grid(row=10, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.left_elbow_angle_label = ttk.Label(left_arm_frame, text="左肘关节角度: --°", font=("Arial", 12))
        self.left_elbow_angle_label.grid(row=0, column=0, sticky=tk.W, padx=(0, 20))
        
        self.left_vertical_angle_label = ttk.Label(left_arm_frame, text="左按压垂直角度: --°", font=("Arial", 12))
        self.left_vertical_angle_label.grid(row=0, column=1, sticky=tk.W)

        # 按压周期信息显示

        self.press_start_label = ttk.Label(evaluate_frame, text="起始深度: -- cm", font=("Arial", 12))
        self.press_start_label.grid(row=1, column=1, sticky=tk.W)
        
        self.press_end_label = ttk.Label(evaluate_frame, text="最低深度: -- cm", font=("Arial", 12))
        self.press_end_label.grid(row=1, column=2, sticky=tk.W)
        
        self.press_diff_label = ttk.Label(evaluate_frame, text="按压深度差: -- cm", font=("Arial", 12))
        self.press_diff_label.grid(row=1, column=3, sticky=tk.W)
        
        # 深度图表
        self.setup_depth_chart(data_frame)

        # 右侧视频面板
        video_frame = ttk.LabelFrame(main_frame, text="实时视频 - 双臂显示", padding="10")
        video_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        video_frame.columnconfigure(0, weight=1)
        video_frame.rowconfigure(0, weight=1)

        # 视频显示
        self.video_canvas = tk.Canvas(video_frame, background="black", highlightthickness=0)
        self.video_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 绑定鼠标点击事件
        self.video_canvas.bind("<Button-1>", self.on_canvas_click)

        # 用于显示图像的标签
        self.video_label = ttk.Label(video_frame, background="black")
        self.video_label.grid(row=0, column=0)
        self.video_label.lower()

    def setup_depth_chart(self, parent):
        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title('实时按压深度变化')
        self.ax.set_xlabel('时间 (秒)')
        self.ax.set_ylabel('深度 (cm)')
        self.ax.set_ylim(0, 7)
        self.ax.grid(True)

        self.ax.axhspan(5, 6, alpha=0.3, color='green', label='目标深度范围')

        self.depth_line, = self.ax.plot([], [], 'b-o', linewidth=2, label='实时深度')
        self.ax.legend()

        # 嵌入图表到Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.get_tk_widget().grid(row=12, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)

    def animate_chart(self, frame):
        if len(self.time_data) > 1 and len(self.depth_data) > 1:
            self.depth_line.set_data(list(self.time_data), list(self.depth_data))
            current_time = self.time_data[-1]
            self.ax.set_xlim(max(0, current_time - 10), max(10, current_time))
        return self.depth_line,

    def frame_capture_worker(self):
        """统一帧捕获线程：按真实帧率控制读取速度"""
        while not self.stop_capture_event.is_set():
            if self.cap is None or not self.cap.isOpened():
                time.sleep(0.01)
                continue

            # 记录开始时间，用于帧率控制
            t_start = time.perf_counter()
            ret, frame = self.cap.read()
            if not ret or frame is None:
                # 文件播放完毕自动停止评估
                if hasattr(self, 'is_file_video') and self.is_file_video:
                    self.root.after(0, lambda: self.toggle_evaluation() if self.is_running else None)
                break

            # 更新共享帧
            with self.frame_lock:
                self.latest_frame = frame

            # 按需写入录像
            if self.is_recording and self.out is not None and self.out.isOpened():
                self.out.write(frame)

            # 送给推理线程
            if self.inference_running and not self.frame_queue.full():
                self.frame_queue.put(frame)

            # 帧率控制（摄像头和文件统一）
            if self.real_camera_fps and self.real_camera_fps > 0:
                frame_time = 1.0 / self.real_camera_fps
                elapsed = time.perf_counter() - t_start
                sleep_time = frame_time - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
        print("帧捕获线程退出")

    def init_camera(self):
        # 尝试打开摄像头
        self.cap = cv2.VideoCapture(1)
        if not self.cap.isOpened():
            for i in range(0, 5):
                self.cap = cv2.VideoCapture(i)
                if self.cap.isOpened():
                    break

        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            # ====================== 新增：获取真实帧率 ======================
            self.cap.set(cv2.CAP_PROP_FPS, 60)

            self.real_camera_fps = self.cap.get(cv2.CAP_PROP_FPS)
            #if self.real_camera_fps <= 0:
            #    self.real_camera_fps = 30.0
            self.save_fps = self.real_camera_fps
            self.frame_interval_ms = self.min_loop_interval
            # ===============================================================
            self.is_file_video = False  # 🔥 摄像头模式，允许录制
            self.stop_capture_event.clear()
            self.capture_thread = threading.Thread(
                target=self.frame_capture_worker, daemon=True
            )
            self.capture_thread.start()
        else:
            self.video_label.config(text="无法访问摄像头", foreground="red")
            self.video_canvas.create_text(320, 240, text="无法访问摄像头", fill="red", font=("Arial", 16))
        
    def upload_video(self):
        # 1. 停止当前帧捕获线程
        self.stop_capture_event.set()
        if self.capture_thread is not None and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        self.stop_capture_event.clear()

        # 2. 如果正在录制，先停止录制并释放 writer
        if self.is_recording:
            self.is_recording = False
            if self.out is not None:
                self.out.release()
                self.out = None
            self.record_button.config(text="🔴 开始录制")

        # 3. 释放旧的 cap（摄像头）
        if self.cap is not None:
            self.cap.release()

        # 4. 选择文件
        file_path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv *.flv"), ("所有文件", "*.*")]
        )
        if not file_path:
            # 用户取消，重新打开摄像头
            self.init_camera()
            return

        # 5. 打开视频文件
        self.cap = cv2.VideoCapture(file_path)
        if not self.cap.isOpened():
            messagebox.showerror("错误", "无法打开视频文件")
            self.init_camera()
            return

        file_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.real_camera_fps = file_fps if file_fps > 0 else 30.0
        self.save_fps = self.real_camera_fps
        self.frame_interval_ms = self.min_loop_interval
        self.is_file_video = True
        self.video_file_path = file_path

        # 6. 重启帧捕获线程（新 cap 对象）
        self.capture_thread = threading.Thread(
            target=self.frame_capture_worker, daemon=True
        )
        self.capture_thread.start()

        # 7. 重置评估数据
        self.reset_data()

        self.video_label.config(text="视频加载成功，点击开始评估", foreground="green")
        print(f"✅ 已加载视频: {file_path}，帧率: {self.real_camera_fps:.1f} FPS")

    def toggle_recording(self):
        if self.is_recording:
            # 停止录制
            self.is_recording = False
            self.record_button.config(text="🔴 开始录制")
            if self.out is not None:
                self.out.release()
                self.out = None
                print("🛑 录像已停止并保存")
        else:
            # 开始录制
            if self.cap is None or not self.cap.isOpened():
                messagebox.showwarning("警告", "没有可用的视频源，请先打开摄像头或加载视频。")
                return

            # 如果是文件视频，通常不需要录制，给出提示
            if hasattr(self, 'is_file_video') and self.is_file_video:
                if not messagebox.askyesno("提示", "当前是文件视频，确定要重新录制该视频吗？"):
                    return

            # 初始化录像 writer（如果尚未创建）
            if self.out is None:
                self.init_video_writer()
            if self.out is not None:
                self.is_recording = True
                self.record_button.config(text="⏹️ 停止录制")
                print("🔴 录像已开始")
            else:
                messagebox.showerror("错误", "无法创建视频文件")

    def init_video_writer(self):
        """初始化视频保存器（全程录制）"""
        if self.out is not None:
            return
        # 🔥 只有摄像头才保存视频，上传视频不保存
        # 因为上传的视频是文件，我们不再重复录制
        if hasattr(self, 'is_file_video') and self.is_file_video:
            return
        if self.cap is not None and self.cap.isOpened() and self.save_video:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            
            filename = f"全程录像_{time.strftime('%Y%m%d_%H%M%S')}.avi"
            
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            #fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.out = cv2.VideoWriter(filename, fourcc, self.save_fps, (width, height))
            print(f"✅ 全程录像已开始：{filename}，帧率：{self.save_fps}")

    def toggle_evaluation(self):
        # 根据视频源设置显示帧率
        #if hasattr(self, 'is_file_video') and self.is_file_video:
        #    self.min_loop_interval = 16   # 文件视频用 30 fps
        #else:
        #    self.min_loop_interval = 16   # 摄像头用 60 fps（或实际能达到的）

        if self.is_running:
            # ==================== 停止评估 ====================
            self.is_running = False
            self.start_button.config(text="开始评估")
            self.min_y = None
            self.max_y = None
            self.operator_id = None
            self.prev_operator_landmarks = None
            self.tracking_status.config(text="状态: 已停止", foreground="gray")
            self.is_calibrating_video = False

            # 停止推理线程
            self.inference_running = False
            if self.inference_thread is not None:
                self.inference_thread.join(timeout=2.0)
                self.inference_thread = None

            # 清空队列和缓存
            while not self.frame_queue.empty():
                self.frame_queue.get()
            while not self.result_queue.empty():
                self.result_queue.get()
            self.last_inference_result = None

            # 打印数据列表
            self.print_compression_data()
            self.show_analysis_report()

        else:
            # ==================== 开始评估 ====================
            # 检查是否已标定
            if self.pixel_per_cm is None:
                self.calibration_instruction.config(
                    text="请先完成手动校准！",
                    foreground="red"
                )
                return

            self.is_running = True
            self.start_button.config(text="停止评估")

            # 重置帧率统计
            self.fps_start_time = time.perf_counter()
            self.fps_frame_count = 0
            self.display_fps = 0.0
            self.inference_fps = 0.0


            # 启动推理线程
            self.inference_running = True
            self.inference_thread = threading.Thread(
                target=self.inference_worker, daemon=True
            )
            self.inference_thread.start()

            # 启动视频显示更新循环
            self.update_video()

    def reset_data(self):
        self.compression_count = 0
        self.correct_depth_count = 0
        self.correct_frequency_count = 0
        self.correct_elbow_count = 0
        self.correct_vertical_count = 0
        self.compression_times.clear()
        self.compression_depths.clear()
        self.time_data.clear()
        self.depth_data.clear()
        self.current_depth = 0
        self.current_frequency = 0
        self.right_elbow_angle = 0
        self.left_elbow_angle = 0
        self.right_press_vertical_angle = 0
        self.left_press_vertical_angle = 0
        self.correct_recoil_count = 0
        self.score = 0
        self.reference_y = None
        self.min_y = None
        self.max_y = None
        self.cycle_highest_y = None   # 一个周期的最高点（回弹）
        self.cycle_lowest_y = None    # 一个周期的最低点（按压深度）
        self.is_pressing = False
        self.last_wrist_y = None
        self.press_max_depth = 0
        self.press_min_y = None
        # 保留标定结果
        self.depth_threshold = self.depth_threshold_cm * self.pixel_per_cm if self.pixel_per_cm else 20
        self.chart_start_time = time.time()
        self.operator_id = None
        self.prev_operator_landmarks = None
        self.tracking_status.config(text="状态: 等待检测", foreground="gray")
        
        # 重置按压周期数据
        self.press_start_y = None
        self.press_start_depth = 0
        self.start_depth_list.clear()  # 清空起始深度列表
        self.min_depth_list.clear()    # 清空最低深度列表
        
        # 重置帧率统计
        self.fps_start_time = time.perf_counter()
        self.fps_frame_count = 0
        self.current_fps = 0.0
        
        self.update_display()

    def print_compression_data(self):
        """打印所有按压的起始深度和最低深度列表，并统计平均值"""
        print("\n" + "="*50)
        print("按压评估数据汇总 - 胸外按压专项分析")
        print("="*50)
        print(f"总按压次数: {len(self.start_depth_list)}")
        print(f"\n起始点深度列表 (cm) [反映回弹情况，越接近0越好]:")
        print([f"{depth:.2f}" for depth in self.start_depth_list])
        print(f"\n按压深度列表 (cm) [最低点深度，目标5-6cm]:")
        print([f"{depth:.2f}" for depth in self.min_depth_list])
        
        # 计算并打印统计信息
        if self.start_depth_list and self.min_depth_list:
            # 计算平均起始点深度和平均按压深度
            avg_start_depth = np.mean(self.start_depth_list)
            avg_press_depth = np.mean(self.min_depth_list)
            
            depth_diffs = [min_depth - start_depth for start_depth, min_depth in 
                        zip(self.start_depth_list, self.min_depth_list)]
            print(f"\n按压深度差列表 (cm) [实际按压行程 = 按压深度 - 起始深度]:")
            print([f"{diff:.2f}" for diff in depth_diffs])
            
            print(f"\n======== 核心指标平均值 ========")
            print(f"平均起始点深度: {avg_start_depth:.2f} cm")
            print(f"平均按压深度: {avg_press_depth:.2f} cm")
            print(f"平均实际按压行程: {np.mean(depth_diffs):.2f} cm")
            
            # 回弹情况评估
            if avg_start_depth <= 0.5:
                rebound_evaluation = "优秀 - 回弹完全，符合胸外按压要求"
            elif avg_start_depth <= 1.0:
                rebound_evaluation = "良好 - 回弹基本到位，需注意完全复位"
            else:
                rebound_evaluation = "需改进 - 回弹不足，可能存在按压后未完全放松"
            print(f"回弹情况评估: {rebound_evaluation}")
            
            # 按压深度评估
            if 5.0 <= avg_press_depth <= 6.0:
                press_evaluation = "优秀 - 按压深度符合指南要求"
            elif avg_press_depth < 5.0:
                press_evaluation = "需加深 - 平均按压深度不足，需加大按压力度"
            else:
                press_evaluation = "需减轻 - 平均按压深度过深，避免造成二次伤害"
            print(f"按压深度评估: {press_evaluation}")
            
            std_diff = np.std(self.min_depth_list)
            max_diff = np.max(depth_diffs)
            min_diff = np.min(depth_diffs)
            
            print(f"\n======== 稳定性指标 ========")
            print(f"深度标准差: {std_diff:.2f} cm (越小说明按压越稳定)")
            print(f"最大按压行程: {max_diff:.2f} cm")
            print(f"最小按压行程: {min_diff:.2f} cm")
            
            # 统计符合标准的按压次数
            valid_presses = [diff for diff in self.min_depth_list if 5 <= diff <= 6]
            print(f"\n符合5-6cm标准的按压次数: {len(valid_presses)}/{len(depth_diffs)}")
            print(f"符合率: {len(valid_presses)/len(depth_diffs)*100:.1f}%")
            valid_recoils = [diff for diff in self.start_depth_list if diff <= 0.5]
            print(f"\n符合0-0.5cm标准的回弹次数: {len(valid_recoils)}/{len(self.start_depth_list)}")
            print(f"符合率: {len(valid_recoils)/len(self.start_depth_list)*100:.1f}%")
        print("="*50 + "\n")
    def save_report_to_file(self, report_text):
        """保存评估报告到本地文本文件"""
        
        
        # 默认文件名
        default_name = f"胸外按压评估报告_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        
        # 选择保存路径
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            initialfile=default_name,
            filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")],
            title="保存评估报告"
        )
        
        if not file_path:
            return
        
        # 写入文件
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"✅ 报告已保存：{file_path}")
            
            # 提示保存成功
            
            messagebox.showinfo("保存成功", f"评估报告已保存！\n\n路径：\n{file_path}")
            
        except Exception as e:
            messagebox.showerror("保存失败", f"保存出错：{str(e)}")
    def show_analysis_report(self):
        """显示按压分析报告弹窗"""
        report_window = tk.Toplevel(self.root)
        report_window.title("胸外按压评估分析报告")
        report_window.geometry("750x650")
        report_window.transient(self.root)  # 置顶
        report_window.grab_set()  # 模态窗口

        # 主框架
        main_frame = ttk.Frame(report_window, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 文本框 + 滚动条
        text_frame = ttk.Frame(main_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        text_widget = tk.Text(text_frame, wrap=tk.WORD, font=("Consolas", 11), 
                            bg="#f8f9fa", fg="#212529", 
                            yscrollcommand=scrollbar.set)
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=text_widget.yview)

        # 生成报告文本（完全和你控制台输出一样）
        report_text = "="*50 + "\n"
        report_text += "按压评估数据汇总 - 胸外按压专项分析\n"
        report_text += "="*50 + "\n"
        report_text += f"总按压次数: {len(self.start_depth_list)}\n\n"

        report_text += f"起始点深度列表 (cm) [反映回弹情况，越接近0越好]:\n"
        report_text += str([f"{depth:.2f}" for depth in self.start_depth_list]) + "\n\n"

        report_text += f"按压深度列表 (cm) [最低点深度，目标5-6cm]:\n"
        report_text += str([f"{depth:.2f}" for depth in self.min_depth_list]) + "\n\n"

        # 深度差
        depth_diffs = [min_d - start_d for start_d, min_d in zip(self.start_depth_list, self.min_depth_list)]
        report_text += f"按压深度差列表 (cm) [实际按压行程 = 按压深度 - 起始深度]:\n"
        report_text += str([f"{diff:.2f}" for diff in depth_diffs]) + "\n\n"

        # 平均值
        if self.start_depth_list and self.min_depth_list:
            avg_start = np.mean(self.start_depth_list)
            avg_press = np.mean(self.min_depth_list)
            avg_diff = np.mean(depth_diffs)

            report_text += f"======== 核心指标平均值 ========\n"
            report_text += f"平均起始点深度: {avg_start:.2f} cm\n"
            report_text += f"平均按压深度: {avg_press:.2f} cm\n"
            report_text += f"平均实际按压行程: {avg_diff:.2f} cm\n\n"

            # 回弹评估
            if avg_start <= 0.5:
                rebound_eval = "优秀 - 回弹完全，符合胸外按压要求"
            elif avg_start <= 1.0:
                rebound_eval = "良好 - 回弹基本到位，需注意完全复位"
            else:
                rebound_eval = "需改进 - 回弹不足，可能存在按压后未完全放松"
            report_text += f"回弹情况评估: {rebound_eval}\n"

            # 深度评估
            if 5.0 <= avg_press <= 6.0:
                press_eval = "优秀 - 按压深度符合指南要求"
            elif avg_press < 5.0:
                press_eval = "需加深 - 平均按压深度不足，需加大按压力度"
            else:
                press_eval = "需减轻 - 平均按压深度过深，避免造成二次伤害"
            report_text += f"按压深度评估: {press_eval}\n\n"

            # 稳定性
            std_diff = np.std(depth_diffs)
            max_diff = np.max(depth_diffs)
            min_diff = np.min(depth_diffs)
            report_text += f"======== 稳定性指标 ========\n"
            report_text += f"深度标准差: {std_diff:.2f} cm (越小说明按压越稳定)\n"
            report_text += f"最大按压行程: {max_diff:.2f} cm\n"
            report_text += f"最小按压行程: {min_diff:.2f} cm\n\n"

            # 符合率
            valid_presses = [d for d in self.min_depth_list if 5 <= d <= 6]
            valid_rate = len(valid_presses) / len(self.min_depth_list) * 100
            report_text += f"符合5-6cm标准的按压次数: {len(valid_presses)}/{len(self.min_depth_list)}\n"
            report_text += f"符合率: {valid_rate:.1f}%\n\n"

            valid_recoils = [d for d in self.start_depth_list if d <= 0.5]
            recoil_rate = len(valid_recoils) / len(self.start_depth_list) * 100
            report_text += f"符合0-0.5cm标准的回弹次数: {len(valid_recoils)}/{len(self.start_depth_list)}\n"
            report_text += f"符合率: {recoil_rate:.1f}%\n"

        report_text += "="*50 + "\n"

        # 插入文本
        text_widget.insert("1.0", report_text)
        text_widget.config(state=tk.DISABLED)  # 只读

        # 关闭按钮
# 按钮框架（关闭 + 保存）
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="保存报告", command=lambda: self.save_report_to_file(report_text), width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="关闭", command=report_window.destroy, width=15).pack(side=tk.LEFT, padx=5)
    def start_manual_calibration(self):
        self.calibrating = True
        self.calibration_points = []
        self.calibration_instruction.config(
            text="请在视频中点击标志物的两端（16cm），点击第一点后等待第二点",
            foreground="blue"
        )
        self.manual_calibrate_button.config(state="disabled")
        self.cancel_calibrate_button.config(state="normal")
        
        # 标定时强制启动视频显示
        if not self.is_running and self.cap and self.cap.isOpened():
            # 临时启动视频更新循环，仅用于标定
            self.is_calibrating_video = True
            self.update_video_for_calibration()

    def cancel_calibration(self):
        self.calibrating = False
        self.calibration_points = []
        self.calibration_instruction.config(text="", foreground="blue")
        self.manual_calibrate_button.config(state="normal")
        self.cancel_calibrate_button.config(state="disabled")
        self.video_canvas.delete("calibration")
        
        # 取消标定时停止临时视频循环
        self.is_calibrating_video = False

    def update_depth_threshold(self):
        """修复：正确计算深度阈值"""
        if self.pixel_per_cm is not None:
            self.depth_threshold = self.depth_threshold_cm * self.pixel_per_cm
        else:
            self.depth_threshold = 20  # 默认值

    def on_canvas_click(self, event):
        if not self.calibrating:
            return

        if self.current_frame is None or self.original_frame_size is None:
            return

        canvas_x = event.x
        canvas_y = event.y
        canvas_width = self.video_canvas.winfo_width()
        canvas_height = self.video_canvas.winfo_height()

        orig_h, orig_w = self.original_frame_size
        scale_w = canvas_width / orig_w
        scale_h = canvas_height / orig_h
        scale = min(scale_w, scale_h)

        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        offset_x = (canvas_width - new_w) // 2
        offset_y = (canvas_height - new_h) // 2

        if (offset_x <= canvas_x <= offset_x + new_w and
                offset_y <= canvas_y <= offset_y + new_h):

            display_x = canvas_x - offset_x
            display_y = canvas_y - offset_y
            orig_x = int(display_x / scale)
            orig_y = int(display_y / scale)

            self.calibration_points.append((orig_x, orig_y))
            radius = 5
            self.video_canvas.create_oval(
                canvas_x - radius, canvas_y - radius,
                canvas_x + radius, canvas_y + radius,
                fill="red", outline="yellow", width=2,
                tags="calibration"
            )

            self.video_canvas.create_text(
                canvas_x, canvas_y - 15,
                text=str(len(self.calibration_points)),
                fill="yellow", font=("Arial", 12, "bold"),
                tags="calibration"
            )

            if len(self.calibration_points) == 2:
                x1, y1 = self.calibration_points[0]
                x2, y2 = self.calibration_points[1]
                
                # 修复：正确计算像素距离
                pixel_distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                
                # 核心修复：标定计算
                self.pixel_per_cm = pixel_distance / self.actual_length_cm  # 多少像素对应1厘米
                self.cm_per_pixel = self.actual_length_cm / pixel_distance  # 多少厘米对应1像素
                
                self.update_depth_threshold()

                self.calibration_label.config(
                    text=f"像素/厘米: {self.pixel_per_cm:.2f} | 厘米/像素: {self.cm_per_pixel:.4f}"
                )
                self.threshold_label.config(
                    text=f"按压阈值: {self.depth_threshold:.1f} px ({self.depth_threshold_cm} cm)"
                )

                self.calibration_instruction.config(
                    text=f"已校准！{pixel_distance:.1f}px = {self.actual_length_cm}cm, {self.pixel_per_cm:.2f}px/cm",
                    foreground="green"
                )

                canvas_x1 = offset_x + int(x1 * scale)
                canvas_y1 = offset_y + int(y1 * scale)
                canvas_x2 = offset_x + int(x2 * scale)
                canvas_y2 = offset_y + int(y2 * scale)

                self.video_canvas.create_line(
                    canvas_x1, canvas_y1, canvas_x2, canvas_y2,
                    fill="yellow", width=2, tags="calibration"
                )

                mid_x = (canvas_x1 + canvas_x2) // 2
                mid_y = (canvas_y1 + canvas_y2) // 2
                self.video_canvas.create_text(
                    mid_x, mid_y,
                    text=f"{self.actual_length_cm}cm",
                    fill="yellow", font=("Arial", 10, "bold"),
                    tags="calibration"
                )

                self.calibrating = False
                self.manual_calibrate_button.config(state="normal")
                self.cancel_calibrate_button.config(state="disabled")
                self.root.after(3000, lambda: self.video_canvas.delete("calibration"))
            else:
                self.calibration_instruction.config(
                    text=f"已点击第{len(self.calibration_points)}个点，请点击第二个点",
                    foreground="blue"
                )

    def get_both_arms_landmarks(self, keypoints):
        """YOLO 关键点：获取双臂关键点"""
        if keypoints is None or len(keypoints) == 0:
            return None, None, None, None, None, None
        
        # YOLO Pose 关键点索引
        # 右肩:6, 右肘:8, 右腕:10
        # 左肩:5, 左肘:7, 左腕:9
        try:
            right_shoulder = keypoints[6]
            right_elbow = keypoints[8]
            right_wrist = keypoints[10]
            
            left_shoulder = keypoints[5]
            left_elbow = keypoints[7]
            left_wrist = keypoints[9]
            
            return right_shoulder, right_elbow, right_wrist, left_shoulder, left_elbow, left_wrist
        except:
            return None, None, None, None, None, None

    def calculate_elbow_angle(self, shoulder, elbow, wrist):
        """计算肘关节角度"""
        if shoulder is None or elbow is None or wrist is None:
            return None

        # YOLO关键点已经是像素坐标
        shoulder_x, shoulder_y = shoulder
        elbow_x, elbow_y = elbow
        wrist_x, wrist_y = wrist

        upper_arm_vec = np.array([shoulder_x - elbow_x, shoulder_y - elbow_y])
        forearm_vec = np.array([wrist_x - elbow_x, wrist_y - elbow_y])

        dot_product = np.dot(upper_arm_vec, forearm_vec)
        norm_upper = np.linalg.norm(upper_arm_vec)
        norm_fore = np.linalg.norm(forearm_vec)

        if norm_upper * norm_fore == 0:
            return None

        cos_angle = dot_product / (norm_upper * norm_fore)
        cos_angle = max(-1.0, min(1.0, cos_angle))

        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)

        return angle_deg

    def calculate_press_vertical_angle(self, shoulder, wrist):
        """计算按压垂直角度"""
        if shoulder is None or wrist is None:
            return None

        shoulder_x, shoulder_y = shoulder
        wrist_x, wrist_y = wrist

        press_vec = np.array([wrist_x - shoulder_x, wrist_y - shoulder_y])
        vertical_vec = np.array([0, 1])

        dot_product = np.dot(press_vec, vertical_vec)
        norm_press = np.linalg.norm(press_vec)
        norm_vertical = np.linalg.norm(vertical_vec)

        if norm_press * norm_vertical == 0:
            return None

        cos_angle = dot_product / (norm_press * norm_vertical)
        cos_angle = max(-1.0, min(1.0, cos_angle))

        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)

        return angle_deg

    def calculate_movement_score(self, current_kpts, prev_kpts):
        """计算运动活跃度得分"""
        if prev_kpts is None or current_kpts is None:
            return 0.0
        
        try:
            # 使用右腕计算运动
            curr_wrist = current_kpts[10]
            prev_wrist = prev_kpts[10]
            
            distance = np.sqrt((curr_wrist[0] - prev_wrist[0])**2 + (curr_wrist[1] - prev_wrist[1])**2)
            normalized_distance = min(distance / 100.0, 1.0)
            
            return normalized_distance
        except:
            return 0.0

    def identify_operator_by_motion(self, all_keypoints):
        """通过运动活跃度识别操作者"""
        if not all_keypoints or len(all_keypoints) == 0:
            return None
        
        if len(all_keypoints) == 1:
            return all_keypoints[0]
        
        best_operator = None
        best_movement_score = 0
        
        for i, keypoints in enumerate(all_keypoints):
            movement_score = 0
            
            if self.prev_operator_landmarks is not None and i == self.operator_id:
                movement_score = self.calculate_movement_score(keypoints, self.prev_operator_landmarks)
            
            if self.prev_operator_landmarks is None:
                try:
                    shoulder_y = keypoints[6][1]
                    if best_operator is None or shoulder_y < best_operator[6][1]:
                        best_operator = keypoints
                        self.operator_id = i
                except:
                    pass
            
            if movement_score > best_movement_score:
                best_movement_score = movement_score
                best_operator = keypoints
                self.operator_id = i
        
        if best_operator is None and len(all_keypoints) > 0:
            best_operator = all_keypoints[0]
            self.operator_id = 0
        
        return best_operator

    def detect_compression(self, wrist_y, frame_height):
        """
        最终正确逻辑：
        1. 周期最高点 = 回弹位置（起始深度）
        2. 周期最低点 = 按压深度（直接记录最低点）
        3. 深度差 = 最低点 - 最高点
        """
        if wrist_y is None or self.cm_per_pixel is None:
            return False

        current_y = wrist_y
        # 实时深度 = 相对于【程序初始基准】的深度（保持界面显示不变）
        if self.reference_y is None:
            self.reference_y = wrist_y
        # 初始化周期极值
        if self.cycle_highest_y is None:
            self.cycle_highest_y = current_y
            self.cycle_lowest_y = current_y
            self.last_wrist_y = current_y
            return False

        # ==============================
        # 捕捉：本周期 最高点 + 最低点
        # ==============================
        self.cycle_highest_y = min(self.cycle_highest_y, current_y)  # 最高点（回弹）
        self.cycle_lowest_y = max(self.cycle_lowest_y, current_y)    # 最低点（按压深度）


            
        current_depth = (current_y - self.reference_y) * self.cm_per_pixel
        #self.current_depth = max(0.0, current_depth)

        # ==============================
        # 按压状态机
        # ==============================
        if not self.is_pressing:
            if current_depth > self.depth_threshold_cm:
                self.is_pressing = True
                self.press_max_depth = current_depth   # 初始化最大值
        else:
            if current_depth > self.press_max_depth:
                self.press_max_depth = current_depth
            if current_depth < self.depth_threshold_cm * 0.8:  # 回弹判断也用局部变量
                self.is_pressing = False

                # ============================
                # ✅ 【完全按你要求赋值】
                # 起始深度 = 周期最高点（真正回弹）
                # 按压深度 = 周期最低点（直接记录）
                # ============================
                self.press_start_depth = (self.cycle_highest_y - self.reference_y) * self.cm_per_pixel
                self.press_max_depth = (self.cycle_lowest_y - self.reference_y) * self.cm_per_pixel

                # 重置下一个周期
                self.cycle_highest_y = current_y
                self.cycle_lowest_y = current_y

                return True

        self.last_wrist_y = current_y
        return False

    def calculate_depth_from_wrist(self, wrist_y, frame_height):
        """
        保持不变：深度 = 当前位置 - 全局基准
        因为你要的按压深度 = 最低点（直接位置）
        """
        if wrist_y is None or self.reference_y is None or self.cm_per_pixel is None:
            return 0

        depth_cm = (wrist_y - self.reference_y) * self.cm_per_pixel
        return max(0.0, depth_cm)

    def update_video_for_calibration(self):
        """专门用于标定的视频更新循环，统一使用共享帧 latest_frame"""
        if not self.is_calibrating_video or not self.cap or not self.cap.isOpened():
            return

        # ✅ 永远从共享帧获取，不做任何 cap.read() 操作
        with self.frame_lock:
            if self.latest_frame is not None:
                frame = self.latest_frame.copy()
            else:
                frame = None

        if frame is None:
            self.root.after(self.frame_interval_ms, self.update_video_for_calibration)
            return

        try:
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            self.current_frame = frame.copy()
            h, w = frame.shape[:2]
            self.original_frame_size = (h, w)

            # 只显示视频，不执行评估逻辑
            self.display_frame_in_tkinter(frame)
        except Exception as e:
            print(f"标定视频显示错误: {e}")
            self.video_canvas.delete("all")
            self.video_canvas.create_text(320, 240, text="视频显示错误", fill="red", font=("Arial", 12))

        # 持续更新
        self.root.after(self.frame_interval_ms, self.update_video_for_calibration)
    def update_video(self):
        if self.is_running and self.cap and self.cap.isOpened():
            try:
                # 🔥 第一步：获取最新原始帧（永远60fps，只在这里复制一次）
                with self.frame_lock:
                    if self.latest_frame is not None:
                        display_frame = self.latest_frame.copy()
                    else:
                        display_frame = None
                
                if display_frame is None:
                    self.root.after(self.frame_interval_ms, self.update_video)
                    return
                    
                h, w = display_frame.shape[:2]
                # 1. 获取绘制用的最新关键点（无锁竞争）
                with self.inference_lock:
                    draw_landmarks = self.last_inference_result

                # 2. 检查推理结果队列，执行评估（不依赖于绘制）
                has_new_result = False
                # 🔥 第二步：检查新结果，只在有新结果时执行评估逻辑
                if not self.result_queue.empty():
                    draw_landmarks, h, w = self.result_queue.get_nowait()
                    has_new_result = True
                    
                    # 🔥 所有评估逻辑只在有新结果时执行（不做任何绘制）
                    wrist_y = None
                    if draw_landmarks is not None:
                        if self.operator_id is not None:
                            self.tracking_status.config(text=f"状态: 跟踪操作者 {self.operator_id+1}", foreground="green")
                        else:
                            self.tracking_status.config(text="状态: 跟踪中", foreground="green")
                        
                        # 获取手腕位置（右腕）
                        try:
                            right_wrist = draw_landmarks[10]
                            wrist_y = right_wrist[1]
                        except:
                            pass
                        
                        # 获取双臂所有关键点
                        (right_shoulder, right_elbow, right_wrist,
                        left_shoulder, left_elbow, left_wrist) = self.get_both_arms_landmarks(draw_landmarks)
                        
                        # 计算右臂角度
                        if right_shoulder is not None and right_elbow is not None and right_wrist is not None:
                            self.right_elbow_angle = self.calculate_elbow_angle(right_shoulder, right_elbow, right_wrist)
                            if self.right_elbow_angle is not None:
                                elbow_deviation = abs(180 - self.right_elbow_angle)
                                if elbow_deviation <= self.elbow_angle_threshold:
                                    if self.is_pressing and not hasattr(self, 'elbow_checked_this_cycle'):
                                        self.correct_elbow_count += 1
                                        self.elbow_checked_this_cycle = True
                        
                        if right_shoulder is not None and right_wrist is not None:
                            self.right_press_vertical_angle = self.calculate_press_vertical_angle(right_shoulder, right_wrist)
                            if self.right_press_vertical_angle is not None:
                                if self.right_press_vertical_angle <= self.vertical_angle_threshold:
                                    if self.is_pressing and not hasattr(self, 'vertical_checked_this_cycle'):
                                        self.correct_vertical_count += 1
                                        self.vertical_checked_this_cycle = True
                        
                        # 计算左臂角度
                        if left_shoulder is not None and left_elbow is not None and left_wrist is not None:
                            self.left_elbow_angle = self.calculate_elbow_angle(left_shoulder, left_elbow, left_wrist)
                        
                        if left_shoulder is not None and left_wrist is not None:
                            self.left_press_vertical_angle = self.calculate_press_vertical_angle(left_shoulder, left_wrist)
                        
                        if wrist_y is not None:
                            current_depth = self.calculate_depth_from_wrist(wrist_y, h)
                            
                            if self.detect_compression(wrist_y, h):
                                if hasattr(self, 'elbow_checked_this_cycle'):
                                    delattr(self, 'elbow_checked_this_cycle')
                                if hasattr(self, 'vertical_checked_this_cycle'):
                                    delattr(self, 'vertical_checked_this_cycle')
                                
                                self.compression_count += 1
                                current_time = time.time()
                                self.compression_times.append(current_time)
                                self.compression_depths.append(self.press_max_depth)
                                
                                if 5 <= self.press_max_depth <= 6:
                                    self.correct_depth_count += 1
                                if self.press_start_depth <= 0.5:
                                    self.correct_recoil_count += 1                            
                                if len(self.compression_times) >= 2:
                                    recent_time_diff = self.compression_times[-1] - self.compression_times[-2]
                                    if recent_time_diff > 0:
                                        instantaneous_frequency = 60.0 / recent_time_diff
                                        if self.current_frequency == 0:
                                            self.current_frequency = instantaneous_frequency
                                        else:
                                            self.current_frequency = self.current_frequency * 0.1 + instantaneous_frequency * 0.9
                                        
                                        if 100 <= instantaneous_frequency <= 120:
                                            self.correct_frequency_count += 1
                                
                                self.calculate_score()
                                
                                # 将起始深度和最低深度添加到列表
                                self.start_depth_list.append(self.press_start_depth)
                                self.min_depth_list.append(self.press_max_depth)
                            
                            # 更新当前深度
                            self.current_depth = current_depth
                        else:
                            self.current_depth = 0
                        
                        # 保存当前操作者landmarks
                        self.prev_operator_landmarks = draw_landmarks
                        
                        # 更新图表数据
                        current_time = time.time() - self.chart_start_time
                        self.time_data.append(current_time)
                        self.depth_data.append(self.current_depth)
                
                # 🔥 第三步：统一绘制（无论有没有新结果，都用缓存的结果绘制）
                if self.last_inference_result is not None:
                    draw_landmarks = self.last_inference_result
                    
                    try:
                        # 获取双臂所有关键点
                        (right_shoulder, right_elbow, right_wrist,
                        left_shoulder, left_elbow, left_wrist) = self.get_both_arms_landmarks(draw_landmarks)
                        
                        # 绘制手腕点
                        try:
                            wrist_x = int(right_wrist[0])
                            wrist_y_pixel = int(right_wrist[1])
                            
                            if self.is_pressing:
                                cv2.circle(display_frame, (wrist_x, wrist_y_pixel), 10, (0, 0, 255), -1)
                                cv2.putText(display_frame, f"Pressing: {self.current_depth:.1f}cm",
                                            (wrist_x + 15, wrist_y_pixel),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            else:
                                cv2.circle(display_frame, (wrist_x, wrist_y_pixel), 10, (0, 255, 255), -1)
                                cv2.putText(display_frame, f"Tracking", (wrist_x + 15, wrist_y_pixel),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                            

                        except:
                            pass
                        
                        # 绘制双臂骨架
                        try:
                            # 右臂：绿色
                            rsx, rsy = int(right_shoulder[0]), int(right_shoulder[1])
                            rex, rey = int(right_elbow[0]), int(right_elbow[1])
                            rwx, rwy = int(right_wrist[0]), int(right_wrist[1])
                            
                            cv2.line(display_frame, (rsx, rsy), (rex, rey), (0,255,0), 3)
                            cv2.line(display_frame, (rex, rey), (rwx, rwy), (0,255,0), 3)
                            cv2.circle(display_frame, (rsx, rsy), 8, (0,255,0), -1)
                            cv2.circle(display_frame, (rex, rey), 8, (0,255,0), -1)
                            cv2.circle(display_frame, (rwx, rwy), 8, (0,255,0), -1)
                            
                            if self.right_elbow_angle is not None:
                                cv2.putText(display_frame, f"R: {self.right_elbow_angle:.0f}°",
                                            (rex+10, rey), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                            
                            # 左臂：蓝色
                            lsx, lsy = int(left_shoulder[0]), int(left_shoulder[1])
                            lex, ley = int(left_elbow[0]), int(left_elbow[1])
                            lwx, lwy = int(left_wrist[0]), int(left_wrist[1])
                            
                            cv2.line(display_frame, (lsx, lsy), (lex, ley), (255,0,0), 3)
                            cv2.line(display_frame, (lex, ley), (lwx, lwy), (255,0,0), 3)
                            cv2.circle(display_frame, (lsx, lsy), 8, (255,0,0), -1)
                            cv2.circle(display_frame, (lex, ley), 8, (255,0,0), -1)
                            cv2.circle(display_frame, (lwx, lwy), 8, (255,0,0), -1)
                            
                            if self.left_elbow_angle is not None:
                                cv2.putText(display_frame, f"L: {self.left_elbow_angle:.0f}°",
                                            (lex+10, ley), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                        except:
                            pass
                        
                        # 绘制操作者框
                        try:
                            xs = []
                            ys = []
                            for kp in draw_landmarks:
                                x, y = kp
                                if x > 0 and y > 0:
                                    xs.append(int(x))
                                    ys.append(int(y))
                            
                            if xs and ys:
                                x_min, x_max = min(xs), max(xs)
                                y_min, y_max = min(ys), max(ys)
                                cv2.rectangle(display_frame, (x_min-10, y_min-10), (x_max+10, y_max+10), (0,255,0), 2)
                                cv2.putText(display_frame, "Operator", (x_min, y_min-15), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                        except:
                            pass
                    except:
                        pass
                
                # 🔥 第四步：绘制通用信息
                cv2.putText(display_frame, f"display: {self.display_fps:.1f} FPS",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, f"inference: {self.inference_fps:.1f} FPS",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                cv2.putText(display_frame, f"Tracking: {'Operator' if self.last_inference_result is not None else 'None'}",
                            (w - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                if self.is_pressing:
                    cv2.putText(display_frame, "STATUS: PRESSING",
                                (w - 200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    cv2.putText(display_frame, "STATUS: RELEASING",
                                (w - 200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if self.reference_y is not None and w > 0:
                    ref_line_y = int(self.reference_y)
                    if 0 <= ref_line_y < h:
                        cv2.line(display_frame, (0, ref_line_y), (w, ref_line_y), (255, 0, 0), 2)
                        cv2.putText(display_frame, "Reference", (10, ref_line_y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                if self.calibrating:
                    cv2.putText(display_frame, "CALIBRATION MODE", (w - 200, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(display_frame, f"Click {len(self.calibration_points) + 1} of 2 points",
                                (w - 200, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.putText(display_frame, "Marker length: 16cm", (w - 200, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # 🔥 第五步：显示视频
                self.display_frame_in_tkinter(display_frame)
                
                # 🔥 只在有新结果时更新UI文本
                if has_new_result:
                    self.update_counter = getattr(self, 'update_counter', 0) + 1
                    if self.update_counter % 10 == 0:
                        self.update_display()
                
                # 统计显示帧率
                self.fps_frame_count += 1
                elapsed = time.perf_counter() - self.fps_start_time
                if elapsed >= 1.0:
                    self.display_fps = self.fps_frame_count / elapsed
                    self.fps_frame_count = 0
                    self.fps_start_time = time.perf_counter()
                    
            except Exception as e:
                print(f"视频处理错误: {type(e).__name__}: {e}")
            
            # 🔥 强制60fps调度
            self.root.after(self.frame_interval_ms, self.update_video)

    def calculate_score(self):
        depth_score = 0
        frequency_score = 0
        posture_score = 0
        recoil_score = 0
        if self.compression_count > 0:
            depth_score = (self.correct_depth_count / self.compression_count) * 40
            frequency_score = (self.correct_frequency_count / self.compression_count) * 30
            recoil_score = (self.correct_recoil_count / self.compression_count) * 10
            elbow_correct_rate = self.correct_elbow_count / max(1, self.compression_count)
            vertical_correct_rate = self.correct_vertical_count / max(1, self.compression_count)
            posture_score = ((elbow_correct_rate + vertical_correct_rate) / 2) * 20

        self.score = round(depth_score + frequency_score + posture_score + recoil_score)

    def display_frame_in_tkinter(self, frame):
        try:
            if frame is None or frame.size == 0:
                return

            # 🔥 缓存画布尺寸和缩放比例，只在窗口大小改变时重新计算
            canvas_w = self.video_canvas.winfo_width()
            canvas_h = self.video_canvas.winfo_height()
            
            # 初始化缓存变量
            if not hasattr(self, '_display_cache'):
                self._display_cache = {
                    'last_canvas_size': (0, 0),
                    'scale': 1.0,
                    'offset_x': 0,
                    'offset_y': 0
                }
            
            # 只有窗口大小改变时才重新计算
            if self._display_cache['last_canvas_size'] != (canvas_w, canvas_h):
                h, w = frame.shape[:2]
                scale = min(canvas_w / w, canvas_h / h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                offset_x = (canvas_w - new_w) // 2
                offset_y = (canvas_h - new_h) // 2
                
                self._display_cache.update({
                    'last_canvas_size': (canvas_w, canvas_h),
                    'scale': scale,
                    'new_w': new_w,
                    'new_h': new_h,
                    'offset_x': offset_x,
                    'offset_y': offset_y
                })
            
            cache = self._display_cache
            new_w, new_h = cache['new_w'], cache['new_h']
            
            if new_w > 0 and new_h > 0:
                # 🔥 一步完成：BGR→RGB + 缩放（OpenCV比PIL快3倍）
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                
                # 🔥 直接从numpy数组创建PhotoImage，跳过PIL中间步骤
                self.photo_image = ImageTk.PhotoImage(image=Image.fromarray(frame_resized))
                
                if self.canvas_image_id is None:
                    self.canvas_image_id = self.video_canvas.create_image(
                        cache['offset_x'], cache['offset_y'],
                        image=self.photo_image, anchor=tk.NW
                    )
                else:
                    self.video_canvas.itemconfig(self.canvas_image_id, image=self.photo_image)

        except Exception as e:
            pass

    def update_display(self):
        self.frequency_label.config(text=f"按压频率: {self.current_frequency:.1f} 次/分钟")
        self.depth_label.config(text=f"当前深度: {self.current_depth:.1f} cm")
        self.count_label.config(text=f"按压次数: {self.compression_count}")
        
        # 修复：标定信息显示
        if self.pixel_per_cm is None:
            self.calibration_label.config(text="像素/厘米: 未标定")
        else:
            self.calibration_label.config(text=f"像素/厘米: {self.pixel_per_cm:.2f} | 厘米/像素: {self.cm_per_pixel:.4f}")
            
        self.threshold_label.config(text=f"按压阈值: {self.depth_threshold:.1f} px ({self.depth_threshold_cm} cm)")

        if self.compression_count > 0:
            depth_percentage = (self.correct_depth_count / self.compression_count) * 100
            frequency_percentage = (self.correct_frequency_count / self.compression_count) * 100
            recoil_percentage = (self.correct_recoil_count / self.compression_count) * 100
            elbow_percentage = (self.correct_elbow_count / self.compression_count) * 100
            vertical_percentage = (self.correct_vertical_count / self.compression_count) * 100

            self.correct_depth_label.config(
                text=f"正确深度: {self.correct_depth_count}/{self.compression_count} ({depth_percentage:.1f}%)"
            )
            self.correct_recoil_label.config(
                text=f"正确回弹: {self.correct_recoil_count}/{self.compression_count} ({recoil_percentage:.1f}%)"
            )
            self.correct_frequency_label.config(
                text=f"正确频率: {self.correct_frequency_count}/{self.compression_count} ({frequency_percentage:.1f}%)"
            )
            self.correct_elbow_label.config(
                text=f"正确肘角: {self.correct_elbow_count}/{self.compression_count} ({elbow_percentage:.1f}%)"
            )
            self.correct_vertical_label.config(
                text=f"正确垂直: {self.correct_vertical_count}/{self.compression_count} ({vertical_percentage:.1f}%)"
            )

        if self.current_frequency == 0:
            self.frequency_status.config(text="频率: 等待检测", foreground="gray")
        elif self.current_frequency < 100:
            self.frequency_status.config(text="频率太慢了，请加快按压频率", foreground="red")
        elif self.current_frequency > 120:
            self.frequency_status.config(text="频率太快了，请减慢按压频率", foreground="red")
        else:
            self.frequency_status.config(text="频率优秀", foreground="green")

        if self.current_depth == 0:
            self.depth_status.config(text="深度: 等待检测", foreground="gray")
        elif self.press_max_depth == 0:
            self.depth_status.config(text="深度: 等待按压完成", foreground="gray")
        elif self.press_max_depth < 4:
            self.depth_status.config(text="按压过浅", foreground="red")
        elif self.press_max_depth > 7:
            self.depth_status.config(text="按压过深", foreground="red")
        elif 5 <= self.press_max_depth <= 6:
            self.depth_status.config(text="深度优秀", foreground="green")
        else:
            if self.press_max_depth < 5:
                self.depth_status.config(text="深度稍浅，请稍微加大力度", foreground="orange")
            else:
                self.depth_status.config(text="深度稍深，请稍微减小力度", foreground="orange")

        # 更新右臂角度显示
        if self.right_elbow_angle is None or self.right_elbow_angle == 0:
            self.right_elbow_angle_label.config(text="右肘关节角度: 未检测到", foreground="gray")
        else:
            elbow_deviation = abs(180 - self.right_elbow_angle)
            self.right_elbow_angle_label.config(text=f"右肘关节角度: {self.right_elbow_angle:.1f}°")
            if elbow_deviation <= self.elbow_angle_threshold:
                self.right_elbow_angle_label.config(foreground="green")
            else:
                self.right_elbow_angle_label.config(foreground="red")

        # 更新左臂角度显示
        if self.left_elbow_angle is None or self.left_elbow_angle == 0:
            self.left_elbow_angle_label.config(text="左肘关节角度: 未检测到", foreground="gray")
        else:
            elbow_deviation = abs(180 - self.left_elbow_angle)
            self.left_elbow_angle_label.config(text=f"左肘关节角度: {self.left_elbow_angle:.1f}°")
            if elbow_deviation <= self.elbow_angle_threshold:
                self.left_elbow_angle_label.config(foreground="green")
            else:
                self.left_elbow_angle_label.config(foreground="blue")

        # 更新右臂垂直角度显示
        if self.right_press_vertical_angle is None or self.right_press_vertical_angle == 0:
            self.right_vertical_angle_label.config(text="右按压垂直角度: 未检测到", foreground="gray")
        else:
            self.right_vertical_angle_label.config(text=f"右按压垂直角度: {self.right_press_vertical_angle:.1f}°")
            if self.right_press_vertical_angle <= self.vertical_angle_threshold:
                self.right_vertical_angle_label.config(foreground="green")
            else:
                self.right_vertical_angle_label.config(foreground="red")

        # 更新左臂垂直角度显示
        if self.left_press_vertical_angle is None or self.left_press_vertical_angle == 0:
            self.left_vertical_angle_label.config(text="左按压垂直角度: 未检测到", foreground="gray")
        else:
            self.left_vertical_angle_label.config(text=f"左按压垂直角度: {self.left_press_vertical_angle:.1f}°")
            if self.left_press_vertical_angle <= self.vertical_angle_threshold:
                self.left_vertical_angle_label.config(foreground="green")
            else:
                self.left_vertical_angle_label.config(foreground="blue")

        # 更新按压周期信息显示
        if self.press_start_depth == 0:
            self.press_start_label.config(text="起始深度: 未检测到", foreground="gray")
        else:
            self.press_start_label.config(text=f"起始深度: {self.press_start_depth:.2f} cm", foreground="black")
        
        if self.press_max_depth == 0:
            self.press_end_label.config(text="最低深度: 未检测到", foreground="gray")
        else:
            self.press_end_label.config(text=f"最低深度: {self.press_max_depth:.2f} cm", foreground="black")
        
        press_depth_diff = self.press_max_depth - self.press_start_depth
        if press_depth_diff <= 0:
            self.press_diff_label.config(text="按压深度差: 未检测到", foreground="gray")
        else:
            self.press_diff_label.config(text=f"按压深度差: {press_depth_diff:.2f} cm", 
                                       foreground="green" if 5 <= press_depth_diff <= 6 else "red")
        
        self.score_label.config(text=f"得分: {self.score}")
        self.score_percentage.config(text=f"{self.score}%")

        if self.score >= 90:
            self.score_percentage.config(foreground="green")
        elif self.score >= 70:
            self.score_percentage.config(foreground="orange")
        else:
            self.score_percentage.config(foreground="red")

    def __del__(self):
        # 恢复Windows系统定时器分辨率
        try:
            winmm.timeEndPeriod(1)
        except:
            pass
        
        # 停止推理线程
        self.inference_running = False
        if self.inference_thread is not None:
            try:
                self.inference_thread.join(timeout=1.0)
            except:
                pass
        
        # 关闭程序时自动保存视频
        if self.out is not None:
            self.out.release()
            print("✅ 全程录像已保存完成")
        
        if self.cap and self.cap.isOpened():
            self.cap.release()


if __name__ == "__main__":
    root = tk.Tk()
    app = ChestCompressionEvaluator(root)
    root.mainloop()