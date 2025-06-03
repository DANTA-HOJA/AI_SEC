import base64
import os
import sys
import threading
from io import BytesIO
from pathlib import Path

import cv2
from flask import Flask, render_template
from flask_socketio import SocketIO
from PIL import Image
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QApplication, QFileDialog
from rich.console import Console

pkg_dir = Path(__file__).parents[1] # `dir_depth` to `repo_root`
if (pkg_dir.exists()) and (str(pkg_dir) not in sys.path):
    sys.path.insert(0, str(pkg_dir)) # add path to scan customized package

from modules.app.cp_seg import cellpose_for_sec

# -----------------------------------------------------------------------------/

# Qt → Flask 通訊橋樑
class Communicator(QObject):
    folder_requested = pyqtSignal()
    folder_selected = pyqtSignal(str)

# Flask app 初始化
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode='threading')  # 非 async 模式用 threading 最穩定

# Qt signal
communicator = Communicator()

# 當前選擇的資料夾
selected_folder = {"path": ""}
scaned_files = {"files": []}

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('request_folder')
def handle_request_folder():
    """ 發出 Qt Signal，叫主執行緒打開 QFileDialog
    """
    print("[SocketIO] 前端請求選擇資料夾")
    communicator.folder_requested.emit()

@socketio.on('disconnect_notice')
def handle_disconnect_notice():
    print("[SocketIO] 前端頁面關閉，準備退出 Qt")
    QApplication.quit()

@socketio.on('request_preview')
def handle_request_preview(data):
    folder = selected_folder["path"]
    filename = data.get("filename")
    if folder and filename:
        full_path = os.path.join(folder, filename)
        if os.path.isfile(full_path):
            try:
                with Image.open(full_path) as img:
                    img.thumbnail((512, 512))  # 可調整縮圖大小
                    buffer = BytesIO()
                    img.save(buffer, format="PNG")
                    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
                    socketio.emit("preview_image", {"filename": filename, "image": encoded})
            except Exception as e:
                print(f"[Error] 預覽失敗：{e}")

@socketio.on('start_processing')
def handle_start_processing():
    socketio.start_background_task(process_files)

def process_files():
    """ main process
    """
    console = Console(record=True)
    cellpose_for_sec(scaned_files["files"], "cp_seg.toml",
                     console, socketio)
    socketio.emit('processing_complete')

def start_flask():
    """ Flask 執行緒
    """
    socketio.run(app, debug=False)

def open_folder_dialog():
    """ Qt Signal callback : 開啟 QFileDialog, 顯示資料夾內檔案但只可選資料夾
    """
    dialog = QFileDialog()
    dialog.setFileMode(QFileDialog.Directory) # 只能選資料夾
    dialog.setOption(QFileDialog.ShowDirsOnly, False) # ✅ 顯示資料夾內的檔案
    dialog.setWindowTitle("選擇資料夾")
    if dialog.exec_():
        folder = dialog.selectedFiles()[0]
        communicator.folder_selected.emit(folder)
    else:
        communicator.folder_selected.emit("")

def send_folder_to_client(path):
    """ Qt Signal callback : 收到選擇結果後傳給前端
    """
    print(f"[Qt] 已選擇資料夾：{path}")
    selected_folder["path"] = path
    socketio.start_background_task(socketio.emit, 'folder_selected', {'path': path})
    
    # 掃描 .tif/.tiff 檔案
    scaned_files["files"] = [file for file in sorted(Path(path).glob("*.tif*"))]
    
    socketio.start_background_task(socketio.emit, 'tiff_list', {
        'folder': path,
        'files': [file.name for file in scaned_files["files"]]
    })

def handle_quit():
    print("[Qt] QApplication 正準備退出！")

# 主程式入口
if __name__ == '__main__':
    # 啟動 Flask background thread
    threading.Thread(target=start_flask, daemon=True).start()

    # Qt 主應用與隱藏視窗
    qt_app = QApplication(sys.argv)
    qt_app.setQuitOnLastWindowClosed(False)  # 這行保證沒視窗也不會退出

    # 設定 signal-slot
    communicator.folder_requested.connect(open_folder_dialog)
    communicator.folder_selected.connect(send_folder_to_client)
    qt_app.aboutToQuit.connect(handle_quit)

    sys.exit(qt_app.exec_())  # 正常結束程式