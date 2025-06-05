import base64
import os
import platform
import shutil
import subprocess
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
from rich.traceback import install

pkg_dir = Path(__file__).parents[1] # `dir_depth` to `repo_root`
if (pkg_dir.exists()) and (str(pkg_dir) not in sys.path):
    sys.path.insert(0, str(pkg_dir)) # add path to scan customized package

from modules.app.cp_seg import cellpose_for_sec
from modules.shared.config import load_config

install()
console = Console(record=True)
# -----------------------------------------------------------------------------/

# Qt → Flask 通訊橋樑
class Communicator(QObject):
    request_select_folder = pyqtSignal() # 請求 Qt 打開資料夾選擇器
    send_folder_selected = pyqtSignal(str) # 回傳選擇的資料夾
    request_open_csv_loca = pyqtSignal(str) # 請求 Qt 打開 csv 所在的資料夾

# Flask app 初始化
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode='threading')  # 非 async 模式用 threading 最穩定

# Qt signal
communicator = Communicator()

# 當前選擇的資料夾
csv_loca = {"path": ""}
selected_folder = {"path": ""}
scaned_files = {"files": []}

def image_to_base64(img_path: Path,
                    thumb_size: tuple = None):
    """
    """
    img_base64: str= ""
    img_name: str = ""
    
    try:
        with Image.open(img_path) as img:
            # downsample
            if thumb_size is not None:
                img.thumbnail(thumb_size)
            
            buffer = BytesIO()
            img.save(buffer, format="PNG")
            img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            img_name = img_path.name
    except:
        console.print("[#FF0000] Can't convert image to base64, "
                                f"file: '{img_path}'")
        # broken image instead
        path = Path(__file__).parent.joinpath("icon", "broken-image.png")
        img_base64, _ = image_to_base64(path, thumb_size)
        img_name = "broken-image.png"

    return img_base64, img_name
    # -------------------------------------------------------------------------/


@app.route('/')
def index():
    """
    """
    return render_template('index.html')
    # -------------------------------------------------------------------------/


@socketio.on('disconnect_notice')
def handle_disconnect_notice():
    """
    """
    print("[SocketIO] 前端頁面關閉，準備退出 Qt")
    QApplication.quit()
    # -------------------------------------------------------------------------/


@socketio.on('request_csv_loca')
def handle_request_csv_loca():
    """
    """
    print(f"[SocketIO] 前端請求開啟 csv 檔案位置, file: '{Path(csv_loca['path'])}'")
    communicator.request_open_csv_loca.emit(csv_loca["path"])
    # -------------------------------------------------------------------------/


@socketio.on('request_folder')
def handle_request_folder():
    """ 發出 Qt Signal，叫主執行緒打開 QFileDialog
    """
    print("[SocketIO] 前端請求選擇資料夾")
    communicator.request_select_folder.emit()
    # -------------------------------------------------------------------------/


@socketio.on('request_thumbs')
def handle_request_thumbs(data):
    """ 生成 Cellpose results 縮圖 (8 pngs)
    
    socketio.emit : 'get_thumbs'
    """
    thumb_size = (64, 64)
    img_thumbs = {}
    img_names = {}

    # Cellpose result pngs (8 images)
    orig_filename = Path(data['filename'])
    proc_dir = Path(selected_folder["path"]).joinpath(orig_filename.stem)
    if proc_dir.is_dir():
        # Load config
        config = load_config("cp_seg.toml")
        # make thumbnails
        for img_type in config["Filesys"]["img_types"]:
            proc_path = proc_dir.joinpath(orig_filename.stem + img_type + ".png")
            img_base64, _ = image_to_base64(proc_path, thumb_size)
            img_thumbs[img_type] = img_base64
            img_names[img_type] = proc_path.name
    else:
        # 沒有 Cellpose Result 就給原圖
        orig_path = Path(selected_folder["path"]).joinpath(orig_filename)
        img_base64, _ = image_to_base64(orig_path, thumb_size)
        img_thumbs["orig"] = img_base64
        img_names["orig"] = orig_path.name

    # send to frontend
    socketio.emit('get_thumbs', {
        'img_thumbs': list(img_thumbs.values()),
        'img_names': list(img_names.values()),
        'img_types': list(img_thumbs.keys()),
    })
    # -------------------------------------------------------------------------/


@socketio.on('request_original_tif')
def handle_request_original_tif(data):
    """ 取得原始 tif
    
    socketio.emit : 'get_original_tif'
    """
    orig_filename = Path(data['filename'])
    orig_path = Path(selected_folder["path"]).joinpath(orig_filename)

    # send to frontend
    try:
        img_base64, _ = image_to_base64(orig_path)
        socketio.emit("get_original_tif", {"image": img_base64,
                                            "filename": orig_path.name})
    except Exception as e:
        print(f"[Error] 預覽 Original Tif 失敗：{e}")
    # -------------------------------------------------------------------------/


@socketio.on('request_preview')
def handle_request_preview(data):
    """ 取得預覽圖
    
    socketio.emit : 'get_preview'
    """
    filename = Path(data['filename'])

    # 判斷是 Original image 還是 Cellpose result
    if filename.suffix == ".png":
        img_type = filename.suffixes[-2]
        orig_filename = filename.stem.replace(img_type, '')
        path = Path(selected_folder["path"]).joinpath(orig_filename, filename)
    else:
        path = Path(selected_folder["path"]).joinpath(filename)

    # send to frontend
    try:
        img_base64, img_name = image_to_base64(path)
        socketio.emit("get_preview", {"image": img_base64,
                                        "filename": img_name})
    except Exception as e:
        print(f"[Error] 預覽失敗：{e}")
    # -------------------------------------------------------------------------/


@socketio.on('start_processing')
def handle_start_processing():
    socketio.start_background_task(process_files)
    # -------------------------------------------------------------------------/


def process_files():
    """ main process
    """
    cellpose_for_sec(scaned_files["files"], "cp_seg.toml",
                     console, socketio)
    socketio.emit('processing_complete')
    # -------------------------------------------------------------------------/


def start_flask():
    """ Flask 執行緒
    """
    socketio.run(app, debug=False)
    # -------------------------------------------------------------------------/


def open_folder_dialog():
    """ Qt Signal callback : 開啟 QFileDialog, 顯示資料夾內檔案但只可選資料夾
    """
    dialog = QFileDialog()
    dialog.setFileMode(QFileDialog.Directory) # 只能選資料夾
    dialog.setOption(QFileDialog.ShowDirsOnly, False) # ✅ 顯示資料夾內的檔案
    dialog.setWindowTitle("選擇資料夾")
    if dialog.exec_():
        folder = dialog.selectedFiles()[0]
        communicator.send_folder_selected.emit(folder)
    else:
        communicator.send_folder_selected.emit("")
    # -------------------------------------------------------------------------/


def send_folder_to_client(path):
    """ Qt Signal callback : 收到選擇結果後傳給前端
    """
    print(f"[Qt] 已選擇資料夾：{path}")
    selected_folder["path"] = path
    socketio.start_background_task(socketio.emit, 'get_folder_selected', {'path': path})
    
    # 掃描 .tif/.tiff 檔案
    scaned_files["files"] = [file for file in sorted(Path(path).glob("*.tif*"))]
    
    socketio.start_background_task(socketio.emit, 'tiff_list', {
        'folder': path,
        'files': [file.name for file in scaned_files["files"]]
    })
    # -------------------------------------------------------------------------/


def open_file_location(path):
    """
    """
    # path = Path(path) if path else Path(__file__) # test

    system = platform.system()
    try:
        if system == "Windows":
            # Windows 用 explorer /select,
            subprocess.run(['explorer', '/select,', str(path)])
        elif system == "Darwin":
            # macOS 用 open -R
            subprocess.run(['open', '-R', str(path)])
        elif system == "Linux":
            # Linux 用 dolphin --select，沒裝提示
            if shutil.which("dolphin"):
                subprocess.run(["dolphin", "--select", str(path)])
            else:
                console.print("[#FFFF00] Linux 系統找不到 dolphin, 請安裝後再試 (例如 : sudo apt install dolphin)")
    except Exception as e:
        console.print("[#FF0000] Can't open file location, "
                                f"path: '{path}' \n {e}")
    # -------------------------------------------------------------------------/


def handle_quit():
    print("[Qt] QApplication 正準備退出！")
    # -------------------------------------------------------------------------/


# 主程式入口
if __name__ == '__main__':
    # 啟動 Flask background thread
    threading.Thread(target=start_flask, daemon=True).start()

    # Qt 主應用與隱藏視窗
    qt_app = QApplication(sys.argv)
    qt_app.setQuitOnLastWindowClosed(False)  # 這行保證沒視窗也不會退出

    # 設定 signal-slot
    communicator.request_select_folder.connect(open_folder_dialog)
    communicator.send_folder_selected.connect(send_folder_to_client)
    communicator.request_open_csv_loca.connect(open_file_location)
    qt_app.aboutToQuit.connect(handle_quit)

    sys.exit(qt_app.exec_())  # 正常結束程式
