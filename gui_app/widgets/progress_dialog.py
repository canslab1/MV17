"""
進度對話框模組 — 為長時間運算提供模態進度顯示。

本模組提供 ProgressDialog 類別，用於在執行耗時運算時
顯示一個模態 (modal) 對話框，包含進度條和狀態訊息。

舊版程式沒有 GUI，長時間運算（如 SIR 模擬、節點屬性計算）
只能在終端機中顯示文字進度。新版 GUI 使用此對話框來提供
視覺化的進度回饋，並支援取消操作。

模態對話框模式 (Modal Dialog Pattern)：
    對話框設定為 Qt.ApplicationModal，意味著當它顯示時，
    使用者無法與應用程式的其他視窗互動。這是為了防止使用者
    在運算進行中修改資料或觸發其他運算，避免資料競爭問題。
    同時，對話框的關閉按鈕被隱藏，使用者只能透過「Cancel」
    按鈕來取消運算。

搭配 Worker 使用：
    ProgressDialog 通常搭配背景執行緒 (QThread + Worker) 使用。
    典型的使用流程：

        1. 建立 Worker 物件（在背景執行緒中執行運算）
        2. 建立 ProgressDialog 物件
        3. 呼叫 dialog.set_worker(worker) 關聯 Worker
        4. 連接 Worker 的進度信號到 dialog.update_progress()
        5. 連接 Worker 的完成信號到 dialog.accept()（關閉對話框）
        6. 啟動 Worker 執行緒
        7. 呼叫 dialog.exec()（阻塞直到對話框關閉）

    程式碼範例：
        worker = SomeWorker(data)
        dialog = ProgressDialog("Computing...", parent=self)
        dialog.set_worker(worker)
        worker.progress.connect(dialog.update_progress)
        worker.finished.connect(dialog.accept)
        worker.start()
        dialog.exec()
"""
from PySide6.QtWidgets import QDialog, QVBoxLayout, QProgressBar, QLabel, QPushButton
from PySide6.QtCore import Qt


class ProgressDialog(QDialog):
    """
    模態進度對話框。

    顯示一個包含以下元素的對話框：
    - 狀態標籤 (QLabel)：顯示當前操作的描述文字
    - 進度條 (QProgressBar)：顯示完成百分比 (0-100)
    - 取消按鈕 (QPushButton)：讓使用者可以中斷運算

    對話框的特性：
    - 模態 (ApplicationModal)：阻止與其他視窗的互動
    - 無關閉按鈕：防止使用者意外關閉對話框
    - 最小寬度 400px：確保進度條和文字有足夠的顯示空間

    參數：
        title (str): 對話框標題。預設為 "Processing..."
        parent (QWidget, optional): 父元件。預設為 None
    """

    def __init__(self, title="Processing...", parent=None):
        """
        初始化進度對話框。

        建立佈局和元件，設定模態屬性，並隱藏視窗的關閉按鈕。
        初始狀態文字為「Initializing...」，進度為 0%。
        """
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(400)

        # 設定為應用程式級別的模態對話框
        # 當此對話框顯示時，使用者無法與應用程式的任何其他視窗互動
        self.setWindowModality(Qt.ApplicationModal)

        # 移除視窗的關閉按鈕 (X)，防止使用者意外關閉對話框
        # 使用者必須透過「Cancel」按鈕來取消操作
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowCloseButtonHint)

        layout = QVBoxLayout(self)

        # 狀態標籤：顯示當前操作的描述文字
        self.label = QLabel("Initializing...")
        layout.addWidget(self.label)

        # 進度條：範圍 0-100，代表完成百分比
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)

        # 取消按鈕：點擊後呼叫 Worker 的 cancel() 方法
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._on_cancel)
        layout.addWidget(self.cancel_btn)

        # 保存對應的 Worker 物件參考，用於在取消時通知 Worker 停止
        self._worker = None

    def set_worker(self, worker):
        """
        設定對應的 Worker 物件。

        Worker 是在背景執行緒中執行的運算物件。
        設定後，當使用者按下「Cancel」按鈕時，
        對話框會呼叫 worker.cancel() 來請求 Worker 停止運算。

        參數：
            worker: 背景工作物件，必須提供 cancel() 方法
        """
        self._worker = worker

    def update_progress(self, percent, message):
        """
        更新進度條和狀態文字。

        通常由 Worker 的 progress 信號觸發。
        Worker 在運算過程中定期發送進度更新，
        透過 Qt 的信號機制跨執行緒安全地更新 UI。

        參數：
            percent (int): 完成百分比，範圍 0-100
            message (str): 當前操作的描述文字，例如
                          "Computing betweenness centrality (45/200)..."
        """
        self.progress_bar.setValue(percent)
        self.label.setText(message)

    def _on_cancel(self):
        """
        取消按鈕的回呼函式。

        當使用者按下「Cancel」按鈕時：
        1. 呼叫 Worker 的 cancel() 方法，設定取消旗標
           （Worker 會在下次檢查時停止運算）
        2. 將狀態文字更新為「Cancelling...」
        3. 禁用取消按鈕，防止重複點擊

        注意：取消操作是非同步的。呼叫 worker.cancel() 只是設定一個旗標，
        Worker 會在下次迴圈迭代中檢查該旗標並停止。
        對話框不會立即關閉，而是等待 Worker 完成清理後，
        由 Worker 的 finished 信號觸發 dialog.accept() 來關閉。
        """
        if self._worker is not None:
            self._worker.cancel()
        self.label.setText("Cancelling...")
        self.cancel_btn.setEnabled(False)
