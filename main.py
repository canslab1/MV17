"""
應用程式入口點 (Application Entry Point)。

本檔案是整個 GUI 應用程式的啟動點。
舊版程式沒有圖形介面，所有功能都是透過命令列腳本個別執行的。
這個新版 GUI 框架將所有功能整合到一個視窗化應用程式中。

本檔案的主要職責：
1. 設定 sys.path — 確保 gui_app/ 目錄下的子模組（core/、tabs/、widgets/）
   可以被正確匯入
2. 設定 matplotlib 後端 — 必須在任何 matplotlib 繪圖匯入之前，
   將後端切換為 'QtAgg'，這樣 matplotlib 的圖表才能嵌入 PySide6 的 Qt 視窗中
3. 設定 CJK 字型 — 因為本專案需要顯示中文（繁體），所以必須在
   plt.rcParams 中指定支援 CJK 的字型清單，否則中文會顯示為方框
4. 建立 QApplication 並啟動主視窗 (MainWindow)

目錄結構說明：
    本檔案 (main.py) 和 main_window.py 位於專案根目錄（MV17/），
    而 core/、tabs/、widgets/ 等子模組位於 gui_app/ 子目錄中。
    因此需要將 gui_app/ 加入 sys.path，讓 Python 能找到這些子模組。

啟動方式：
    cd MV17
    python main.py
"""
import sys
import os

# ---------------------------------------------------------------------------
# 路徑設定
# ---------------------------------------------------------------------------
# 取得本檔案 (main.py) 所在的目錄，即專案根目錄 (MV17/)
# 根目錄包含 edgelist/（邊列表檔案）等資料目錄，
# 以及 main_window.py（主視窗模組）
project_root = os.path.dirname(os.path.abspath(__file__))

# gui_app/ 子目錄包含 core/、tabs/、widgets/ 等套件
# 將它加入 sys.path，這樣 Python 才能解析這些子模組的匯入
# （例如 core/network_manager.py 中的 from core import algorithm_adapter）
gui_app_dir = os.path.join(project_root, 'gui_app')
if gui_app_dir not in sys.path:
    sys.path.insert(0, gui_app_dir)

# ---------------------------------------------------------------------------
# matplotlib 後端設定（必須在任何 matplotlib 繪圖 import 之前完成）
# ---------------------------------------------------------------------------
# 使用 'QtAgg' 後端，讓 matplotlib 的圖表可以嵌入 Qt 視窗元件中。
# 如果不設定這個後端，matplotlib 會使用預設的獨立視窗，
# 無法與 PySide6 的介面整合。
# 注意：matplotlib.use() 必須在 import matplotlib.pyplot 之前呼叫，
# 否則後端切換會失敗。
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# CJK 字型設定
# ---------------------------------------------------------------------------
# 本專案的圖表標題、軸標籤等可能包含中文字元（繁體中文）。
# matplotlib 預設字型不支援 CJK 字元，因此需要手動指定字型清單。
# 清單中的字型會按照順序嘗試：
#   - 'PingFang TC'：macOS 內建的繁體中文字型（蘋方）
#   - 'Heiti TC'：macOS 內建的黑體繁體
#   - 'Microsoft JhengHei'：Windows 內建的微軟正黑體
#   - 'Noto Sans CJK TC'：Google 的 Noto 字型（跨平台）
#   - 'Arial Unicode MS'：包含多國語言的 Arial 變體
#   - 'sans-serif'：最後的備用選項
# axes.unicode_minus = False 是為了避免負號 (-) 在 CJK 字型下顯示異常
plt.rcParams['font.sans-serif'] = [
    'PingFang TC', 'Heiti TC', 'Microsoft JhengHei',
    'Noto Sans CJK TC', 'Arial Unicode MS', 'sans-serif'
]
plt.rcParams['axes.unicode_minus'] = False

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont

from main_window import MainWindow


def main():
    """
    應用程式主函式。

    建立 QApplication 實例、設定全域字型大小、
    建立並顯示主視窗，然後進入 Qt 事件迴圈。
    當主視窗關閉後，app.exec() 會返回，程式隨之結束。
    """
    app = QApplication(sys.argv)

    # 設定全域字型大小為 12pt，讓所有 UI 元件的文字更易閱讀
    font = QFont()
    font.setPointSize(12)
    app.setFont(font)

    # 建立主視窗，傳入專案根目錄路徑，
    # 讓主視窗能夠存取 edgelist/ 等資料目錄
    window = MainWindow(project_root)
    window.show()

    # 進入 Qt 事件迴圈，直到視窗關閉
    # sys.exit() 確保程式的退出碼能正確傳遞給作業系統
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
