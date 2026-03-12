"""
Tab 1: 網路 I/O (Network I/O) — 載入/儲存邊列表 (edgelist)、提取最大連通分量 (GCC)、
載入輔助屬性資料。

本分頁對應舊版 measure_node_attribute.py 的 main_function() 中的以下步驟:

  舊版 Step 1:「從邊列表檔案建立網路」(Create a network from edgelist file)
      ─ 載入 .txt 邊列表檔並解析為 NetworkX 圖物件 (Graph)。
      ─ 在本 GUI 中，使用者透過左側檔案樹瀏覽 edgelist/ 資料夾，雙擊或按「Load Network」
        按鈕來載入，取代舊版手動設定 file_name 變數的方式。

  舊版 Step 3:「附加新屬性」(Append new attributes)
      ─ 載入介數中心性 (_tbet.txt)、接近中心性 (_clos.txt)、節點座標 (_pos.txt) 等
        預先計算好的輔助檔案，並寫入網路節點的屬性字典中。
      ─ 在本 GUI 中，對應右下方「Load Auxiliary Data」群組的四顆按鈕。

  此外本分頁還提供:
      ─ 提取 GCC (Giant Connected Component，最大連通分量)
      ─ 儲存邊列表檔 (Save Edgelist)

工作執行緒模式 (Worker Thread Pattern):
    為避免載入大型網路時 UI 凍結，所有耗時操作都在背景執行緒中執行:
      1. _load_file(filepath) 建立 NetworkLoadWorker 並啟動它
      2. NetworkLoadWorker 在背景執行緒中讀取邊列表、建構圖
      3. 完成後發射 finished 信號，觸發 _on_load_finished() 回到主執行緒更新 UI
    GCC 提取也使用相同模式 (GCCExtractWorker)。
"""

import os
import networkx as nx
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QGroupBox, QLabel,
    QPushButton, QTreeWidget, QTreeWidgetItem, QFileDialog, QMessageBox
)
from PySide6.QtCore import Qt
from core.worker_threads import NetworkLoadWorker, GCCExtractWorker
from core import algorithm_adapter as algo
from widgets.progress_dialog import ProgressDialog


class TabNetworkIO(QWidget):
    """
    網路 I/O 分頁 (Tab 1)。

    職責:
      - 左側面板: 掃描並顯示 edgelist/ 資料夾中的邊列表檔案樹
        (取代舊版在 measure_node_attribute.py 中手動指定 file_name 變數的做法)
      - 右側面板:
          * 「Network Info」群組: 顯示已載入網路的基本資訊 (名稱、節點數、邊數、連通分量數)
          * 「Actions」群組: 載入網路 / 提取 GCC / 儲存邊列表
          * 「Load Auxiliary Data」群組: 載入輔助屬性檔 (_tbet.txt, _clos.txt, _pos.txt, .umsgpack)

    參數:
        manager: NetworkManager 實例，集中管理目前的網路 (G)、節點屬性 (net_attr)、
                 座標 (pos) 等共享狀態。
        parent: 父 QWidget (預設 None)。
    """

    def __init__(self, manager, parent=None):
        """
        初始化 TabNetworkIO。

        - 儲存 manager 參照，用於讀寫共享的網路資料。
        - self._worker: 持有目前正在執行的背景工作執行緒的參照，
          防止它被 Python 垃圾回收 (garbage collected) 而提前中斷。
        - 呼叫 _init_ui() 建構介面元件。
        - 連接 manager 的 network_loaded / network_cleared 信號，
          讓載入或清除網路時自動更新本分頁的 UI 狀態。
        """
        super().__init__(parent)
        self.manager = manager
        # 持有背景工作執行緒的參照，避免被垃圾回收
        self._worker = None
        self._init_ui()
        # 當 manager 發出「網路已載入」信號時，更新右側資訊面板
        self.manager.network_loaded.connect(self._on_network_loaded)
        # 當 manager 發出「網路已清除」信號時，重設右側資訊面板
        self.manager.network_cleared.connect(self._on_network_cleared)

    # ---------------------------------------------------------------
    # UI 建構
    # ---------------------------------------------------------------

    def _init_ui(self):
        """
        建構本分頁的介面配置 (Layout)。

        整體為水平佈局 (QHBoxLayout)，分為左右兩欄:

        左欄 (佔比 4):
            - 標籤「Edgelist Files:」
            - QTreeWidget 檔案樹: 顯示 edgelist/ 資料夾下的 .txt 檔案
              (排除 _tbet.txt, _clos.txt, _abet.txt, _pos.txt, _sirr.txt 等輔助檔)
              使用者雙擊檔案即觸發載入。
              此檔案樹取代舊版手動在程式碼中設定 file_name 的流程。
            - 「Browse...」按鈕: 允許使用者透過系統檔案對話框選擇任意 .txt 邊列表檔

        右欄 (佔比 6):
            - 「Network Info」群組: 顯示名稱、節點數 (N)、邊數 (E)、連通分量數
            - 「Actions」群組:
                * Load Network: 載入左側選取的檔案 (對應舊版 Step 1)
                * Extract GCC: 從目前的網路中提取最大連通分量
                * Save Edgelist: 將目前的網路儲存為邊列表檔
            - 「Load Auxiliary Data」群組 (對應舊版 Step 3):
                * Load Betweenness (_tbet.txt): 載入介數中心性
                * Load Closeness (_clos.txt): 載入接近中心性
                * Load Position (_pos.txt): 載入節點座標
                * Load Attributes (.umsgpack): 載入完整節點屬性 (含多種中心性指標)
        """
        layout = QHBoxLayout(self)

        # ---- 左欄: 檔案樹 ----
        left = QVBoxLayout()
        left_label = QLabel("Edgelist Files:")
        left.addWidget(left_label)

        # 檔案樹元件: 顯示 edgelist/ 資料夾結構
        # 雙擊某個檔案項目時會觸發 _on_file_double_clicked
        self.file_tree = QTreeWidget()
        self.file_tree.setHeaderLabels(["File"])
        self.file_tree.itemDoubleClicked.connect(self._on_file_double_clicked)
        left.addWidget(self.file_tree)

        # 「Browse...」按鈕: 透過系統對話框手動選擇檔案
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_file)
        left.addWidget(browse_btn)

        # 左欄佔水平空間的比例為 4
        layout.addLayout(left, 4)

        # ---- 右欄: 資訊顯示 + 操作按鈕 ----
        right = QVBoxLayout()

        # -- Network Info 群組: 顯示已載入網路的基本統計資訊 --
        info_group = QGroupBox("Network Info")
        info_layout = QVBoxLayout(info_group)
        self.lbl_name = QLabel("Name: -")
        self.lbl_nodes = QLabel("Nodes (N): -")
        self.lbl_edges = QLabel("Edges (E): -")
        self.lbl_components = QLabel("Connected Components: -")
        for lbl in [self.lbl_name, self.lbl_nodes, self.lbl_edges, self.lbl_components]:
            info_layout.addWidget(lbl)
        right.addWidget(info_group)

        # -- Actions 群組: 主要操作按鈕 --
        action_group = QGroupBox("Actions")
        action_layout = QVBoxLayout(action_group)

        # 「Load Network」按鈕: 載入左側檔案樹中選取的邊列表檔
        # 對應舊版 Step 1: 從邊列表檔建立網路
        self.btn_load = QPushButton("Load Network")
        self.btn_load.clicked.connect(self._load_selected)
        action_layout.addWidget(self.btn_load)

        # 「Extract GCC」按鈕: 從已載入的網路中提取最大連通分量 (Giant Connected Component)
        # 載入網路後才啟用
        self.btn_gcc = QPushButton("Extract GCC")
        self.btn_gcc.clicked.connect(self._extract_gcc)
        self.btn_gcc.setEnabled(False)
        action_layout.addWidget(self.btn_gcc)

        # 「Save Edgelist...」按鈕: 將目前網路的邊列表儲存為 .txt 檔
        # 載入網路後才啟用
        self.btn_save = QPushButton("Save Edgelist...")
        self.btn_save.clicked.connect(self._save_edgelist)
        self.btn_save.setEnabled(False)
        action_layout.addWidget(self.btn_save)

        right.addWidget(action_group)

        # -- Load Auxiliary Data 群組: 載入輔助屬性檔案 --
        # 對應舊版 measure_node_attribute.py main_function() 的 Step 3:
        # 「Append new attributes」── 將預先計算好的節點屬性附加到 net_attr 字典中。
        # 這些輔助檔案的命名慣例是在原始邊列表檔名後加上特定後綴。
        aux_group = QGroupBox("Load Auxiliary Data")
        aux_layout = QVBoxLayout(aux_group)

        # 載入介數中心性 (betweenness centrality) 檔案
        # 檔名後綴: _tbet.txt (truncated betweenness)
        # 對應舊版 Step 3 中讀取 betweenness 資料的部分
        self.btn_load_bet = QPushButton("Load Betweenness (_tbet.txt)")
        self.btn_load_bet.clicked.connect(lambda: self._load_auxiliary('betweenness', '_tbet.txt'))
        self.btn_load_bet.setEnabled(False)
        aux_layout.addWidget(self.btn_load_bet)

        # 載入接近中心性 (closeness centrality) 檔案
        # 檔名後綴: _clos.txt
        # 對應舊版 Step 3 中讀取 closeness 資料的部分
        self.btn_load_clos = QPushButton("Load Closeness (_clos.txt)")
        self.btn_load_clos.clicked.connect(lambda: self._load_auxiliary('closeness', '_clos.txt'))
        self.btn_load_clos.setEnabled(False)
        aux_layout.addWidget(self.btn_load_clos)

        # 載入節點位置座標檔案
        # 檔名後綴: _pos.txt
        # 對應舊版 Step 3 中讀取 position 資料的部分
        # 座標用於後續的網路視覺化繪圖
        self.btn_load_pos = QPushButton("Load Position (_pos.txt)")
        self.btn_load_pos.clicked.connect(self._load_pos)
        self.btn_load_pos.setEnabled(False)
        aux_layout.addWidget(self.btn_load_pos)

        # 載入完整屬性資料 (.umsgpack 格式)
        # umsgpack 是 MessagePack 序列化格式，可一次載入所有預先計算的節點屬性
        # (包含介數中心性、接近中心性、度數等多種指標)
        self.btn_load_attr = QPushButton("Load Attributes (.umsgpack)")
        self.btn_load_attr.clicked.connect(self._load_umsgpack)
        self.btn_load_attr.setEnabled(False)
        aux_layout.addWidget(self.btn_load_attr)

        right.addWidget(aux_group)
        # 底部彈性空間，讓群組靠上對齊
        right.addStretch()
        # 右欄佔水平空間的比例為 6
        layout.addLayout(right, 6)

    # ---------------------------------------------------------------
    # 檔案樹操作
    # ---------------------------------------------------------------

    def populate_file_tree(self, edgelist_dir):
        """
        掃描指定的 edgelist 資料夾，將其中的 .txt 邊列表檔填入左側檔案樹。

        此方法取代舊版 measure_node_attribute.py 中手動設定 file_name 變數的做法。
        舊版流程中，使用者需要在程式碼裡直接修改 file_name = 'xxx' 來切換要分析的網路；
        現在改為由此檔案樹自動掃描資料夾、列出所有可用的邊列表檔，使用者只需點選即可。

        掃描時會跳過以下後綴的輔助檔案 (它們由「Load Auxiliary Data」群組的按鈕載入):
            _tbet.txt (介數中心性)
            _clos.txt (接近中心性)
            _abet.txt (近似介數中心性)
            _pos.txt  (節點座標)
            _sirr.txt (SIR 模擬結果)

        參數:
            edgelist_dir: edgelist 資料夾的絕對路徑。
        """
        self.file_tree.clear()
        if not os.path.isdir(edgelist_dir):
            return
        # 記住 edgelist 資料夾路徑，供後續使用
        self._edgelist_dir = edgelist_dir
        # 定義要跳過的輔助檔案後綴
        suffixes_to_skip = ('_tbet.txt', '_clos.txt', '_abet.txt', '_pos.txt', '_sirr.txt')

        # 遞迴走訪資料夾結構
        for root, dirs, files in sorted(os.walk(edgelist_dir)):
            dirs.sort()
            # 計算相對於 edgelist_dir 的相對路徑
            rel = os.path.relpath(root, edgelist_dir)
            if rel == '.':
                # 根目錄的檔案直接加到樹的不可見根項目下
                parent_item = self.file_tree.invisibleRootItem()
            else:
                # 子資料夾建立為樹的頂層項目 (資料夾名稱)
                parent_item = QTreeWidgetItem([rel])
                # 資料夾項目的 UserRole 設為 None，表示它不是可載入的檔案
                parent_item.setData(0, Qt.UserRole, None)
                self.file_tree.addTopLevelItem(parent_item)

            # 將該目錄下的 .txt 檔案 (排除輔助檔) 加為子項目
            for fname in sorted(files):
                if not fname.endswith('.txt'):
                    continue
                # 跳過輔助屬性檔案
                if any(fname.endswith(s) for s in suffixes_to_skip):
                    continue
                child = QTreeWidgetItem([fname])
                # 將檔案的完整路徑存入 UserRole，以便之後載入時取用
                child.setData(0, Qt.UserRole, os.path.join(root, fname))
                parent_item.addChild(child)

        # 展開所有節點，方便使用者一覽全部檔案
        self.file_tree.expandAll()

    # ---------------------------------------------------------------
    # 檔案選取與載入
    # ---------------------------------------------------------------

    def _get_selected_path(self):
        """
        取得檔案樹中目前選取項目的完整檔案路徑。

        傳回值:
            str: 檔案的絕對路徑；若未選取或選取的是資料夾項目則傳回 None。
        """
        items = self.file_tree.selectedItems()
        if not items:
            return None
        return items[0].data(0, Qt.UserRole)

    def _on_file_double_clicked(self, item, column):
        """
        當使用者在檔案樹中雙擊某個項目時觸發。

        若該項目儲存了有效的檔案路徑 (即它是邊列表檔而非資料夾)，
        則呼叫 _load_file() 開始載入網路。

        參數:
            item: 被雙擊的 QTreeWidgetItem。
            column: 被雙擊的欄位索引 (此處只有一欄)。
        """
        path = item.data(0, Qt.UserRole)
        if path:
            self._load_file(path)

    def _browse_file(self):
        """
        開啟系統檔案對話框，讓使用者手動選擇一個邊列表檔案。

        當使用者的邊列表檔不在預設的 edgelist/ 資料夾中時，
        可透過此按鈕從檔案系統中任意位置選取。
        """
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Edgelist", "", "Text Files (*.txt);;All Files (*)")
        if path:
            self._load_file(path)

    def _load_selected(self):
        """
        「Load Network」按鈕的點擊處理函式。

        從檔案樹取得目前選取的邊列表檔路徑，然後呼叫 _load_file() 載入。
        若未選取任何檔案，彈出警告訊息。

        對應舊版 Step 1: 從邊列表檔建立網路。
        """
        path = self._get_selected_path()
        if path is None:
            QMessageBox.warning(self, "Warning", "Please select a file first.")
            return
        self._load_file(path)

    # ---------------------------------------------------------------
    # 背景執行緒載入網路 (Worker Thread Pattern)
    # ---------------------------------------------------------------
    #
    # 載入大型網路 (例如數萬個節點) 可能需要數秒甚至更長時間。
    # 若在主執行緒 (GUI 執行緒) 中直接執行，UI 會凍結無法回應使用者操作。
    #
    # 因此採用以下工作執行緒模式:
    #
    #   _load_file(filepath)
    #       |
    #       |  1. 建立 ProgressDialog (進度對話框)
    #       |  2. 建立 NetworkLoadWorker (背景工作執行緒)
    #       |  3. 連接信號: progress -> 更新進度條
    #       |              finished -> _on_load_finished (完成回呼)
    #       |              error -> _on_error (錯誤回呼)
    #       |  4. 啟動工作執行緒 (.start())
    #       v
    #   NetworkLoadWorker (在背景執行緒中運行)
    #       |
    #       |  讀取邊列表檔 -> 解析 -> 建構 NetworkX 圖物件
    #       |  期間透過 progress 信號回報進度
    #       v
    #   _on_load_finished(result, filepath)  [回到主執行緒]
    #       |
    #       |  關閉進度對話框
    #       |  從檔名提取網路名稱
    #       |  呼叫 manager.set_network() 將結果存入共享狀態
    #       v
    #   manager.network_loaded 信號被發射
    #       -> _on_network_loaded() 更新右側的 Network Info 面板
    #

    def _load_file(self, filepath):
        """
        啟動背景執行緒載入邊列表檔案，對應舊版 Step 1:「從邊列表檔建立網路」。

        流程:
          1. 建立 ProgressDialog 顯示載入進度
          2. 建立 NetworkLoadWorker 背景工作執行緒
          3. 連接信號與槽 (signal-slot):
             - progress: 更新進度對話框的進度值
             - finished: 載入完成後呼叫 _on_load_finished
             - error: 發生錯誤時呼叫 _on_error
          4. 顯示進度對話框並啟動工作執行緒

        參數:
            filepath: 邊列表檔案的絕對路徑。
        """
        # 建立並顯示進度對話框
        self._progress = ProgressDialog("Loading Network...", self)
        # 建立背景工作執行緒，傳入檔案路徑
        self._worker = NetworkLoadWorker(filepath)
        # 將工作執行緒註冊到進度對話框 (用於取消操作等)
        self._progress.set_worker(self._worker)
        # 連接信號: 工作執行緒的 progress 信號 -> 進度對話框的更新方法
        self._worker.progress.connect(self._progress.update_progress)
        # 連接信號: 工作執行緒的 finished 信號 -> 載入完成回呼
        # 使用 lambda 捕獲 filepath，以便在完成時知道是哪個檔案
        self._worker.finished.connect(lambda result: self._on_load_finished(result, filepath))
        # 連接信號: 工作執行緒的 error 信號 -> 錯誤處理回呼
        self._worker.error.connect(self._on_error)
        self._progress.show()
        # 啟動背景執行緒 (呼叫 QThread.start()，實際工作在 run() 中執行)
        self._worker.start()

    def _on_load_finished(self, result, filepath):
        """
        網路載入完成的回呼函式 (在主執行緒中執行)。

        當 NetworkLoadWorker 在背景執行緒中完成網路建構後，
        會發射 finished 信號，攜帶 result 字典。此方法接收該結果，
        關閉進度對話框，並透過 manager.set_network() 將網路儲存到共享狀態。

        manager.set_network() 內部會發射 network_loaded 信號，
        進而觸發 _on_network_loaded() 更新 UI。

        參數:
            result: 字典，包含:
                - 'G': 建構完成的 NetworkX 圖物件
                - 'pos': 節點座標字典 (若邊列表檔中有嵌入座標; 否則可能為 None)
            filepath: 載入的邊列表檔案路徑。
        """
        self._progress.close()
        # 從檔案路徑提取網路名稱 (去掉目錄和副檔名)
        name = os.path.splitext(os.path.basename(filepath))[0]
        # 將網路、名稱、檔案路徑、座標存入 manager 共享狀態
        self.manager.set_network(result['G'], name, filepath, result['pos'])

    # ---------------------------------------------------------------
    # 提取最大連通分量 (GCC)
    # ---------------------------------------------------------------

    def _extract_gcc(self):
        """
        提取目前網路的最大連通分量 (Giant Connected Component, GCC)。

        許多網路分析演算法要求輸入為連通圖。若原始網路包含多個連通分量，
        可使用此功能提取最大的那個分量，丟棄孤立的小群組。

        同樣使用背景工作執行緒模式 (GCCExtractWorker) 以避免 UI 凍結。
        """
        if not self.manager.has_network():
            return
        self._progress = ProgressDialog("Extracting GCC...", self)
        # 建立 GCC 提取的背景工作執行緒
        self._worker = GCCExtractWorker(self.manager.G)
        self._progress.set_worker(self._worker)
        self._worker.progress.connect(self._progress.update_progress)
        self._worker.finished.connect(self._on_gcc_finished)
        self._worker.error.connect(self._on_error)
        self._progress.show()
        self._worker.start()

    def _on_gcc_finished(self, result):
        """
        GCC 提取完成的回呼函式。

        取得提取後的子圖，在名稱後加上 '_gcc' 後綴以資區別，
        然後更新 manager 中的網路。

        參數:
            result: 字典，包含:
                - 'G': GCC 子圖 (NetworkX Graph)
                - 'pos': 新的節點座標 (僅保留 GCC 中的節點)
        """
        self._progress.close()
        name = self.manager.network_name
        # 若名稱尚未包含 '_gcc' 後綴，則加上
        if not name.endswith('_gcc'):
            name += '_gcc'
        self.manager.set_network(result['G'], name, self.manager.edgelist_path, result['pos'])

    # ---------------------------------------------------------------
    # 儲存邊列表
    # ---------------------------------------------------------------

    def _save_edgelist(self):
        """
        將目前網路的邊列表儲存為 .txt 檔案。

        開啟「另存新檔」對話框讓使用者選擇儲存路徑，
        然後呼叫 algo.write_edgelist() 寫出檔案。

        適用情境: 提取 GCC 後，想將精簡過的網路儲存下來供後續使用。
        """
        if not self.manager.has_network():
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Edgelist", "", "Text Files (*.txt)")
        if path:
            algo.write_edgelist(path, self.manager.G)

    # ---------------------------------------------------------------
    # 載入輔助屬性資料 (對應舊版 Step 3: Append new attributes)
    # ---------------------------------------------------------------
    #
    # 舊版 measure_node_attribute.py 的 Step 3 會讀取預先計算好的節點屬性檔案，
    # 並將這些屬性附加 (append) 到 net_attr 字典中。每個節點的屬性存放在
    # net_attr[node_id] 字典裡，鍵值對如 {NODE_BETWEENNESS: 0.123, ...}。
    #
    # 在本 GUI 中，這些操作由「Load Auxiliary Data」群組的四顆按鈕觸發:
    #   - Load Betweenness (_tbet.txt)  -> _load_auxiliary('betweenness', '_tbet.txt')
    #   - Load Closeness (_clos.txt)    -> _load_auxiliary('closeness', '_clos.txt')
    #   - Load Position (_pos.txt)      -> _load_pos()
    #   - Load Attributes (.umsgpack)   -> _load_umsgpack()
    #
    # 輔助檔案的命名慣例:
    #   若邊列表檔為 network_name.txt，則對應的輔助檔為:
    #     network_name_tbet.txt  (介數中心性)
    #     network_name_clos.txt  (接近中心性)
    #     network_name_pos.txt   (節點座標)
    #   程式會先嘗試自動拼接路徑尋找這些檔案；若找不到，則開啟檔案對話框讓使用者手動選擇。
    #

    def _load_auxiliary(self, attr_type, suffix):
        """
        載入介數中心性或接近中心性的輔助屬性檔案。

        對應舊版 measure_node_attribute.py 的 Step 3:
        將預先計算好的節點屬性從檔案讀入，並附加到 manager.net_attr 字典中。

        自動路徑拼接邏輯:
          將目前邊列表檔的副檔名去掉，加上 suffix (如 '_tbet.txt')，
          若該路徑存在則直接載入；否則開啟檔案對話框。

        參數:
            attr_type: 屬性類型字串，'betweenness' 或 'closeness'。
            suffix: 輔助檔案的後綴，如 '_tbet.txt' 或 '_clos.txt'。
        """
        if not self.manager.has_network():
            return
        # 根據目前邊列表檔路徑，拼接出預期的輔助檔案路徑
        base = os.path.splitext(self.manager.edgelist_path)[0]
        default_path = base + suffix
        if os.path.exists(default_path):
            path = default_path
        else:
            # 預設路徑不存在，開啟檔案對話框讓使用者手動選擇
            path, _ = QFileDialog.getOpenFileName(
                self, f"Load {attr_type}", "", "Text Files (*.txt)")
            if not path:
                return

        try:
            # 讀取「節點-數值」對的檔案，格式為每行 "node_id value"
            data = algo.read_pairvalue_file(path)
            # 若 net_attr 尚未初始化，先為每個節點建立空的屬性字典
            if self.manager.net_attr is None:
                net_attr = {}
                for ni in self.manager.G.nodes():
                    net_attr[ni] = {algo.NODE_ID: ni}
                self.manager.net_attr = net_attr

            # 根據 attr_type 選擇正確的屬性鍵名 (常數定義在 algorithm_adapter 中)
            attr_name = algo.NODE_BETWEENNESS if attr_type == 'betweenness' else algo.NODE_CLOSENESS
            # 將讀取到的資料附加到 net_attr 字典中的每個節點
            algo.append_attribute(self.manager.net_attr, data, attr_name)
            QMessageBox.information(self, "Success", f"Loaded {attr_type} data ({len(data)} nodes).")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _load_pos(self):
        """
        載入節點座標檔案 (_pos.txt)。

        對應舊版 Step 3 中讀取位置資料的部分。
        座標資料用於網路視覺化繪圖，決定每個節點在二維平面上的 (x, y) 位置。

        與 _load_auxiliary() 類似，先嘗試自動拼接路徑，找不到則開啟檔案對話框。
        讀取後直接存入 manager.pos。
        """
        if not self.manager.has_network():
            return
        base = os.path.splitext(self.manager.edgelist_path)[0]
        default_path = base + '_pos.txt'
        if os.path.exists(default_path):
            path = default_path
        else:
            path, _ = QFileDialog.getOpenFileName(
                self, "Load Position", "", "Text Files (*.txt)")
            if not path:
                return
        try:
            # 讀取座標檔案，格式為每行 "node_id x y"
            self.manager.pos = algo.read_pos_file(path)
            QMessageBox.information(self, "Success", "Loaded position data.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _load_umsgpack(self):
        """
        載入完整的節點屬性資料 (.umsgpack 格式)。

        umsgpack (MessagePack) 是一種高效率的二進位序列化格式。
        此檔案包含所有預先計算好的節點屬性 (介數中心性、接近中心性、度數等)，
        是舊版 Step 3 輔助檔案的整合版本 —— 一個檔案即可載入全部屬性。

        預設路徑: project_root/file/{network_name}-attr.umsgpack
        若預設路徑不存在，則開啟檔案對話框。
        讀取後呼叫 manager.set_attributes() 存入共享狀態。
        """
        if not self.manager.has_network():
            return
        base_name = self.manager.network_name
        project_root = self.manager.project_root
        # 預設路徑: project_root/file/{網路名稱}-attr.umsgpack
        default_path = os.path.join(project_root, 'file', f'{base_name}-attr.umsgpack')
        if os.path.exists(default_path):
            path = default_path
        else:
            path, _ = QFileDialog.getOpenFileName(
                self, "Load Attributes", "", "MsgPack Files (*.umsgpack);;All Files (*)")
            if not path:
                return
        try:
            data = algo.read_umsgpack_data(path)
            self.manager.set_attributes(data)
            QMessageBox.information(self, "Success", f"Loaded attributes ({len(data)} nodes).")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    # ---------------------------------------------------------------
    # UI 狀態更新 (信號回呼)
    # ---------------------------------------------------------------

    def _on_network_loaded(self, name):
        """
        當網路成功載入後更新右側的「Network Info」面板。

        此方法由 manager.network_loaded 信號觸發。
        讀取 manager.G 的基本統計資訊並顯示在標籤中，
        同時啟用所有需要已載入網路才能操作的按鈕。

        參數:
            name: 已載入網路的名稱 (字串)。
        """
        G = self.manager.G
        # 計算連通分量數 (用於判斷是否需要提取 GCC)
        n_components = len(list(nx.connected_components(G)))
        self.lbl_name.setText(f"Name: {name}")
        self.lbl_nodes.setText(f"Nodes (N): {len(G.nodes())}")
        self.lbl_edges.setText(f"Edges (E): {len(G.edges())}")
        self.lbl_components.setText(f"Connected Components: {n_components}")
        # 啟用所有操作按鈕 (載入網路後才有意義的操作)
        self.btn_gcc.setEnabled(True)
        self.btn_save.setEnabled(True)
        self.btn_load_bet.setEnabled(True)
        self.btn_load_clos.setEnabled(True)
        self.btn_load_pos.setEnabled(True)
        self.btn_load_attr.setEnabled(True)

    def _on_network_cleared(self):
        """
        當網路被清除時重設右側的「Network Info」面板。

        此方法由 manager.network_cleared 信號觸發。
        將所有標籤恢復為預設值 ('-')，並停用所有需要已載入網路的按鈕。
        """
        self.lbl_name.setText("Name: -")
        self.lbl_nodes.setText("Nodes (N): -")
        self.lbl_edges.setText("Edges (E): -")
        self.lbl_components.setText("Connected Components: -")
        # 停用所有操作按鈕
        self.btn_gcc.setEnabled(False)
        self.btn_save.setEnabled(False)
        self.btn_load_bet.setEnabled(False)
        self.btn_load_clos.setEnabled(False)
        self.btn_load_pos.setEnabled(False)
        self.btn_load_attr.setEnabled(False)

    # ---------------------------------------------------------------
    # 錯誤處理
    # ---------------------------------------------------------------

    def _on_error(self, msg):
        """
        背景工作執行緒發生錯誤時的回呼函式。

        關閉進度對話框並彈出錯誤訊息框。
        此方法由工作執行緒的 error 信號觸發。

        參數:
            msg: 錯誤訊息字串。
        """
        self._progress.close()
        QMessageBox.critical(self, "Error", msg)
