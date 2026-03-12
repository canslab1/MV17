"""
主視窗模組 — 以分頁 (Tab) 佈局組織所有功能面板。

本檔案定義了 GUI 應用程式的主視窗 (MainWindow)，是使用者看到的最上層容器。
舊版程式沒有圖形介面，使用者需要依序在命令列中執行多個 Python 腳本來完成分析流程。
新版 GUI 將這些步驟整合為 5 個分頁，對應舊版的工作流程：

    分頁 1 — Network I/O（載入網路）：
        對應舊版的「讀取 edgelist 檔案」步驟。
        讓使用者瀏覽並載入邊列表檔案，建立 NetworkX 圖物件。

    分頁 2 — Visualization（視覺化）：
        對應舊版的「繪製網路圖」步驟。
        將載入的網路以圖形方式顯示，支援不同的佈局演算法。

    分頁 3 — Node Attributes（計算節點屬性）：
        對應舊版的「計算中心性指標」步驟。
        計算各種節點重要性指標（如度數中心性、介數中心性、
        MV17 指標等），並將結果顯示在表格中。

    分頁 4 — SIR Experiment（執行 SIR 傳播模擬）：
        對應舊版的「執行 SIR 模型」步驟。
        設定傳播參數（感染率 beta、復原率 mu）並執行蒙地卡羅模擬。

    分頁 5 — Statistics（統計分析）：
        對應舊版的「分析模擬結果」步驟。
        對模擬結果進行統計分析、繪製圖表、比較不同指標的預測能力。

架構說明：
    MainWindow 持有一個 NetworkManager 實例作為共享的資料核心。
    所有分頁都透過 NetworkManager 來存取和修改網路資料，
    並透過 Qt Signal/Slot 機制來接收資料變更通知。
    這樣的設計讓各分頁之間解耦，同時又能保持資料一致性。
"""
import os
from PySide6.QtWidgets import QMainWindow, QTabWidget, QMenuBar, QStatusBar, QMessageBox
from PySide6.QtGui import QAction

from core.network_manager import NetworkManager
from tabs.tab_network_io import TabNetworkIO
from tabs.tab_network_viz import TabNetworkViz
from tabs.tab_node_attributes import TabNodeAttributes
from tabs.tab_sir_experiment import TabSIRExperiment
from tabs.tab_statistics import TabStatistics


class MainWindow(QMainWindow):
    """
    應用程式主視窗。

    繼承自 QMainWindow，提供選單列、狀態列和分頁式中央元件。
    負責：
    - 建立並管理 NetworkManager（共享資料層）
    - 建立 5 個功能分頁並加入 QTabWidget
    - 建立選單列（File / Help）
    - 連接 NetworkManager 的信號到狀態列更新

    參數：
        project_root (str): 專案根目錄的絕對路徑，用於定位 edgelist/ 等資料目錄
        parent (QWidget, optional): 父元件，預設為 None（頂層視窗）
    """

    def __init__(self, project_root, parent=None):
        """
        初始化主視窗。

        步驟：
        1. 設定視窗標題和最小尺寸
        2. 建立 NetworkManager 並設定專案根目錄
        3. 建立 5 個分頁，每個分頁接收 NetworkManager 的參考
        4. 初始化檔案樹（掃描 edgelist/ 目錄）
        5. 建立選單列和狀態列
        6. 連接信號與槽
        """
        super().__init__(parent)
        self.project_root = project_root
        self.setWindowTitle("Network Spreader Analysis Tool")
        self.setMinimumSize(1100, 750)

        # ---------------------------------------------------------------
        # 共享資料層：NetworkManager
        # ---------------------------------------------------------------
        # NetworkManager 是所有分頁共用的資料核心，
        # 包含當前載入的網路圖 (G)、節點屬性、模擬結果等。
        # 它也提供 Qt 信號，當資料發生變化時通知所有分頁更新。
        self.manager = NetworkManager(self)
        self.manager.project_root = project_root

        # ---------------------------------------------------------------
        # 分頁元件 (QTabWidget)
        # ---------------------------------------------------------------
        # QTabWidget 是中央元件，包含 5 個分頁。
        # 每個分頁是一個獨立的 QWidget 子類別，負責特定的功能。
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)

        # 建立各分頁，傳入共享的 NetworkManager
        self.tab_io = TabNetworkIO(self.manager)          # 分頁 1：載入/儲存網路
        self.tab_viz = TabNetworkViz(self.manager)        # 分頁 2：網路視覺化
        self.tab_attr = TabNodeAttributes(self.manager)   # 分頁 3：節點屬性計算
        self.tab_sir = TabSIRExperiment(self.manager)     # 分頁 4：SIR 傳播模擬
        self.tab_stats = TabStatistics(self.manager)      # 分頁 5：統計分析

        # 將分頁加入 QTabWidget，標籤文字說明各分頁的功能
        self.tab_widget.addTab(self.tab_io, "Network I/O")
        self.tab_widget.addTab(self.tab_viz, "Visualization")
        self.tab_widget.addTab(self.tab_attr, "Node Attributes")
        self.tab_widget.addTab(self.tab_sir, "SIR Experiment")
        self.tab_widget.addTab(self.tab_stats, "Statistics")

        # ---------------------------------------------------------------
        # 初始化檔案樹
        # ---------------------------------------------------------------
        # 掃描 edgelist/ 目錄，將找到的邊列表檔案顯示在
        # Network I/O 分頁的檔案樹中，方便使用者快速選取
        edgelist_dir = os.path.join(project_root, 'edgelist')
        self.tab_io.populate_file_tree(edgelist_dir)

        # ---------------------------------------------------------------
        # 選單列
        # ---------------------------------------------------------------
        self._create_menu()

        # ---------------------------------------------------------------
        # 狀態列
        # ---------------------------------------------------------------
        # 狀態列位於視窗底部，顯示目前操作狀態的簡短訊息
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready.")

        # ---------------------------------------------------------------
        # 信號連接
        # ---------------------------------------------------------------
        # 將 NetworkManager 的信號連接到狀態列更新，
        # 讓使用者知道目前的操作進度
        self.manager.network_loaded.connect(self._on_network_loaded)
        self.manager.network_cleared.connect(lambda: self.status_bar.showMessage("Network cleared."))
        self.manager.attributes_computed.connect(
            lambda: self.status_bar.showMessage("Attributes computed."))
        self.manager.propagation_completed.connect(
            lambda: self.status_bar.showMessage("SIR simulation completed."))

    def _create_menu(self):
        """
        建立選單列。

        包含兩個選單：
        - File 選單：
            - Open Edgelist...：開啟檔案對話框，讓使用者選擇邊列表檔案
              （委派給 TabNetworkIO 的 _browse_file 方法）
            - Exit：關閉應用程式
        - Help 選單：
            - About：顯示關於對話框，說明本工具的學術出處
        """
        menu_bar = self.menuBar()

        # File 選單
        file_menu = menu_bar.addMenu("File")

        open_action = QAction("Open Edgelist...", self)
        open_action.triggered.connect(self.tab_io._browse_file)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Help 選單
        help_menu = menu_bar.addMenu("Help")
        about_action = QAction("About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _on_network_loaded(self, name):
        """
        當網路載入完成時的回呼函式。

        在狀態列顯示載入的網路名稱、節點數 (N) 和邊數 (E)，
        讓使用者確認載入是否正確。

        參數：
            name (str): 載入的網路名稱（通常是檔案名稱）
        """
        G = self.manager.G
        self.status_bar.showMessage(
            f"Loaded: {name} (N={len(G.nodes())}, E={len(G.edges())})")

    def _show_about(self):
        """
        顯示「關於」對話框。

        說明本工具的學術出處（Fu 等人在 Physica A 上發表的論文），
        以及所使用的主要技術框架。
        """
        QMessageBox.about(
            self, "About",
            "Network Spreader Analysis Tool\n\n"
            "Based on: Fu et al., Physica A 433 (2015) 344-355\n"
            "\"Using Global Diversity and Local Topology Features\n"
            "to Identify Influential Network Spreaders\"\n\n"
            "Built with PySide6 + NetworkX + Matplotlib")
