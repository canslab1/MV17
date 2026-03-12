"""
Tab 5: 統計分析頁籤

本頁籤將多個舊腳本的功能整合到 5 個子頁籤中：

  子頁籤 A「Basic Statistics」(基本統計量):
      對應 code/analysis1.py 中的 network_basic_analysis() 函式，
      計算單一網路的基本拓撲統計量（節點數 N、邊數 E、平均群聚係數 <c>、
      最大度 k_max、平均度 <k>、度平方均值 <k^2>、最大 k-shell ks_max、
      平均 k-shell <ks>、同配係數 r、Shannon 熵 H、傳播閾值 beta_thd 等）。

  子頁籤 B「Scatter Plot」(單一散佈圖):
      從舊版 network_attribute_scatter.py 簡化而來的單一散佈圖。
      使用者可選擇 X 軸與 Y 軸的節點屬性，並控制取樣數量。
      注意：相關係數（Pearson r）是在「全部節點」上計算的（與舊版行為一致），
      但繪圖時僅使用取樣後的資料以提升效能。使用者可透過 Samples 旋轉框
      控制繪圖取樣量。

  子頁籤 C「Scatter Matrix」(散佈圖矩陣):
      忠實重現舊版 code/network_attribute_scatter.py 的 draw_attribute_scatter() 函式：
      - 舊版使用 PLOT_NUM_ROWS=3、PLOT_NUM_COLS=5 的子圖佈局
      - 相關係數在「全部資料」上計算，繪圖也使用「全部資料」
        （對應舊版 draw_plot 第 64 行的行為）
      - 標題格式 'z-axis (correlation): ...' 與舊版程式一致
      - 字體大小 fontsize=10 與舊版程式一致

  子頁籤 D「Propagation Curves」(傳播曲線):
      對應 code/experiment1_draw plot.py，載入並比較不同演算法的傳播曲線。
      可載入多個結果檔案進行疊圖比較。

  子頁籤 E「Batch Analysis」(批次分析):
      對應 code/analysis1.py 的主流程（main workflow），
      批次遍歷資料夾中的多個邊列表檔案（file_name_list），
      對每個網路執行基本統計分析。掃描檔案時會根據後綴名過濾
      跳過輔助檔案（_tbet.txt、_clos.txt、_abet.txt、_pos.txt、_sirr.txt 等），
      僅保留主要的邊列表檔案。
"""
import os
import random
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QGroupBox, QLabel,
    QPushButton, QComboBox, QSpinBox, QTableWidget, QTableWidgetItem,
    QMessageBox, QFileDialog, QHeaderView, QCheckBox
)
from PySide6.QtCore import Qt
from core import algorithm_adapter as algo
from core.worker_threads import BatchAnalysisWorker
from widgets.matplotlib_canvas import PlotWidget
from widgets.progress_dialog import ProgressDialog


# ============================================================================
# 節點屬性選項列表
# 每個元組為 (屬性鍵名, 顯示標籤)，用於散佈圖的 X/Y 軸下拉選單
# 以及散佈圖矩陣的勾選框。
# ============================================================================
ATTR_CHOICES = [
    ('node_degree', 'Degree'),              # 度 (degree)
    ('node_k-core', 'K-Core'),              # k-core 分解值
    ('node_pagerank', 'PageRank'),           # PageRank 中心性
    ('node_cc', 'Clustering Coeff.'),        # 群聚係數
    ('node_k-core-entropy', 'K-Core Entropy'),  # k-core 熵
    ('node_neighbor-core', 'Neighbor Core'),    # 鄰居 core 均值
    ('node_neighbor-degree', 'Neighbor Degree'),# 鄰居度均值
    ('node_mv17', 'MV17 (Proposed)'),        # 提議方法 MV17
    ('node_betweenness', 'Betweenness'),     # 介數中心性
    ('node_closeness', 'Closeness'),         # 接近中心性
]

# ============================================================================
# 基本分析的統計量鍵名與表頭
# 這些鍵名對應 algo.compute_basic_analysis() 回傳字典的鍵。
# ============================================================================
ANALYSIS_KEYS = ['N', 'E', '<c>', 'k_max', '<k>', '<k^2>', 'ks_max', '<ks>', 'r', 'H', 'beta_thd']
ANALYSIS_HEADERS = ['N', 'E', '<c>', 'k_max', '<k>', '<k^2>', 'ks_max', '<ks>', 'r', 'H', 'beta_thd']


class TabStatistics(QWidget):
    """
    統計分析主頁籤。

    包含 5 個子頁籤（Basic Statistics、Scatter Plot、Scatter Matrix、
    Propagation Curves、Batch Analysis），整合了舊版多個分析腳本的功能。

    透過 manager（NetworkManager）取得當前載入的網路資料、節點屬性、
    傳播結果等，並監聽相關訊號以啟用/停用對應的按鈕。
    """

    def __init__(self, manager, parent=None):
        super().__init__(parent)
        self.manager = manager

        # 已載入的傳播曲線資料，鍵為檔案名稱或 'current'，值為傳播結果字典
        self._loaded_curves = {}
        # 批次分析結果字典，鍵為網路名稱，值為統計量字典
        self._batch_result = None
        # 批次分析工作執行緒的參考
        self._batch_worker = None

        self._init_ui()

        # ---- 連接 manager 的訊號 ----
        # 當網路載入完成時，啟用「計算基本統計」按鈕
        self.manager.network_loaded.connect(self._on_network_loaded)
        # 當節點屬性計算完成時，啟用散佈圖相關按鈕
        self.manager.attributes_computed.connect(self._on_attributes_computed)
        # 當傳播模擬完成時，啟用「繪製當前結果」按鈕
        self.manager.propagation_completed.connect(self._on_propagation_completed)
        # 當網路被清除時，重置所有子頁籤的狀態
        self.manager.network_cleared.connect(self._on_cleared)

    # ========================================================================
    # UI 初始化
    # ========================================================================

    def _init_ui(self):
        """初始化主版面：建立 QTabWidget 並加入 5 個子頁籤。"""
        layout = QVBoxLayout(self)

        self.sub_tabs = QTabWidget()
        layout.addWidget(self.sub_tabs)

        # 子頁籤 A：基本統計量（對應 analysis1.py -> network_basic_analysis()）
        self._init_analysis_tab()
        # 子頁籤 B：單一散佈圖（簡化自 network_attribute_scatter.py）
        self._init_scatter_tab()
        # 子頁籤 C：散佈圖矩陣（對應 network_attribute_scatter.py -> draw_attribute_scatter()）
        self._init_scatter_matrix_tab()
        # 子頁籤 D：傳播曲線比較（對應 experiment1_draw plot.py）
        self._init_curves_tab()
        # 子頁籤 E：批次分析（對應 analysis1.py 的主流程批次迴圈）
        self._init_batch_tab()

    # ------------------------------------------------------------------------
    # 子頁籤 A：基本統計量
    # ------------------------------------------------------------------------

    def _init_analysis_tab(self):
        """
        建立「Basic Statistics」子頁籤的 UI。

        包含：
          - 「Compute Basic Statistics」按鈕：觸發 _compute_analysis() 計算統計量
          - 「Save Analysis...」按鈕：將結果儲存為文字檔
          - QTableWidget：以表格形式顯示統計量（N, E, <c>, k_max, ...）
        """
        w = QWidget()
        layout = QVBoxLayout(w)

        # ---- 控制列 ----
        controls = QHBoxLayout()

        self.btn_analyze = QPushButton("Compute Basic Statistics")
        self.btn_analyze.clicked.connect(self._compute_analysis)
        self.btn_analyze.setEnabled(False)  # 尚未載入網路前停用
        controls.addWidget(self.btn_analyze)

        self.btn_save_analysis = QPushButton("Save Analysis...")
        self.btn_save_analysis.clicked.connect(self._save_analysis)
        self.btn_save_analysis.setEnabled(False)  # 尚未計算前停用
        controls.addWidget(self.btn_save_analysis)

        controls.addStretch()
        layout.addLayout(controls)

        # ---- 統計量表格 ----
        self.analysis_table = QTableWidget()
        self.analysis_table.setColumnCount(len(ANALYSIS_HEADERS))
        self.analysis_table.setHorizontalHeaderLabels(ANALYSIS_HEADERS)
        self.analysis_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        layout.addWidget(self.analysis_table)

        self.sub_tabs.addTab(w, "Basic Statistics")

    # ------------------------------------------------------------------------
    # 子頁籤 B：單一散佈圖
    # ------------------------------------------------------------------------

    def _init_scatter_tab(self):
        """
        建立「Scatter Plot」子頁籤的 UI。

        簡化自舊版 network_attribute_scatter.py，只繪製單一散佈圖。

        包含：
          - X 軸屬性下拉選單
          - Y 軸屬性下拉選單
          - Samples 旋轉框：控制繪圖取樣數量（100 ~ 50000，預設 1000）
            注意：相關係數始終在全部節點上計算（與舊版一致），
            此取樣僅用於繪圖以提升大型網路的渲染效能。
          - 「Draw」按鈕：觸發 _draw_scatter()
          - PlotWidget：顯示散佈圖
        """
        w = QWidget()
        layout = QVBoxLayout(w)

        controls = QHBoxLayout()

        # X 軸屬性選擇
        controls.addWidget(QLabel("X-axis:"))
        self.combo_x = QComboBox()
        for key, label in ATTR_CHOICES:
            self.combo_x.addItem(label, key)
        controls.addWidget(self.combo_x)

        # Y 軸屬性選擇
        controls.addWidget(QLabel("Y-axis:"))
        self.combo_y = QComboBox()
        for key, label in ATTR_CHOICES:
            self.combo_y.addItem(label, key)
        self.combo_y.setCurrentIndex(1)  # 預設選第二個屬性，避免與 X 軸相同
        controls.addWidget(self.combo_y)

        # 繪圖取樣數量控制（僅影響繪圖，不影響相關係數計算）
        controls.addWidget(QLabel("Samples:"))
        self.spin_samples = QSpinBox()
        self.spin_samples.setRange(100, 50000)
        self.spin_samples.setValue(1000)
        controls.addWidget(self.spin_samples)

        self.btn_scatter = QPushButton("Draw")
        self.btn_scatter.clicked.connect(self._draw_scatter)
        self.btn_scatter.setEnabled(False)  # 尚未計算屬性前停用
        controls.addWidget(self.btn_scatter)
        controls.addStretch()
        layout.addLayout(controls)

        # 散佈圖畫布
        self.scatter_plot = PlotWidget(figsize=(7, 6))
        layout.addWidget(self.scatter_plot)

        self.sub_tabs.addTab(w, "Scatter Plot")

    # ------------------------------------------------------------------------
    # 子頁籤 C：散佈圖矩陣
    # ------------------------------------------------------------------------

    def _init_scatter_matrix_tab(self):
        """
        建立「Scatter Matrix」子頁籤的 UI。

        忠實重現舊版 network_attribute_scatter.py 的 draw_attribute_scatter() 函式。
        舊版使用 PLOT_NUM_ROWS=3、PLOT_NUM_COLS=5 的多面板子圖佈局，
        將所有選定屬性的兩兩配對繪製在一張大圖上。

        包含：
          - 勾選框列：選擇要納入矩陣的屬性（預設前 6 個）
          - 「Draw Matrix」按鈕：觸發 _draw_scatter_matrix()
          - 「Save...」按鈕：儲存矩陣圖為 PNG 或 PDF
          - PlotWidget：大尺寸畫布（15x8 英吋）用於顯示多面板散佈圖
        """
        w = QWidget()
        layout = QVBoxLayout(w)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Select attributes:"))

        # 為前 6 個屬性建立勾選框（預設全選）
        self.matrix_checks = {}
        for key, label in ATTR_CHOICES[:6]:
            cb = QCheckBox(label)
            cb.setChecked(True)
            self.matrix_checks[key] = cb
            controls.addWidget(cb)

        self.btn_draw_matrix = QPushButton("Draw Matrix")
        self.btn_draw_matrix.clicked.connect(self._draw_scatter_matrix)
        self.btn_draw_matrix.setEnabled(False)  # 尚未計算屬性前停用
        controls.addWidget(self.btn_draw_matrix)

        self.btn_save_matrix = QPushButton("Save...")
        self.btn_save_matrix.clicked.connect(self._save_scatter_matrix)
        self.btn_save_matrix.setEnabled(False)  # 尚未繪圖前停用
        controls.addWidget(self.btn_save_matrix)

        controls.addStretch()
        layout.addLayout(controls)

        # 大尺寸畫布，用於容納多面板散佈圖矩陣
        self.matrix_plot = PlotWidget(figsize=(15, 8))
        layout.addWidget(self.matrix_plot)

        self.sub_tabs.addTab(w, "Scatter Matrix")

    # ------------------------------------------------------------------------
    # 子頁籤 D：傳播曲線比較
    # ------------------------------------------------------------------------

    def _init_curves_tab(self):
        """
        建立「Propagation Curves」子頁籤的 UI。

        對應舊版 code/experiment1_draw plot.py，用於載入並疊圖比較
        不同演算法的傳播模擬結果曲線。

        包含：
          - 「Load Result File...」按鈕：從檔案載入傳播結果
          - 「Draw Current Results」按鈕：繪製當前 manager 中的傳播結果
          - 「Clear」按鈕：清除所有已載入的曲線
          - PlotWidget：顯示傳播曲線圖
        """
        w = QWidget()
        layout = QVBoxLayout(w)

        controls = QHBoxLayout()

        self.btn_load_curve = QPushButton("Load Result File...")
        self.btn_load_curve.clicked.connect(self._load_curve_file)
        controls.addWidget(self.btn_load_curve)

        self.btn_draw_curves = QPushButton("Draw Current Results")
        self.btn_draw_curves.clicked.connect(self._draw_current_curves)
        self.btn_draw_curves.setEnabled(False)  # 尚未執行傳播模擬前停用
        controls.addWidget(self.btn_draw_curves)

        self.btn_clear_curves = QPushButton("Clear")
        self.btn_clear_curves.clicked.connect(self._clear_curves)
        controls.addWidget(self.btn_clear_curves)

        controls.addStretch()
        layout.addLayout(controls)

        # 傳播曲線畫布
        self.curves_plot = PlotWidget(figsize=(7, 5))
        layout.addWidget(self.curves_plot)

        self.sub_tabs.addTab(w, "Propagation Curves")

    # ------------------------------------------------------------------------
    # 子頁籤 E：批次分析
    # ------------------------------------------------------------------------

    def _init_batch_tab(self):
        """
        建立「Batch Analysis」子頁籤的 UI。

        對應 code/analysis1.py 的主流程（main workflow），批次遍歷
        資料夾中的多個邊列表檔案，對每個網路執行 network_basic_analysis()。

        掃描資料夾時會根據後綴名過濾，跳過以下輔助檔案（這些是
        analysis1.py 產生的中間結果檔）：
          - _tbet.txt  (top betweenness 結果)
          - _clos.txt  (closeness 結果)
          - _abet.txt  (另一種 betweenness 結果)
          - _pos.txt   (節點位置檔)
          - _sirr.txt  (SIR 模擬結果)
        僅保留主要的邊列表 .txt 檔案。

        包含：
          - 「Select Edgelist Folder...」按鈕：選擇包含邊列表檔案的資料夾
          - 「Run Batch Analysis」按鈕：啟動批次分析（在背景執行緒中執行）
          - 「Save Results...」按鈕：將批次結果儲存為文字檔
          - 資料夾路徑標籤：顯示已選資料夾與檔案數量
          - QTableWidget：以表格顯示各網路的統計量
        """
        w = QWidget()
        layout = QVBoxLayout(w)

        controls = QHBoxLayout()

        self.btn_batch_select = QPushButton("Select Edgelist Folder...")
        self.btn_batch_select.clicked.connect(self._select_batch_folder)
        controls.addWidget(self.btn_batch_select)

        self.btn_batch_run = QPushButton("Run Batch Analysis")
        self.btn_batch_run.clicked.connect(self._run_batch_analysis)
        self.btn_batch_run.setEnabled(False)  # 尚未選擇資料夾前停用
        controls.addWidget(self.btn_batch_run)

        self.btn_batch_save = QPushButton("Save Results...")
        self.btn_batch_save.clicked.connect(self._save_batch_results)
        self.btn_batch_save.setEnabled(False)  # 尚未產生結果前停用
        controls.addWidget(self.btn_batch_save)

        self.lbl_batch_folder = QLabel("Folder: (not selected)")
        controls.addWidget(self.lbl_batch_folder)
        controls.addStretch()
        layout.addLayout(controls)

        # 批次分析結果表格（第一欄為網路名稱，其餘欄為統計量）
        self.batch_table = QTableWidget()
        batch_headers = ['Network'] + ANALYSIS_HEADERS
        self.batch_table.setColumnCount(len(batch_headers))
        self.batch_table.setHorizontalHeaderLabels(batch_headers)
        self.batch_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        layout.addWidget(self.batch_table)

        self.sub_tabs.addTab(w, "Batch Analysis")

        # 批次分析的內部狀態
        self._batch_folder = None   # 已選擇的資料夾路徑
        self._batch_files = []      # 過濾後的邊列表檔案名稱列表（不含副檔名）

    # ========================================================================
    # 子頁籤 A 操作：基本統計量
    # ========================================================================

    def _compute_analysis(self):
        """
        計算當前網路的基本統計量。

        呼叫 algo.compute_basic_analysis(G) 取得統計量字典，
        然後透過 manager.set_basic_analysis() 儲存結果，
        並在表格中顯示。
        """
        if not self.manager.has_network():
            return
        try:
            result = algo.compute_basic_analysis(self.manager.G)
            self.manager.set_basic_analysis(result)
            self._show_analysis(result)
            self.btn_save_analysis.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _show_analysis(self, result):
        """
        將統計量字典填入 analysis_table 表格。

        只用一列（row=0），每一欄對應 ANALYSIS_KEYS 中的一個統計量。
        表格項目設為唯讀（移除 ItemIsEditable 旗標）。
        """
        self.analysis_table.setRowCount(1)
        for col, key in enumerate(ANALYSIS_KEYS):
            val = result.get(key, '')
            item = QTableWidgetItem()
            item.setData(Qt.DisplayRole, val)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.analysis_table.setItem(0, col, item)

    def _save_analysis(self):
        """
        將基本統計量結果儲存為文字檔。

        使用 algo.write_analysis_result() 寫出，格式為
        {網路名稱: 統計量字典} 的結構。
        """
        if not self.manager.basic_analysis:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Analysis", "", "Text Files (*.txt)")
        if path:
            result = {self.manager.network_name: self.manager.basic_analysis}
            algo.write_analysis_result(path, result)

    # ========================================================================
    # 子頁籤 B 操作：單一散佈圖
    # ========================================================================

    def _draw_scatter(self):
        """
        繪製兩個節點屬性的散佈圖。

        重要行為說明（與舊版 network_attribute_scatter.py 一致）：
          1. 相關係數（Pearson r）是在「全部節點」上計算的，
             確保統計數值準確反映整體網路的屬性關聯。
          2. 繪圖時僅使用「取樣後的資料」（由 spin_samples 控制數量），
             這是為了在大型網路（數萬～數十萬節點）上保持繪圖效能。
          3. 使用者可透過 Samples 旋轉框調整取樣量：
             較大的取樣量能更完整呈現分布但繪圖較慢，
             較小的取樣量繪圖較快但可能遺漏部分分布特徵。
        """
        if not self.manager.has_attributes():
            QMessageBox.warning(self, "Warning", "Compute attributes first.")
            return

        attr_x = self.combo_x.currentData()   # 取得 X 軸屬性的鍵名
        attr_y = self.combo_y.currentData()   # 取得 Y 軸屬性的鍵名
        net_attr = self.manager.net_attr      # 節點屬性字典 {node_id: {attr_key: value}}

        # ---- 在全部節點上計算 Pearson 相關係數（與舊版行為一致）----
        all_nodes = list(net_attr.keys())
        all_x = [net_attr[n].get(attr_x, 0) for n in all_nodes]
        all_y = [net_attr[n].get(attr_y, 0) for n in all_nodes]
        r = 0
        if len(all_x) > 1:
            corr_matrix = np.corrcoef(all_x, all_y)
            r = corr_matrix[0, 1] if not np.isnan(corr_matrix[0, 1]) else 0

        # ---- 為繪圖進行取樣（僅影響視覺呈現，不影響相關係數）----
        nodes = list(net_attr.keys())
        max_samples = self.spin_samples.value()
        if len(nodes) > max_samples:
            nodes = random.sample(nodes, max_samples)

        x_vals = [net_attr[n].get(attr_x, 0) for n in nodes]
        y_vals = [net_attr[n].get(attr_y, 0) for n in nodes]

        if not x_vals or not y_vals:
            return

        # ---- 繪製散佈圖 ----
        ax = self.scatter_plot.get_axes()
        ax.clear()
        ax.scatter(x_vals, y_vals, s=10, alpha=0.5, marker='+', color='b')
        # 設定座標軸範圍，留出少許邊距避免點貼邊
        max_x = max(x_vals) + 0.01 if x_vals else 1
        max_y = max(y_vals) + 0.01 if y_vals else 1
        ax.axis([-0.01, max_x, -0.01, max_y])
        ax.set_xlabel(self.combo_x.currentText())
        ax.set_ylabel(self.combo_y.currentText())
        ax.set_title(f"r = {r:.4f}")  # 標題顯示相關係數
        ax.grid(True, alpha=0.3)
        self.scatter_plot.refresh()

    # ========================================================================
    # 子頁籤 C 操作：散佈圖矩陣
    # ========================================================================

    def _draw_scatter_matrix(self):
        """
        繪製屬性散佈圖矩陣（所有兩兩配對組合）。

        忠實重現舊版 code/network_attribute_scatter.py 的 draw_attribute_scatter()：

          佈局：
            - 舊版使用 PLOT_NUM_COLS=5（每列最多 5 個子圖）
            - 列數根據配對總數自動計算（向上取整）
            - 圖寬固定 15 英吋，圖高隨列數調整（每列 3 英吋）

          資料處理（與舊版完全一致）：
            - 相關係數在「全部資料」上計算（非取樣）
            - 繪圖也使用「全部資料」（對應舊版 draw_plot 第 64 行，
              該行直接將全部節點的屬性值傳入 scatter()，未做任何取樣）

          格式細節（與舊版一致）：
            - 標題格式：'z-axis (correlation): {r值}'
              舊版將相關係數稱為 z-axis，本版保留此命名
            - xlabel / ylabel 字體大小：fontsize=10
            - 標題字體大小：fontsize=10
            - 散佈點：marker='+', color='b', s=10, alpha=0.5

          實作說明：
            - 使用 fig.clf() 清除整個 Figure 後重新建立子圖，
              因為 add_subplot() 會動態建立不同數量的 Axes，
              無法僅靠 ax.clear() 處理
        """
        if not self.manager.has_attributes():
            QMessageBox.warning(self, "Warning", "Compute attributes first.")
            return

        # 取得使用者勾選的屬性
        selected_attrs = [k for k, cb in self.matrix_checks.items() if cb.isChecked()]
        if len(selected_attrs) < 2:
            QMessageBox.warning(self, "Warning", "Select at least 2 attributes.")
            return

        net_attr = self.manager.net_attr
        nodes = list(net_attr.keys())

        # 產生所有兩兩配對（j > i），避免重複與自身配對
        pairs = []
        for i in range(len(selected_attrs)):
            for j in range(len(selected_attrs)):
                if j > i:
                    pairs.append((selected_attrs[i], selected_attrs[j]))

        if not pairs:
            return

        # 子圖佈局：每列 5 個子圖（對應舊版 PLOT_NUM_COLS=5）
        num_plots = len(pairs)
        num_cols = 5
        num_rows = max(1, (num_plots + num_cols - 1) // num_cols)  # 向上取整

        # 清除整個 Figure 並重新建立子圖
        # 注意：這裡必須用 fig.clf() 而非 ax.clear()，
        # 因為每次繪製的子圖數量可能不同（取決於勾選的屬性數量）
        fig = self.matrix_plot.get_figure()
        fig.clf()
        fig.set_size_inches(15, 3 * num_rows)
        self.matrix_plot.canvas.axes = None  # 子圖由 add_subplot 管理，不再使用預設 axes

        for idx, (attr_x, attr_y) in enumerate(pairs):
            ax = fig.add_subplot(num_rows, num_cols, idx + 1)

            # 在全部資料上計算相關係數與繪圖（與舊版行為完全一致）
            # 舊版 draw_plot 第 64 行直接使用全部節點資料繪圖，未做取樣
            all_x = [net_attr[n].get(attr_x, 0) for n in nodes]
            all_y = [net_attr[n].get(attr_y, 0) for n in nodes]
            r = 0
            if len(all_x) > 1:
                corr_val = np.corrcoef(all_x, all_y)[0][1]
                r = round(corr_val, 4) if not np.isnan(corr_val) else 0

            # 座標軸範圍（留出少許邊距）
            max_x = max(all_x) + 0.01 if all_x else 1
            max_y = max(all_y) + 0.01 if all_y else 1

            ax.axis([-0.01, max_x, -0.01, max_y])
            ax.tick_params(axis='both', labelsize=8)
            # 字體大小 fontsize=10 與舊版程式一致
            ax.set_xlabel(attr_x, fontdict={'fontsize': 10})
            ax.set_ylabel(attr_y, fontdict={'fontsize': 10})
            # 標題格式 'z-axis (correlation): ...' 與舊版程式一致
            ax.set_title(f'z-axis (correlation): {r}', fontdict={'fontsize': 10})
            ax.scatter(all_x, all_y, s=10, marker='+', c='b', alpha=0.5)

        fig.tight_layout()
        self.matrix_plot.refresh()
        self.btn_save_matrix.setEnabled(True)

    def _save_scatter_matrix(self):
        """將散佈圖矩陣儲存為圖檔（PNG 或 PDF），解析度 300 DPI。"""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Scatter Matrix", "", "PNG (*.png);;PDF (*.pdf)")
        if path:
            self.matrix_plot.get_figure().savefig(path, dpi=300, bbox_inches='tight')

    # ========================================================================
    # 子頁籤 D 操作：傳播曲線比較
    # ========================================================================

    def _load_curve_file(self):
        """
        從檔案載入傳播模擬結果。

        使用 algo.read_propagation_result() 讀取結果檔案，
        以檔案名稱為鍵存入 _loaded_curves 字典，
        然後重繪所有曲線。

        可多次載入不同檔案進行疊圖比較
        （對應舊版 experiment1_draw plot.py 的比較功能）。
        """
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Propagation Result", "", "Text Files (*.txt)")
        if not path:
            return
        try:
            data = algo.read_propagation_result(path)
            name = os.path.basename(path)
            self._loaded_curves[name] = data
            self._draw_all_curves()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _draw_current_curves(self):
        """
        將 manager 中當前的傳播模擬結果加入曲線圖。

        以 'current' 為鍵存入 _loaded_curves，
        與其他已載入的檔案結果一起疊圖顯示。
        """
        if self.manager.has_propagation():
            self._loaded_curves['current'] = self.manager.propagation_results
            self._draw_all_curves()

    def _clear_curves(self):
        """清除所有已載入的傳播曲線資料並清空畫布。"""
        self._loaded_curves.clear()
        self.curves_plot.clear()

    def _draw_all_curves(self):
        """
        繪製所有已載入的傳播曲線。

        遍歷 _loaded_curves 中的每個來源（檔案或 'current'），
        對其中的每條曲線（以演算法名稱為鍵）使用不同顏色繪製。

        當有多個來源時，圖例標籤會加上來源名稱前綴以區分。
        顏色從 algo.COLOR_LIST 循環取用。
        """
        ax = self.curves_plot.get_axes()
        ax.clear()
        ax.grid(True, alpha=0.3)

        color_idx = 0
        for source_name, data in self._loaded_curves.items():
            for key in sorted(data.keys()):
                color = algo.COLOR_LIST[color_idx % len(algo.COLOR_LIST)]
                # 從屬性鍵名擷取演算法簡稱作為圖例標籤
                # 特殊處理：'node_mv17' 顯示為 'proposed'
                label = key.split('_')[-1] if key != 'node_mv17' else 'proposed'
                # 多來源時加上來源名稱前綴
                if len(self._loaded_curves) > 1:
                    label = f"{source_name}: {label}"
                ax.plot(data[key], color=color, linewidth=1.5, label=label)
                color_idx += 1

        ax.set_xlabel('Time step')
        ax.set_ylabel('Recovered density (R/N)')
        ax.legend(loc='lower right', fontsize='x-small')
        self.curves_plot.refresh()

    # ========================================================================
    # 子頁籤 E 操作：批次分析
    # ========================================================================

    def _select_batch_folder(self):
        """
        選擇包含邊列表檔案的資料夾，並掃描可用的檔案。

        對應 code/analysis1.py 主流程中的 file_name_list 建構邏輯：
        掃描資料夾中所有 .txt 檔案，但跳過以下後綴的輔助檔案
        （這些是 analysis1.py 在分析過程中產生的中間/結果檔案）：
          - _tbet.txt  (top betweenness centrality 排序結果)
          - _clos.txt  (closeness centrality 結果)
          - _abet.txt  (另一種 betweenness 結果)
          - _pos.txt   (節點佈局位置檔)
          - _sirr.txt  (SIR 模擬結果)

        過濾後的檔案名稱（去除副檔名）儲存在 _batch_files 列表中。
        """
        folder = QFileDialog.getExistingDirectory(self, "Select Edgelist Folder")
        if not folder:
            return
        self._batch_folder = folder

        # 掃描檔案並根據後綴過濾
        self._batch_files = []
        suffixes_to_skip = ('_tbet.txt', '_clos.txt', '_abet.txt', '_pos.txt', '_sirr.txt')
        for fname in sorted(os.listdir(folder)):
            if fname.endswith('.txt') and not any(fname.endswith(s) for s in suffixes_to_skip):
                # 儲存不含副檔名的檔案名稱
                self._batch_files.append(os.path.splitext(fname)[0])

        self.lbl_batch_folder.setText(f"Folder: {folder} ({len(self._batch_files)} files)")
        self.btn_batch_run.setEnabled(len(self._batch_files) > 0)

    def _run_batch_analysis(self):
        """
        啟動批次分析。

        使用 BatchAnalysisWorker 在背景執行緒中執行，
        避免阻塞 GUI。顯示進度對話框讓使用者了解進度。
        完成後呼叫 _on_batch_done() 顯示結果。
        """
        if not self._batch_folder or not self._batch_files:
            return

        # 建立並顯示進度對話框
        self._progress = ProgressDialog("Running Batch Analysis...", self)

        # 建立背景工作執行緒
        self._batch_worker = BatchAnalysisWorker(
            self._batch_folder, self._batch_files)
        self._progress.set_worker(self._batch_worker)

        # 連接訊號
        self._batch_worker.progress.connect(self._progress.update_progress)
        self._batch_worker.finished.connect(self._on_batch_done)
        self._batch_worker.error.connect(self._on_batch_error)

        self._progress.show()
        self._batch_worker.start()

    def _on_batch_done(self, result):
        """
        批次分析完成後的回呼。

        將結果填入 batch_table 表格：
          - 第一欄為網路名稱
          - 其餘欄為各項統計量（對應 ANALYSIS_KEYS）
        所有表格項目設為唯讀。
        """
        self._progress.close()
        if not result:
            return
        self._batch_result = result
        self.btn_batch_save.setEnabled(True)

        # 將結果填入表格
        names = sorted(result.keys())
        self.batch_table.setRowCount(len(names))
        for row, name in enumerate(names):
            # 第一欄：網路名稱
            item = QTableWidgetItem()
            item.setData(Qt.DisplayRole, name)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.batch_table.setItem(row, 0, item)
            # 後續欄位：各項統計量
            for col, key in enumerate(ANALYSIS_KEYS):
                val = result[name].get(key, '')
                item = QTableWidgetItem()
                item.setData(Qt.DisplayRole, val)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.batch_table.setItem(row, col + 1, item)

    def _save_batch_results(self):
        """將批次分析結果儲存為文字檔。"""
        if not self._batch_result:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Batch Results", "", "Text Files (*.txt)")
        if path:
            algo.write_analysis_result(path, self._batch_result)

    def _on_batch_error(self, msg):
        """批次分析發生錯誤時的回呼，關閉進度對話框並顯示錯誤訊息。"""
        self._progress.close()
        QMessageBox.critical(self, "Error", msg)

    # ========================================================================
    # 訊號處理器（Signal Handlers）
    # ========================================================================

    def _on_network_loaded(self, name):
        """
        當網路載入完成時觸發。
        啟用「Compute Basic Statistics」按鈕。
        """
        self.btn_analyze.setEnabled(True)

    def _on_attributes_computed(self):
        """
        當節點屬性計算完成時觸發。
        啟用散佈圖和散佈圖矩陣的繪圖按鈕。
        """
        self.btn_scatter.setEnabled(True)
        self.btn_draw_matrix.setEnabled(True)

    def _on_propagation_completed(self):
        """
        當傳播模擬完成時觸發。
        啟用「Draw Current Results」按鈕。
        """
        self.btn_draw_curves.setEnabled(True)

    def _on_cleared(self):
        """
        當網路被清除時觸發，重置所有子頁籤的狀態。

        散佈圖矩陣的清除說明：
            這裡使用 fig.clf()（清除整個 Figure）而非 ax.clear()（清除單一 Axes），
            原因是散佈圖矩陣會透過 fig.add_subplot() 動態建立多個子圖（Axes），
            這些子圖的數量隨使用者勾選的屬性而變化。如果只用 ax.clear()，
            無法移除已建立的多個子圖，會導致殘留的空白子圖框架。

            fig.clf() 會清除 Figure 上的所有 Axes，然後我們重新建立一個
            預設的單一 Axes（fig.add_subplot(111)），並將它指派回
            canvas.axes，使 PlotWidget 回到初始的正常狀態。
        """
        # 停用所有按鈕
        self.btn_analyze.setEnabled(False)
        self.btn_save_analysis.setEnabled(False)
        self.btn_scatter.setEnabled(False)
        self.btn_draw_matrix.setEnabled(False)
        self.btn_save_matrix.setEnabled(False)
        self.btn_draw_curves.setEnabled(False)
        self.btn_batch_save.setEnabled(False)

        # 清除基本統計量表格
        self.analysis_table.setRowCount(0)

        # 清除單一散佈圖
        self.scatter_plot.clear()

        # 清除散佈圖矩陣
        # 必須用 fig.clf() 清除所有動態建立的子圖，再重建預設 Axes
        fig = self.matrix_plot.get_figure()
        fig.clf()
        ax = fig.add_subplot(111)
        self.matrix_plot.canvas.axes = ax
        self.matrix_plot.refresh()

        # 清除傳播曲線
        self._loaded_curves.clear()
        self.curves_plot.clear()

        # 清除批次分析
        self._batch_result = None
        self.batch_table.setRowCount(0)
