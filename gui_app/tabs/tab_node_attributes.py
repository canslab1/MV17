"""
Tab 3: 節點屬性 (Node Attributes) - 計算並顯示各種中心性指標。

本分頁對應舊版 measure_node_attribute.py 的步驟四「量測節點屬性」功能，
以及原本從預先計算好的檔案（_tbet.txt / _clos.txt）中載入的
betweenness centrality 與 closeness centrality 計算流程。

在舊版工作流程中：
  - measure_node_attribute() 負責計算 k-core、PageRank、CC（聚集係數）、
    degree（度）、k-core entropy（k-核心熵）、鄰居屬性、MV17 等基礎指標。
  - betweenness / closeness 則是事先由獨立腳本計算完畢後，儲存為
    _tbet.txt 與 _clos.txt 文字檔，再於此步驟中讀取合併。

新版 GUI 將上述所有流程整合至本分頁，中心性指標改為即時計算（real-time），
不再依賴預先計算好的外部檔案。

本分頁提供以下功能按鈕：
  - 「Compute Basic Attributes」: 計算基礎屬性（對應 measure_node_attribute()）
  - 「Compute Betweenness」: 即時計算 betweenness centrality（取代舊版讀取 _tbet.txt）
  - 「Compute Closeness」: 即時計算 closeness centrality（取代舊版讀取 _clos.txt）
  - 「Approx. Betweenness」: 即時計算近似 betweenness centrality（用於大型網路的加速版）
  - 「Compute All」: 先執行基礎屬性計算，再依序計算所有中心性指標
                     （模擬舊版完整工作流程）

表格顯示 12 欄屬性資料，可點擊欄位標頭進行排序。
點擊欄位標頭時，下方繪圖區域會繪製該屬性的直方圖（histogram）。
"""
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTableWidget, QTableWidgetItem, QSplitter, QMessageBox, QHeaderView
)
from PySide6.QtCore import Qt
from core.worker_threads import AttributeComputeWorker, CentralityComputeWorker
from core import algorithm_adapter as algo
from widgets.matplotlib_canvas import PlotWidget
from widgets.progress_dialog import ProgressDialog


# ---------------------------------------------------------------------------
# DISPLAY_ATTRS: 表格中每一欄對應的節點屬性鍵（key）清單。
#
# 這 12 個屬性鍵與 net_attr 字典中各節點的 key 一一對應，
# 決定了表格每一欄要顯示哪個屬性值。順序與 DISPLAY_HEADERS 對齊。
#
# 各屬性說明：
#   'node_id'                 - 節點 ID（唯一識別碼）
#   'node_degree'             - 節點的度（degree，即連接的邊數）
#   'node_k-core'             - k-核心值（k-core number）
#   'node_pagerank'           - PageRank 值
#   'node_cc'                 - 聚集係數（clustering coefficient）
#   'node_k-core-entropy'     - k-核心熵（基於鄰居 k-core 分布的熵值）
#   'node_neighbor-core'      - 鄰居平均 k-core 值
#   'node_neighbor-degree'    - 鄰居平均度（degree）
#   'node_mv17'               - MV17 指標（本研究提出的複合指標）
#   'node_betweenness'        - 介數中心性（betweenness centrality）
#   'node_closeness'          - 接近中心性（closeness centrality）
#   'node_approx-betweenness' - 近似介數中心性（approximate betweenness）
# ---------------------------------------------------------------------------
DISPLAY_ATTRS = [
    'node_id', 'node_degree', 'node_k-core', 'node_pagerank', 'node_cc',
    'node_k-core-entropy', 'node_neighbor-core', 'node_neighbor-degree',
    'node_mv17', 'node_betweenness', 'node_closeness', 'node_approx-betweenness'
]

# ---------------------------------------------------------------------------
# DISPLAY_HEADERS: 表格水平標頭的顯示文字清單。
#
# 與 DISPLAY_ATTRS 一一對應，共 12 欄。
# 使用者在表格中看到的欄位名稱即為此處定義的字串。
# 點擊欄位標頭可觸發該欄的排序，同時在下方繪圖區繪製該屬性的直方圖。
# ---------------------------------------------------------------------------
DISPLAY_HEADERS = [
    'ID', 'Degree', 'K-Core', 'PageRank', 'CC',
    'K-Core Entropy', 'Neighbor Core', 'Neighbor Degree',
    'MV17 (Proposed)', 'Betweenness', 'Closeness', 'Approx. Betw.'
]


class TabNodeAttributes(QWidget):
    """
    節點屬性分頁（Tab 3）。

    此分頁整合了舊版 measure_node_attribute.py 的所有功能：
      1. 計算基礎節點屬性（k-core, PageRank, CC, degree, entropy, 鄰居屬性, MV17）
      2. 計算中心性指標（betweenness, closeness, approx. betweenness）
      3. 以 12 欄表格顯示所有節點的屬性值
      4. 支援欄位排序與直方圖繪製

    屬性：
        manager: NetworkManager 實例，管理網路資料與全域狀態
        _worker: 背景計算執行緒（AttributeComputeWorker 或 CentralityComputeWorker）
    """

    def __init__(self, manager, parent=None):
        """
        初始化節點屬性分頁。

        參數：
            manager: NetworkManager 實例，提供網路圖 (G)、屬性字典 (net_attr)
                     以及信號（network_loaded / attributes_computed / network_cleared）
            parent:  父級 QWidget（可選）
        """
        super().__init__(parent)
        self.manager = manager
        # _worker: 保存目前正在執行的背景執行緒參考，避免被垃圾回收
        self._worker = None
        self._init_ui()

        # --- 連接 manager 的信號（signal）到本分頁的槽函式（slot） ---
        # 當網路載入完成時，啟用所有計算按鈕
        self.manager.network_loaded.connect(self._on_network_loaded)
        # 當屬性計算完成時，重新填充表格
        self.manager.attributes_computed.connect(self._on_attributes_computed)
        # 當網路被清除時，重置 UI 狀態
        self.manager.network_cleared.connect(self._on_cleared)

    def _init_ui(self):
        """
        建構使用者介面（UI）。

        版面配置由上到下分為兩個區域：
          1. 頂部控制列：包含 5 個計算按鈕 + 狀態文字
          2. 垂直分割器（QSplitter）：上方為屬性表格，下方為直方圖繪圖區
        """
        layout = QVBoxLayout(self)

        # ===================================================================
        # 頂部控制列（Controls）：水平排列的按鈕與狀態標籤
        # ===================================================================
        controls = QHBoxLayout()

        # --- 「Compute Basic Attributes」按鈕 ---
        # 對應舊版 measure_node_attribute.py 中的 measure_node_attribute() 函式，
        # 計算以下基礎屬性：
        #   - k-core（k-核心分解）
        #   - PageRank
        #   - CC（聚集係數，clustering coefficient）
        #   - degree（節點的度）
        #   - k-core entropy（k-核心熵）
        #   - neighbor-core / neighbor-degree（鄰居屬性）
        #   - MV17（本研究提出的複合影響力指標）
        self.btn_basic = QPushButton("Compute Basic Attributes")
        self.btn_basic.clicked.connect(self._compute_basic)
        self.btn_basic.setEnabled(False)  # 初始時停用，等網路載入後才啟用
        controls.addWidget(self.btn_basic)

        # --- 「Compute Betweenness」按鈕 ---
        # 對應舊版工作流程中從 _tbet.txt 檔案載入 betweenness centrality 的步驟。
        # 新版改為即時計算，使用 NetworkX 的 betweenness_centrality()。
        self.btn_bet = QPushButton("Compute Betweenness")
        self.btn_bet.clicked.connect(self._compute_betweenness)
        self.btn_bet.setEnabled(False)
        controls.addWidget(self.btn_bet)

        # --- 「Compute Closeness」按鈕 ---
        # 對應舊版工作流程中從 _clos.txt 檔案載入 closeness centrality 的步驟。
        # 新版改為即時計算，使用 NetworkX 的 closeness_centrality()。
        self.btn_clos = QPushButton("Compute Closeness")
        self.btn_clos.clicked.connect(self._compute_closeness)
        self.btn_clos.setEnabled(False)
        controls.addWidget(self.btn_clos)

        # --- 「Approx. Betweenness」按鈕 ---
        # 計算近似介數中心性（approximate betweenness centrality）。
        # 對於大型網路，精確的 betweenness 計算非常耗時（O(VE)），
        # 近似演算法透過隨機取樣部分節點來加速計算。
        # 此功能在舊版中無直接對應，為新版新增的加速選項。
        self.btn_approx_bet = QPushButton("Approx. Betweenness")
        self.btn_approx_bet.clicked.connect(self._compute_approx_betweenness)
        self.btn_approx_bet.setEnabled(False)
        controls.addWidget(self.btn_approx_bet)

        # --- 「Compute All」按鈕 ---
        # 模擬舊版的完整工作流程：
        #   第一步：先執行 measure_node_attribute()（計算基礎屬性）
        #   第二步：再計算 betweenness 與 closeness 中心性
        # 內部先啟動 AttributeComputeWorker，完成後在回呼中
        # 自動接續啟動 CentralityComputeWorker。
        self.btn_all = QPushButton("Compute All")
        self.btn_all.clicked.connect(self._compute_all)
        self.btn_all.setEnabled(False)
        controls.addWidget(self.btn_all)

        # 狀態標籤：顯示目前計算進度或結果摘要
        self.lbl_status = QLabel("")
        controls.addWidget(self.lbl_status)
        controls.addStretch()  # 彈性空間，將按鈕推向左側
        layout.addLayout(controls)

        # ===================================================================
        # 垂直分割器（QSplitter）：上方表格 + 下方直方圖
        # ===================================================================
        splitter = QSplitter(Qt.Vertical)

        # --- 屬性表格（QTableWidget）---
        # 顯示 12 欄屬性資料（對應 DISPLAY_HEADERS），每列為一個節點。
        # 支援點擊欄位標頭排序（setSortingEnabled(True)）。
        # 點擊欄位標頭時，同時觸發 _on_column_clicked()，
        # 在下方繪圖區繪製該欄屬性值的直方圖。
        self.table = QTableWidget()
        self.table.setColumnCount(len(DISPLAY_HEADERS))         # 設定欄數為 12
        self.table.setHorizontalHeaderLabels(DISPLAY_HEADERS)   # 設定欄位標頭文字
        self.table.setSortingEnabled(True)                      # 啟用排序功能
        # 連接欄位標頭點擊信號 → 繪製直方圖
        self.table.horizontalHeader().sectionClicked.connect(self._on_column_clicked)
        # 欄位寬度自動適應內容
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        splitter.addWidget(self.table)

        # --- 直方圖繪圖區（PlotWidget）---
        # 使用 matplotlib 嵌入 Qt 的繪圖元件，用於顯示選定欄位的屬性分布直方圖。
        # figsize=(8, 3) 設定預設圖表尺寸（寬 8 英寸、高 3 英寸）。
        self.plot_widget = PlotWidget(figsize=(8, 3))
        splitter.addWidget(self.plot_widget)

        # 設定分割器中兩個區域的伸展比例：表格佔 3 份，直方圖佔 1 份
        splitter.setStretchFactor(0, 3)  # 表格區域
        splitter.setStretchFactor(1, 1)  # 直方圖區域
        layout.addWidget(splitter)

    # =======================================================================
    # 信號槽函式（Slots）：回應 manager 發出的信號
    # =======================================================================

    def _on_network_loaded(self, name):
        """
        當網路載入完成時觸發（manager.network_loaded 信號）。

        啟用所有計算按鈕，清空表格與狀態文字，
        讓使用者可以開始計算節點屬性。

        參數：
            name: 載入的網路名稱（此處未使用，僅為信號簽名所需）
        """
        self.btn_basic.setEnabled(True)
        self.btn_bet.setEnabled(True)
        self.btn_clos.setEnabled(True)
        self.btn_approx_bet.setEnabled(True)
        self.btn_all.setEnabled(True)
        self.table.setRowCount(0)   # 清空表格列數
        self.lbl_status.setText("")

    def _on_cleared(self):
        """
        當網路被清除時觸發（manager.network_cleared 信號）。

        停用所有計算按鈕，清空表格、直方圖與狀態文字，
        回復到初始狀態。
        """
        self.btn_basic.setEnabled(False)
        self.btn_bet.setEnabled(False)
        self.btn_clos.setEnabled(False)
        self.btn_approx_bet.setEnabled(False)
        self.btn_all.setEnabled(False)
        self.table.setRowCount(0)
        self.plot_widget.clear()
        self.lbl_status.setText("")

    def _on_attributes_computed(self):
        """
        當屬性計算完成時觸發（manager.attributes_computed 信號）。

        呼叫 _populate_table() 重新填充表格，顯示最新的屬性資料。
        """
        self._populate_table()

    # =======================================================================
    # 基礎屬性計算
    # =======================================================================

    def _compute_basic(self):
        """
        計算基礎節點屬性。

        對應舊版 measure_node_attribute.py 中的 measure_node_attribute() 函式。
        使用 AttributeComputeWorker 在背景執行緒中計算以下屬性：
          - k-core（k-核心分解）
          - PageRank
          - CC（聚集係數）
          - degree（節點的度）
          - k-core entropy（k-核心熵，基於鄰居 k-core 分布計算的資訊熵）
          - neighbor-core（鄰居平均 k-core）
          - neighbor-degree（鄰居平均度）
          - MV17（本研究提出的複合影響力指標）

        計算完成後觸發 _on_basic_done() 回呼。
        """
        if not self.manager.has_network():
            return
        # 建立進度對話框，顯示計算進度
        self._progress = ProgressDialog("Computing Attributes...", self)
        # 建立背景計算執行緒，傳入網路圖 G
        self._worker = AttributeComputeWorker(self.manager.G)
        self._progress.set_worker(self._worker)
        # 連接進度更新信號
        self._worker.progress.connect(self._progress.update_progress)
        # 連接完成信號 → 處理基礎屬性計算結果
        self._worker.finished.connect(self._on_basic_done)
        # 連接錯誤信號 → 顯示錯誤訊息
        self._worker.error.connect(self._on_error)
        self._progress.show()
        self._worker.start()

    def _on_basic_done(self, net_attr):
        """
        基礎屬性計算完成後的回呼函式。

        將計算結果合併到 manager 的屬性字典中。
        特別注意：如果先前已經計算過中心性指標（betweenness / closeness），
        需要在新的 net_attr 中保留這些既有的中心性值，避免被覆蓋。

        參數：
            net_attr: dict，計算完成的節點屬性字典。
                      結構為 {node_id: {attr_key: value, ...}, ...}
        """
        self._progress.close()
        # 合併邏輯：若先前已有屬性資料，保留已計算的中心性值
        # （因為基礎屬性計算不包含 betweenness / closeness，
        #  若不保留的話，重新計算基礎屬性會遺失先前的中心性結果）
        if self.manager.net_attr is not None:
            for ni in net_attr:
                if ni in self.manager.net_attr:
                    for key in [algo.NODE_BETWEENNESS, algo.NODE_CLOSENESS]:
                        if key in self.manager.net_attr[ni]:
                            net_attr[ni][key] = self.manager.net_attr[ni][key]
        # 將屬性設定到 manager 中（會觸發 attributes_computed 信號 → 更新表格）
        self.manager.set_attributes(net_attr)
        self.lbl_status.setText(f"Computed attributes for {len(net_attr)} nodes.")

    # =======================================================================
    # 中心性指標計算
    # =======================================================================

    def _compute_betweenness(self):
        """
        計算介數中心性（betweenness centrality）。

        對應舊版工作流程中從 _tbet.txt 檔案載入的步驟。
        新版改為使用 CentralityComputeWorker 即時計算。

        介數中心性衡量一個節點在網路中作為「橋樑」的重要程度，
        計算的是經過該節點的最短路徑佔所有最短路徑的比例。
        """
        self._compute_centrality(True, False, False)

    def _compute_closeness(self):
        """
        計算接近中心性（closeness centrality）。

        對應舊版工作流程中從 _clos.txt 檔案載入的步驟。
        新版改為使用 CentralityComputeWorker 即時計算。

        接近中心性衡量一個節點到網路中其他所有節點的平均距離的倒數，
        值越大表示該節點越「接近」網路中心。
        """
        self._compute_centrality(False, True, False)

    def _compute_approx_betweenness(self):
        """
        計算近似介數中心性（approximate betweenness centrality）。

        使用隨機取樣方法加速計算，適用於大型網路。
        精確的 betweenness 計算時間複雜度為 O(VE)，
        對於數萬節點以上的網路可能需要數小時，
        近似演算法可大幅縮短計算時間。
        """
        self._compute_centrality(False, False, True)

    def _compute_all(self):
        """
        執行完整計算流程：先計算基礎屬性，再計算中心性。

        對應舊版的完整工作流程（模擬 measure_node_attribute.py 的全部步驟）：
          第一步：執行 measure_node_attribute()（基礎屬性）
          第二步：載入 _tbet.txt 與 _clos.txt（中心性指標）

        新版在第一步完成後，透過回呼 _on_basic_done_then_centrality()
        自動接續第二步的中心性即時計算。
        """
        if not self.manager.has_network():
            return
        # 第一步：先計算基礎屬性
        self._progress = ProgressDialog("Computing All Attributes...", self)
        self._worker = AttributeComputeWorker(self.manager.G)
        self._progress.set_worker(self._worker)
        self._worker.progress.connect(self._progress.update_progress)
        # 注意：這裡連接的是 _on_basic_done_then_centrality，
        # 而非一般的 _on_basic_done，因為完成後要接續計算中心性
        self._worker.finished.connect(self._on_basic_done_then_centrality)
        self._worker.error.connect(self._on_error)
        self._progress.show()
        self._worker.start()

    def _on_basic_done_then_centrality(self, net_attr):
        """
        「Compute All」流程中，基礎屬性計算完成後的回呼。

        先將基礎屬性結果儲存到 manager，
        然後自動啟動 betweenness 與 closeness 的中心性計算。

        參數：
            net_attr: dict，基礎屬性計算結果
        """
        self._progress.close()
        # 儲存基礎屬性（此時不需要保留舊的中心性，因為接下來就會重新計算）
        self.manager.set_attributes(net_attr)
        # 第二步：接續計算 betweenness 與 closeness（兩者皆計算）
        self._compute_centrality(True, True)

    def _compute_centrality(self, do_bet, do_clos, do_approx_bet=False):
        """
        啟動中心性計算的通用方法。

        使用 CentralityComputeWorker 在背景執行緒中計算指定的中心性指標。
        此方法被 _compute_betweenness()、_compute_closeness()、
        _compute_approx_betweenness() 以及 _on_basic_done_then_centrality() 呼叫。

        參數：
            do_bet:        bool，是否計算 betweenness centrality
            do_clos:       bool，是否計算 closeness centrality
            do_approx_bet: bool，是否計算近似 betweenness centrality（預設 False）
        """
        if not self.manager.has_network():
            return
        # 建立進度對話框
        self._progress = ProgressDialog("Computing Centrality...", self)
        # 建立中心性計算的背景執行緒
        self._worker = CentralityComputeWorker(
            self.manager.G, do_bet, do_clos,
            compute_approx_betweenness=do_approx_bet)
        self._progress.set_worker(self._worker)
        self._worker.progress.connect(self._progress.update_progress)
        # 完成後呼叫 _on_centrality_done 來合併結果
        self._worker.finished.connect(self._on_centrality_done)
        self._worker.error.connect(self._on_error)
        self._progress.show()
        self._worker.start()

    def _on_centrality_done(self, result):
        """
        中心性計算完成後的回呼函式。

        將計算出的中心性值逐一合併（append）到 manager 的 net_attr 字典中。
        合併使用 algo.append_attribute()，將中心性值寫入各節點的對應屬性鍵。

        參數：
            result: dict，中心性計算結果。可能包含以下鍵：
                    - 'betweenness':       {node_id: value, ...}
                    - 'closeness':         {node_id: value, ...}
                    - 'approx_betweenness': {node_id: value, ...}
        """
        self._progress.close()
        # 檢查是否已有基礎屬性資料（中心性需要附加到已有的屬性字典上）
        if self.manager.net_attr is None:
            QMessageBox.warning(self, "Warning", "Compute basic attributes first.")
            return

        # 將各中心性結果合併到 net_attr 中
        if 'betweenness' in result:
            algo.append_attribute(self.manager.net_attr, result['betweenness'], algo.NODE_BETWEENNESS)
        if 'closeness' in result:
            algo.append_attribute(self.manager.net_attr, result['closeness'], algo.NODE_CLOSENESS)
        if 'approx_betweenness' in result:
            algo.append_attribute(self.manager.net_attr, result['approx_betweenness'], algo.NODE_APPROX_BETWEENNESS)

        # 發送屬性已計算完成的信號，觸發表格更新
        self.manager.attributes_computed.emit()
        self.lbl_status.setText("Centrality computation complete.")

    # =======================================================================
    # 表格填充與顯示
    # =======================================================================

    def _populate_table(self):
        """
        將 manager.net_attr 中的屬性資料填充到表格中。

        表格共 12 欄（由 DISPLAY_ATTRS / DISPLAY_HEADERS 定義），
        每一列代表一個節點。節點依 ID 排序後逐列填入。

        填入時的處理邏輯：
          - 浮點數值：四捨五入至小數點後 6 位顯示
          - 其他值（整數、字串等）：直接顯示
          - 缺失的屬性：顯示為空字串

        注意：填充前先暫時停用排序（setSortingEnabled(False)），
        填充完成後再啟用，避免填充過程中觸發不必要的排序操作。
        """
        if not self.manager.has_attributes():
            return

        net_attr = self.manager.net_attr
        nodes = sorted(net_attr.keys())  # 依節點 ID 排序

        # 暫停排序功能，避免逐列填入時反覆觸發排序
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(nodes))

        for row, ni in enumerate(nodes):
            for col, attr in enumerate(DISPLAY_ATTRS):
                # 取得該節點的該屬性值，若不存在則預設為空字串
                val = net_attr[ni].get(attr, '')
                if isinstance(val, float):
                    # 浮點數使用 Qt.DisplayRole 設定，保持數值排序正確
                    item = QTableWidgetItem()
                    item.setData(Qt.DisplayRole, round(val, 6))
                else:
                    # 整數或字串等直接設定
                    item = QTableWidgetItem()
                    item.setData(Qt.DisplayRole, val)
                # 將儲存格設為唯讀（不可編輯）
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(row, col, item)

        # 重新啟用排序功能
        self.table.setSortingEnabled(True)
        self.lbl_status.setText(f"Showing {len(nodes)} nodes.")

    # =======================================================================
    # 欄位點擊事件：繪製直方圖
    # =======================================================================

    def _on_column_clicked(self, col):
        """
        當使用者點擊表格的欄位標頭時觸發。

        在下方的 PlotWidget 中繪製該欄屬性值的直方圖（histogram），
        用於觀察屬性值的分布情況。

        特殊處理：
          - 第 0 欄（ID）為識別碼，不繪製直方圖
          - 非數值型的屬性值會被過濾掉，僅對 int / float 值進行統計
          - 若該欄沒有任何有效數值，則不繪圖

        參數：
            col: int，被點擊的欄位索引（0-based）
        """
        # ID 欄（第 0 欄）不繪製直方圖
        if col == 0 or not self.manager.has_attributes():
            return

        # 取得對應的屬性鍵
        attr = DISPLAY_ATTRS[col]
        # 收集所有節點中該屬性的數值型值（過濾掉非數值的值）
        values = [self.manager.net_attr[ni].get(attr, 0)
                  for ni in self.manager.net_attr
                  if isinstance(self.manager.net_attr[ni].get(attr, 0), (int, float))]
        if not values:
            return

        # 取得 matplotlib 的 Axes 物件並繪製直方圖
        ax = self.plot_widget.get_axes()
        ax.clear()  # 清除先前的圖表
        ax.hist(values, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
        ax.set_title(f"Distribution of {DISPLAY_HEADERS[col]}")  # 圖表標題
        ax.set_xlabel(DISPLAY_HEADERS[col])                       # X 軸標籤
        ax.set_ylabel("Count")                                    # Y 軸標籤
        self.plot_widget.refresh()  # 刷新繪圖區域以顯示新圖表

    # =======================================================================
    # 錯誤處理
    # =======================================================================

    def _on_error(self, msg):
        """
        背景執行緒發生錯誤時的回呼函式。

        關閉進度對話框並顯示錯誤訊息對話框。

        參數：
            msg: str，錯誤訊息文字
        """
        self._progress.close()
        QMessageBox.critical(self, "Error", msg)
