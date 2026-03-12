"""
Tab 2: 網路視覺化 (Network Visualization)

本分頁提供互動式網路圖繪製功能，是全新的獨立功能模組。

舊版程式（experiment1.py）僅具備以下兩種圖表：
  1. 節點屬性散佈圖（Attribute Scatter Plots）
  2. SIR 傳播模擬圖（SIR Propagation Plots）
舊版並沒有獨立的網路拓撲視覺化頁面。

本分頁是新增功能，讓使用者可以在 GUI 中互動式地探索網路結構：
  - 依據節點屬性（如 degree、pagerank、k-core 等）映射節點顏色與大小
  - 標記 Top-K 重要節點（對應 experiment1.py 中的 retrieve_topk_nodes 概念）
  - 支援大型網路的取樣繪圖（當節點數 > 2000 時自動取樣）
  - 提供 K-Core 視覺化，以 k-core number 為節點著色
"""
import networkx as nx
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QSpinBox,
    QLabel, QPushButton, QCheckBox, QMessageBox
)
from widgets.matplotlib_canvas import PlotWidget
from core import algorithm_adapter as algo


class TabNetworkViz(QWidget):
    """網路視覺化分頁元件。

    負責繪製網路拓撲圖，支援以下功能：
      - 按屬性著色（Color by）與按屬性調整大小（Size by）
      - Top-K 節點高亮標記
      - 大型網路取樣繪製
      - K-Core 分解視覺化
    """

    def __init__(self, manager, parent=None):
        """初始化網路視覺化分頁。

        參數:
            manager: 全域狀態管理器（NetworkManager），持有目前載入的網路 G、
                     佈局座標 pos、節點屬性 net_attr 等共享資料。
            parent:  父級 Qt 元件，預設為 None。
        """
        super().__init__(parent)
        self.manager = manager
        self._init_ui()

        # 連接管理器的訊號（Signal），讓本分頁在網路載入、屬性計算、清除時做出回應
        self.manager.network_loaded.connect(self._on_network_loaded)       # 網路載入完成時觸發
        self.manager.attributes_computed.connect(self._on_attributes_computed)  # 節點屬性計算完成時觸發
        self.manager.network_cleared.connect(self._on_cleared)             # 網路被清除時觸發

    # ──────────────────────────────────────────────
    #  UI 初始化
    # ──────────────────────────────────────────────

    def _init_ui(self):
        """建立本分頁的所有 UI 元件，包括工具列與繪圖區域。"""
        layout = QVBoxLayout(self)

        # ── 工具列 (Toolbar) ─────────────────────
        toolbar = QHBoxLayout()

        # 「Color by」下拉選單：選擇要用哪個節點屬性來映射節點顏色。
        # 例如選擇 node_pagerank，則 pagerank 值越高的節點顏色越深（使用 YlOrRd 色階）。
        toolbar.addWidget(QLabel("Color by:"))
        self.combo_color = QComboBox()
        self.combo_color.addItem("None")       # 預設為 None，表示所有節點使用統一顏色
        self.combo_color.setEnabled(False)      # 在屬性尚未計算前禁用
        toolbar.addWidget(self.combo_color)

        # 「Size by」下拉選單：選擇要用哪個節點屬性來映射節點大小。
        # 例如選擇 node_degree，則 degree 越大的節點繪製時圓圈越大。
        toolbar.addWidget(QLabel("Size by:"))
        self.combo_size = QComboBox()
        self.combo_size.addItem("None")        # 預設為 None，表示所有節點使用統一大小
        self.combo_size.setEnabled(False)       # 在屬性尚未計算前禁用
        toolbar.addWidget(self.combo_size)

        # 「Top-K」數值輸入框：指定要高亮標記的前 K 個重要節點。
        # 此功能對應 experiment1.py 中的 retrieve_topk_nodes 概念：
        # 根據所選 Color-by 屬性排序，取出排名前 K 的節點，以紅色圓圈 + 黑色邊框標記。
        # 設定為 0 表示不進行 Top-K 高亮。
        toolbar.addWidget(QLabel("Top-K:"))
        self.spin_topk = QSpinBox()
        self.spin_topk.setRange(0, 9999)       # 範圍 0~9999，0 代表關閉此功能
        self.spin_topk.setValue(0)              # 預設不高亮任何節點
        toolbar.addWidget(self.spin_topk)

        # 「Labels」核取方塊：是否在節點旁顯示節點標籤（節點 ID）。
        # 僅在節點數 <= 500 時才會實際繪製標籤，避免畫面過於擁擠。
        self.chk_labels = QCheckBox("Labels")
        toolbar.addWidget(self.chk_labels)

        # 「Sample (N>2000)」核取方塊：大型網路取樣功能。
        # 這是新增功能，當網路節點數超過 2000 時，僅取前 2000 個節點的子圖進行繪製，
        # 以避免繪圖時間過長或記憶體不足。預設為勾選（啟用取樣）。
        self.chk_sample = QCheckBox("Sample (N>2000)")
        self.chk_sample.setChecked(True)        # 預設啟用取樣
        toolbar.addWidget(self.chk_sample)

        # 「Layout iters」數值輸入框：Spring Layout（彈簧佈局）的迭代次數。
        # 迭代次數越多，佈局越精細（節點位置越趨於穩定），但計算時間也越長。
        # 範圍 10~5000，預設 50 次。K-Core Viz 按鈕也使用此參數。
        toolbar.addWidget(QLabel("Layout iters:"))
        self.spin_layout_iters = QSpinBox()
        self.spin_layout_iters.setRange(10, 5000)
        self.spin_layout_iters.setValue(50)
        toolbar.addWidget(self.spin_layout_iters)

        # 「Draw」按鈕：觸發一般網路圖的繪製
        self.btn_draw = QPushButton("Draw")
        self.btn_draw.clicked.connect(self._draw_network)
        self.btn_draw.setEnabled(False)         # 在網路尚未載入前禁用
        toolbar.addWidget(self.btn_draw)

        # 「K-Core Viz」按鈕：觸發 K-Core 視覺化繪製。
        # K-Core 視覺化是新增功能，利用 nx.core_number() 計算每個節點的 k-core 數值，
        # 並以色階映射呈現，方便觀察網路的核心-邊緣結構。
        self.btn_kcore_viz = QPushButton("K-Core Viz")
        self.btn_kcore_viz.clicked.connect(self._draw_kcore_network)
        self.btn_kcore_viz.setEnabled(False)    # 在網路尚未載入前禁用
        self.btn_kcore_viz.setToolTip(
            "Draw network with nodes colored by k-core number "
            "and a high-iteration spring layout.")
        toolbar.addWidget(self.btn_kcore_viz)

        toolbar.addStretch()                    # 彈性空白，將按鈕靠左排列
        layout.addLayout(toolbar)

        # ── 繪圖區域 (Plot Area) ─────────────────
        # PlotWidget 封裝了 matplotlib 的 FigureCanvas，提供嵌入 Qt 介面的繪圖功能。
        self.plot_widget = PlotWidget(figsize=(8, 7))
        layout.addWidget(self.plot_widget)

    # ──────────────────────────────────────────────
    #  下拉選單填充
    # ──────────────────────────────────────────────

    def _populate_combos(self):
        """填充 Color-by 與 Size-by 下拉選單的可用屬性清單。

        當 manager 發出 attributes_computed 訊號後呼叫此方法。
        支援的屬性包括 degree、k-core、pagerank、聚集係數(cc)、
        k-core-entropy、neighbor-core、neighbor-degree、mv17、
        betweenness、closeness 等。
        """
        attrs = ['None', 'node_degree', 'node_k-core', 'node_pagerank',
                 'node_cc', 'node_k-core-entropy', 'node_neighbor-core',
                 'node_neighbor-degree', 'node_mv17',
                 'node_betweenness', 'node_closeness']
        for combo in [self.combo_color, self.combo_size]:
            combo.clear()                       # 清空舊選項
            combo.addItems(attrs)               # 填入新選項
            combo.setEnabled(True)              # 啟用下拉選單

    # ──────────────────────────────────────────────
    #  訊號處理 (Signal Slots)
    # ──────────────────────────────────────────────

    def _on_network_loaded(self, name):
        """網路載入完成時的回呼函式。

        參數:
            name: 載入的網路名稱（如檔案名稱）。

        啟用 Draw 與 K-Core Viz 按鈕，但暫時禁用 Color/Size 下拉選單
        （因為屬性尚未計算），並自動執行一次基本繪圖。
        """
        self.btn_draw.setEnabled(True)
        self.btn_kcore_viz.setEnabled(True)
        self.combo_color.setEnabled(False)      # 屬性尚未計算，先禁用
        self.combo_size.setEnabled(False)
        self._draw_network()                    # 自動繪製一次基本網路圖

    def _on_attributes_computed(self):
        """節點屬性計算完成時的回呼函式。

        填充 Color-by / Size-by 下拉選單，讓使用者可以選擇屬性進行視覺映射。
        """
        self._populate_combos()

    def _on_cleared(self):
        """網路被清除時的回呼函式。

        禁用所有操作按鈕與下拉選單，並清空繪圖區域。
        """
        self.btn_draw.setEnabled(False)
        self.btn_kcore_viz.setEnabled(False)
        self.combo_color.setEnabled(False)
        self.combo_size.setEnabled(False)
        self.plot_widget.clear()

    # ──────────────────────────────────────────────
    #  一般網路繪圖
    # ──────────────────────────────────────────────

    def _draw_network(self):
        """繪製一般網路拓撲圖。

        此方法實作以下功能：

        1. 佈局計算：
           使用 nx.spring_layout()（Fruchterman-Reingold 力導向演算法）計算節點位置。
           若 manager.pos 已快取則直接使用，避免重複計算。
           迭代次數由 spin_layout_iters 控制。

        2. 大型網路取樣（新增功能）：
           當節點數 N > 2000 且使用者勾選了「Sample (N>2000)」時，
           僅取前 2000 個節點的子圖（subgraph）進行繪製。
           這是舊版 experiment1.py 沒有的功能，旨在處理大規模網路時的效能問題。

        3. 節點著色（Color by）：
           透過 Color-by 下拉選單，使用者可選擇一個節點屬性（如 node_pagerank），
           將其數值正規化至 [0, 1] 區間後，用 YlOrRd 色階（黃-橘-紅）映射為顏色。
           若選擇 None，則所有節點使用統一的 steelblue 顏色。

        4. 節點大小（Size by）：
           透過 Size-by 下拉選單，使用者可選擇一個節點屬性，
           將其數值正規化後映射為節點大小（範圍 30~330 像素）。
           若選擇 None，則所有節點使用統一大小 30。

        5. Top-K 高亮標記：
           此功能對應 experiment1.py 中的 retrieve_topk_nodes 概念。
           當 Top-K > 0 且已選擇 Color-by 屬性時，呼叫
           algo.retrieve_topk_nodes() 取得排名前 K 的節點，
           以紅色大圓圈 + 黑色粗邊框覆蓋繪製，使其在圖中格外醒目。

        6. fig.clf() 清理模式：
           這裡使用 fig.clf()（清除整個 Figure）而非 ax.clear()（僅清除 Axes），
           這是因為若先前的繪圖包含 colorbar，ax.clear() 無法正確移除 colorbar
           所佔用的額外 Axes，會導致 colorbar 殘留或佈局錯亂。
           fig.clf() 可以徹底清除所有子圖與 colorbar，再重新建立 subplot。
        """
        G = self.manager.G
        if G is None:
            return

        # ── 計算或取得快取的佈局座標 ──
        pos = self.manager.pos
        if pos is None:
            # 尚未計算過佈局，使用 spring_layout 計算並快取到 manager
            pos = nx.spring_layout(G, iterations=self.spin_layout_iters.value())
            self.manager.pos = pos

        # ── 大型網路取樣（新增功能） ──
        # 當節點數超過 2000 且取樣選項被勾選時，取前 2000 個節點進行繪製。
        # 這可以大幅降低繪圖時間與記憶體使用量，適用於大規模網路的快速預覽。
        draw_G = G
        draw_pos = pos
        n = len(G.nodes())
        if n > 2000 and self.chk_sample.isChecked():
            nodes = list(G.nodes())
            sample_nodes = sorted(nodes[:2000])             # 取前 2000 個節點並排序
            draw_G = G.subgraph(sample_nodes)               # 建立子圖
            draw_pos = {k: v for k, v in pos.items() if k in sample_nodes}  # 過濾對應的座標

        # ── 使用 fig.clf() 清除整個 Figure ──
        # 不使用 ax.clear() 是因為當前一次繪圖包含 colorbar 時，
        # ax.clear() 無法清除 colorbar 所建立的額外 Axes，
        # 會導致殘留的 colorbar 或佈局異常。fig.clf() 能徹底清除一切。
        fig = self.plot_widget.get_figure()
        fig.clf()                                           # 清除整個 Figure（含所有 Axes 與 colorbar）
        ax = fig.add_subplot(111)                           # 重新建立單一子圖
        self.plot_widget.canvas.axes = ax                   # 更新 canvas 的 axes 參照

        # ── 節點顏色映射 ──
        # 從 Color-by 下拉選單讀取使用者選擇的屬性，
        # 將屬性值正規化至 [0, 1] 範圍，用於 YlOrRd 色階映射。
        color_attr = self.combo_color.currentText()
        node_colors = 'steelblue'                           # 預設：統一使用 steelblue 顏色
        if color_attr != 'None' and self.manager.has_attributes():
            vals = []
            for ni in draw_G.nodes():
                v = self.manager.net_attr.get(ni, {}).get(color_attr, 0)
                vals.append(v)
            if vals:
                vmin, vmax = min(vals), max(vals)
                if vmax > vmin:
                    # 正規化至 [0, 1]，讓 matplotlib 的 cmap 進行色階映射
                    node_colors = [(v - vmin) / (vmax - vmin) for v in vals]
                else:
                    # 所有值相同時，統一設定為中間值 0.5
                    node_colors = [0.5] * len(vals)

        # ── 節點大小映射 ──
        # 從 Size-by 下拉選單讀取使用者選擇的屬性，
        # 將屬性值正規化後映射至 [30, 330] 像素範圍的節點大小。
        size_attr = self.combo_size.currentText()
        node_sizes = 30                                     # 預設：統一大小 30
        if size_attr != 'None' and self.manager.has_attributes():
            vals = []
            for ni in draw_G.nodes():
                v = self.manager.net_attr.get(ni, {}).get(size_attr, 0)
                vals.append(v)
            if vals:
                vmin, vmax = min(vals), max(vals)
                if vmax > vmin:
                    # 映射至 [30, 330]：最小值對應 30，最大值對應 330
                    node_sizes = [30 + 300 * (v - vmin) / (vmax - vmin) for v in vals]
                else:
                    # 所有值相同時，統一設定為中間大小 60
                    node_sizes = [60] * len(vals)

        # ── 繪製邊 ──
        # 邊使用低透明度（alpha=0.2）與細線寬（0.5），避免喧賓奪主
        nx.draw_networkx_edges(draw_G, draw_pos, ax=ax, alpha=0.2, edge_color='gray', width=0.5)

        # ── 繪製節點 ──
        # 根據 node_colors 的型別決定繪製方式：
        #   - list：表示已映射屬性，使用 YlOrRd 色階
        #   - str（'steelblue'）：表示使用統一顏色
        if isinstance(node_colors, list):
            nx.draw_networkx_nodes(draw_G, draw_pos, ax=ax,
                                   node_size=node_sizes, node_color=node_colors,
                                   cmap='YlOrRd', alpha=0.8)
        else:
            nx.draw_networkx_nodes(draw_G, draw_pos, ax=ax,
                                   node_size=node_sizes, node_color=node_colors, alpha=0.8)

        # ── Top-K 高亮標記 ──
        # 此功能對應 experiment1.py 中的 retrieve_topk_nodes 概念：
        # 根據 Color-by 所選屬性，取出數值排名前 K 的節點，
        # 以紅色填充 + 黑色粗邊框（linewidths=2）覆蓋繪製，讓這些關鍵節點一目了然。
        top_k = self.spin_topk.value()
        if top_k > 0 and self.manager.has_attributes() and color_attr != 'None':
            topk_nodes = algo.retrieve_topk_nodes(self.manager.net_attr, color_attr, top_k)
            topk_in_draw = [n for n in topk_nodes if n in draw_G.nodes()]
            if topk_in_draw:
                nx.draw_networkx_nodes(draw_G, draw_pos, nodelist=topk_in_draw, ax=ax,
                                       node_size=200, node_color='red',
                                       edgecolors='black', linewidths=2)

        # ── 繪製節點標籤 ──
        # 僅在使用者勾選 Labels 且節點數 <= 500 時才繪製，避免密集標籤造成畫面混亂
        if self.chk_labels.isChecked() and len(draw_G.nodes()) <= 500:
            nx.draw_networkx_labels(draw_G, draw_pos, ax=ax, font_size=6)

        # ── 標題與軸設定 ──
        ax.set_title(f"{self.manager.network_name} (N={len(G.nodes())}, E={len(G.edges())})")
        ax.axis('off')                                      # 隱藏坐標軸
        self.plot_widget.refresh()                          # 觸發畫布重繪

    # ──────────────────────────────────────────────
    #  K-Core 視覺化
    # ──────────────────────────────────────────────

    def _draw_kcore_network(self):
        """K-Core 分解視覺化：依據 k-core number 為節點著色。

        K-Core 視覺化是新增功能，舊版 experiment1.py 沒有這項獨立的圖形展示。
        此方法使用 nx.core_number() 計算每個節點的 k-core 數值（即該節點所屬的
        最大 k-core 子圖的 k 值），然後以 YlOrRd 色階將 k-core 值映射為顏色，
        並附帶 colorbar 作為色階圖例。

        主要特性：
          - 使用 nx.core_number() 計算 k-core 數值：k-core 是圖論中的分解方法，
            k-core 值越高表示節點處於網路越核心的位置。
          - Spring Layout 迭代次數可設定：透過 spin_layout_iters 控制
            Fruchterman-Reingold 力導向佈局的迭代次數。較高的迭代次數能讓佈局
            更精緻，但計算時間也更長，適合用於仔細觀察 k-core 結構。
          - 包含 colorbar（色階圖例）：繪圖時會在圖的右側加入 colorbar，
            顯示 k-core 數值與顏色的對應關係。因為 colorbar 會在 Figure 中
            建立額外的 Axes，所以重繪時必須使用 fig.clf() 而非 ax.clear()，
            否則 colorbar 會殘留或疊加。
          - N > 5000 效能警告：當節點數超過 5000 時，會彈出確認對話框提醒使用者
            K-Core 視覺化搭配高迭代次數的 spring layout 可能會很慢，
            讓使用者決定是否繼續。
        """
        G = self.manager.G
        if G is None:
            return

        # ── N > 5000 效能警告對話框 ──
        # 大型網路搭配高迭代次數的 spring layout 可能需要較長計算時間，
        # 因此在節點數超過 5000 時顯示警告，讓使用者確認是否繼續。
        n = len(G.nodes())
        if n > 5000:
            reply = QMessageBox.question(
                self, "Confirm",
                f"Network has {n} nodes. K-Core visualization with "
                f"high-iteration layout may be slow. Continue?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply != QMessageBox.Yes:
                return

        # ── 計算 Spring Layout 佈局 ──
        # 這裡每次都重新計算佈局（不使用 manager.pos 快取），
        # 因為 K-Core Viz 通常搭配較高的迭代次數，使用者期望得到更精緻的佈局。
        iterations = self.spin_layout_iters.value()
        pos = nx.spring_layout(G, iterations=iterations)

        # ── 計算 K-Core 數值 ──
        # nx.core_number(G) 回傳字典 {節點: k-core 數值}，
        # k-core 數值表示該節點所屬的最大 k-core 子圖的 k 值。
        # 例如 core_number=3 表示該節點屬於 3-core（每個節點度數 >= 3 的最大子圖）。
        core_numbers = nx.core_number(G)
        core_vals = [core_numbers[n] for n in G.nodes()]    # 按節點順序取出 k-core 值列表
        max_core = max(core_vals) if core_vals else 1       # 最大 k-core 值，用於色階範圍

        # ── 使用 fig.clf() 清除整個 Figure ──
        # K-Core 視覺化會建立 colorbar，而 colorbar 會在 Figure 中新增額外的 Axes。
        # 如果使用 ax.clear() 僅清除主 Axes，colorbar 的 Axes 會殘留，
        # 導致下次繪圖時出現多餘的 colorbar 或佈局異常。
        # 因此必須使用 fig.clf() 徹底清除整個 Figure，再重新建立 subplot。
        fig = self.plot_widget.get_figure()
        fig.clf()                                           # 清除整個 Figure（含 colorbar 的 Axes）
        ax = fig.add_subplot(111)                           # 重新建立子圖
        self.plot_widget.canvas.axes = ax                   # 更新 canvas 的 axes 參照

        # ── 繪製邊 ──
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.2, edge_color='gray', width=0.5)

        # ── 繪製節點（依 k-core 值著色） ──
        # 使用 YlOrRd 色階，vmin=0、vmax=max_core，讓色階完整覆蓋 k-core 值範圍。
        # sc 是 PathCollection 物件，後續傳給 fig.colorbar() 以建立色階圖例。
        sc = nx.draw_networkx_nodes(
            G, pos, ax=ax, node_color=core_vals, cmap='YlOrRd',
            node_size=50, alpha=0.9, vmin=0, vmax=max_core)

        # ── 繪製節點標籤 ──
        if self.chk_labels.isChecked() and n <= 500:
            nx.draw_networkx_labels(G, pos, ax=ax, font_size=6)

        # ── 標題 ──
        # 顯示網路名稱、最大 k-core 值、使用的佈局迭代次數
        ax.set_title(
            f"K-Core Visualization: {self.manager.network_name} "
            f"(max k-core={max_core}, iters={iterations})")
        ax.axis('off')                                      # 隱藏坐標軸

        # ── 建立 Colorbar（色階圖例） ──
        # colorbar 以 shrink=0.6 縮小至圖的 60%，並標註「K-Core Number」。
        # 注意：colorbar 會在 fig 中建立一個額外的 Axes，
        # 這就是為什麼重繪時必須使用 fig.clf() 而非 ax.clear() 的原因。
        fig.colorbar(sc, ax=ax, shrink=0.6).set_label('K-Core Number')

        self.plot_widget.refresh()                          # 觸發畫布重繪
