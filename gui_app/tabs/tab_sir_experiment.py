"""
Tab 4：SIR 傳播實驗分頁

本分頁對應以下三個原始腳本的功能：

1. experiment1.py — SIR 傳播實驗（主流程）
   - 以各中心性指標排序，取 Top-K 或 Top-P 節點作為初始感染源
   - 對每個指標執行多輪 SIR 模擬，計算平均恢復密度隨時間的變化曲線
   - 相關常數：NUM_ROUND, NUM_TIME_STEP, RATE_INFECTION, RATE_RECOVERY,
     NUM_TOPK, NUM_TOPP

2. experiment1_draw plot.py — 繪製傳播曲線
   - 使用 COLOR_LIST 為各指標分配線條顏色
   - 圖例邏輯：若指標為 node_mv17 則顯示 'proposed'，
     否則取 split('_')[-1]（例如 'node_degree' → 'degree'）

3. code/sir_ranking_file_writer.py — 逐節點 SIR 排名
   - 以每個節點為唯一初始感染源，獨立執行多輪 SIR
   - 支援多組感染率（以逗號分隔輸入）
   - 當 N > 500 時會警告，因為時間複雜度為 O(N * rounds * time_steps)
"""
import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QSpinBox, QDoubleSpinBox, QComboBox,
    QCheckBox, QTextEdit, QSplitter, QMessageBox, QFileDialog
)
from PySide6.QtCore import Qt
from core.worker_threads import SIRPropagationWorker, SIRRankingWorker
from core import algorithm_adapter as algo
from widgets.matplotlib_canvas import PlotWidget
from widgets.progress_dialog import ProgressDialog


# ──────────────────────────────────────────────────────────
# 可選的中心性指標列表
# ──────────────────────────────────────────────────────────
# 每個元組 (內部鍵名, 顯示在 UI 上的標籤)
# 此列表對應 experiment1.py 中的 measurement_list：
#   measurement_list = ['node_betweenness', 'node_closeness', 'node_degree',
#                       'node_k-core', 'node_mv17', 'node_neighbor-core',
#                       'node_pagerank']
# 使用者可在 UI 上透過勾選框自由選擇要比較的指標。
MEASURE_OPTIONS = [
    ('node_degree', 'Degree'),               # 度中心性
    ('node_betweenness', 'Betweenness'),     # 介數中心性
    ('node_closeness', 'Closeness'),         # 接近中心性
    ('node_k-core', 'K-Core'),               # K-核心分解值
    ('node_neighbor-core', 'Neighbor Core'), # 鄰居核心值
    ('node_pagerank', 'PageRank'),           # PageRank 值
    ('node_mv17', 'MV17 (Proposed)'),        # 本研究提出的 MV17 方法
]


class TabSIRExperiment(QWidget):
    """
    SIR 傳播實驗分頁。

    提供兩大功能區塊：
    (A) SIR 傳播模擬 — 比較各中心性指標在 SIR 模型中的傳播能力
    (B) 逐節點 SIR 排名 — 對每個節點獨立跑 SIR，產出 ground-truth 排名
    """

    def __init__(self, manager, parent=None):
        """
        初始化分頁。

        參數:
            manager: 全域的 NetworkManager，負責管理圖 (G)、節點屬性
                     (net_attr)、傳播結果等共用狀態
            parent:  父 widget（可為 None）
        """
        super().__init__(parent)
        self.manager = manager

        # 持有背景執行緒的參考，以便可在執行中途取消
        self._worker = None           # SIR 傳播模擬的工作執行緒
        self._rank_worker = None      # 逐節點 SIR 排名的工作執行緒
        self._sir_ranking_result = None  # 儲存排名結果 dict[node_id -> dict[rate -> score]]

        # 建立 UI 元件
        self._init_ui()

        # 當網路載入或清除時，啟用 / 禁用按鈕
        self.manager.network_loaded.connect(self._on_network_loaded)
        self.manager.network_cleared.connect(self._on_cleared)

    # ================================================================
    #  UI 初始化
    # ================================================================
    def _init_ui(self):
        """建立整個分頁的 UI 佈局。"""
        layout = QVBoxLayout(self)

        # ────────────────────────────────────────
        # 第一區：SIR 參數設定 (QGroupBox)
        # ────────────────────────────────────────
        # 此區段的欄位對應 experiment1.py 裡的常數：
        #   NUM_ROUND       → spin_rounds
        #   NUM_TIME_STEP   → spin_timesteps
        #   RATE_INFECTION  → spin_beta
        #   RATE_RECOVERY   → spin_gamma
        #   NUM_TOPK        → spin_topk
        #   NUM_TOPP        → spin_topp
        param_group = QGroupBox("SIR Parameters")
        pg = QHBoxLayout(param_group)

        # ---------- 第 1 欄：模擬次數與時間步數 ----------
        col1 = QVBoxLayout()

        # 「Rounds（模擬輪數）」：每輪獨立跑一次 SIR，最後取平均
        # 對應論文中的 "5000 simulations for each network dataset"
        col1.addWidget(QLabel("Rounds:"))
        self.spin_rounds = QSpinBox()
        self.spin_rounds.setRange(10, 50000)
        self.spin_rounds.setValue(5000)
        col1.addWidget(self.spin_rounds)

        # 「Time steps（時間步數）」：每輪 SIR 模擬的最大時間步
        # 對應 experiment1.py 的 NUM_TIME_STEP（預設 50）
        col1.addWidget(QLabel("Time steps:"))
        self.spin_timesteps = QSpinBox()
        self.spin_timesteps.setRange(10, 200)
        self.spin_timesteps.setValue(50)
        col1.addWidget(self.spin_timesteps)
        pg.addLayout(col1)

        # ---------- 第 2 欄：感染率與恢復率 ----------
        col2 = QVBoxLayout()

        # 「Infection rate (beta)」：SIR 模型的感染機率
        # 對應 experiment1.py 的 RATE_INFECTION
        col2.addWidget(QLabel("Infection rate (beta):"))
        self.spin_beta = QDoubleSpinBox()
        self.spin_beta.setRange(0.01, 1.0)
        self.spin_beta.setSingleStep(0.01)
        self.spin_beta.setValue(0.2)
        self.spin_beta.setDecimals(3)
        col2.addWidget(self.spin_beta)

        # 「Recovery rate (gamma)」：SIR 模型中每個時間步感染者恢復的機率
        # 對應 experiment1.py 的 RATE_RECOVERY（預設 1.0 表示感染後下一步必定恢復）
        col2.addWidget(QLabel("Recovery rate (gamma):"))
        self.spin_gamma = QDoubleSpinBox()
        self.spin_gamma.setRange(0.01, 1.0)
        self.spin_gamma.setSingleStep(0.1)
        self.spin_gamma.setValue(1.0)
        self.spin_gamma.setDecimals(2)
        col2.addWidget(self.spin_gamma)
        pg.addLayout(col2)

        # ---------- 第 3 欄：Top-K / Top-P 模式 ----------
        # 選擇初始感染源的方式：
        #   Top-K：取排名前 K 個節點作為感染源（對應 NUM_TOPK）
        #   Top-P：取排名前 P 比例的節點（對應 NUM_TOPP）
        col3 = QVBoxLayout()

        col3.addWidget(QLabel("Mode:"))
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["Top-K", "Top-P"])
        # 切換模式時，啟用對應的 spinbox、禁用另一個
        self.combo_mode.currentIndexChanged.connect(self._on_mode_changed)
        col3.addWidget(self.combo_mode)

        # Top-K 值（預設 1：只取排名最高的 1 個節點）
        col3.addWidget(QLabel("Top-K:"))
        self.spin_topk = QSpinBox()
        self.spin_topk.setRange(1, 9999)
        self.spin_topk.setValue(1)
        col3.addWidget(self.spin_topk)

        # Top-P 值（預設 0.01 即前 1% 的節點）
        col3.addWidget(QLabel("Top-P:"))
        self.spin_topp = QDoubleSpinBox()
        self.spin_topp.setRange(0.001, 1.0)
        self.spin_topp.setSingleStep(0.01)
        self.spin_topp.setValue(0.01)
        self.spin_topp.setDecimals(3)
        self.spin_topp.setEnabled(False)  # 預設為 Top-K 模式，故 Top-P 禁用
        col3.addWidget(self.spin_topp)
        pg.addLayout(col3)

        # ---------- 第 4 欄：中心性指標勾選框 ----------
        # 對應 experiment1.py 中 measurement_list 裡的 7 個指標。
        # 使用者勾選哪些指標，模擬時就會比較哪些指標的傳播能力。
        col4 = QVBoxLayout()
        col4.addWidget(QLabel("Measures:"))
        self.measure_checks = {}  # dict[str, QCheckBox]：鍵 = 指標內部名稱
        for key, label in MEASURE_OPTIONS:
            cb = QCheckBox(label)
            cb.setChecked(True)  # 預設全部勾選
            self.measure_checks[key] = cb
            col4.addWidget(cb)
        pg.addLayout(col4)

        layout.addWidget(param_group)

        # ────────────────────────────────────────
        # 第二區：動作按鈕列
        # ────────────────────────────────────────
        btn_layout = QHBoxLayout()

        # 「Start Simulation」按鈕 — 開始 SIR 傳播模擬
        self.btn_start = QPushButton("Start Simulation")
        self.btn_start.clicked.connect(self._start_simulation)
        self.btn_start.setEnabled(False)  # 載入網路後才啟用
        btn_layout.addWidget(self.btn_start)

        # 「Stop」按鈕 — 取消正在執行的模擬
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self._stop_simulation)
        self.btn_stop.setEnabled(False)
        btn_layout.addWidget(self.btn_stop)

        # 「Save Results...」按鈕 — 將模擬結果存為文字檔
        self.btn_save_result = QPushButton("Save Results...")
        self.btn_save_result.clicked.connect(self._save_results)
        self.btn_save_result.setEnabled(False)
        btn_layout.addWidget(self.btn_save_result)

        # 「Save Plot...」按鈕 — 將傳播曲線圖匯出為 PNG 或 PDF
        self.btn_save_plot = QPushButton("Save Plot...")
        self.btn_save_plot.clicked.connect(self._save_plot)
        self.btn_save_plot.setEnabled(False)
        btn_layout.addWidget(self.btn_save_plot)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # ────────────────────────────────────────
        # 第三區：逐節點 SIR 排名 (Per-Node SIR Ranking)
        # ────────────────────────────────────────
        # 對應 code/sir_ranking_file_writer.py 的功能：
        #   - 以每個節點為唯一初始感染源，獨立跑多輪 SIR
        #   - 對每個節點計算其平均傳播影響力（最終恢復密度）
        #   - 支援同時計算多組感染率（逗號分隔）
        #   - 當 N > 500 時會彈出警告，因為每個節點都要跑一次完整 SIR 模擬，
        #     總計算量為 O(N * rounds * time_steps)，非常耗時
        rank_group = QGroupBox("Per-Node SIR Ranking")
        rg = QHBoxLayout(rank_group)

        # 感染率輸入框：可輸入多個感染率，以逗號分隔
        # 例如 "0.05, 0.1, 0.2" 會分別為三種感染率計算排名
        rg.addWidget(QLabel("Infection rates (comma-sep):"))
        from PySide6.QtWidgets import QLineEdit
        self.edit_rank_rates = QLineEdit("0.05")
        self.edit_rank_rates.setMaximumWidth(200)
        rg.addWidget(self.edit_rank_rates)

        # 排名用的模擬輪數（與上方傳播實驗的 rounds 獨立設定）
        rg.addWidget(QLabel("Rounds:"))
        self.spin_rank_rounds = QSpinBox()
        self.spin_rank_rounds.setRange(10, 50000)
        self.spin_rank_rounds.setValue(5000)
        rg.addWidget(self.spin_rank_rounds)

        # 「Compute SIR Ranking」按鈕 — 開始逐節點排名計算
        self.btn_rank_start = QPushButton("Compute SIR Ranking")
        self.btn_rank_start.clicked.connect(self._start_sir_ranking)
        self.btn_rank_start.setEnabled(False)
        rg.addWidget(self.btn_rank_start)

        # 「Stop」按鈕 — 取消正在執行的排名計算
        self.btn_rank_stop = QPushButton("Stop")
        self.btn_rank_stop.clicked.connect(self._stop_sir_ranking)
        self.btn_rank_stop.setEnabled(False)
        rg.addWidget(self.btn_rank_stop)

        # 「Save Ranking...」按鈕 — 將排名結果存為文字檔
        self.btn_rank_save = QPushButton("Save Ranking...")
        self.btn_rank_save.clicked.connect(self._save_sir_ranking)
        self.btn_rank_save.setEnabled(False)
        rg.addWidget(self.btn_rank_save)

        rg.addStretch()
        layout.addWidget(rank_group)

        # ────────────────────────────────────────
        # 第四區：結果顯示區（左圖 + 右文字）
        # ────────────────────────────────────────
        # 使用 QSplitter 讓使用者可以自由拖動分割比例
        splitter = QSplitter(Qt.Horizontal)

        # 左側：matplotlib 繪圖區域，用來顯示 SIR 傳播曲線
        self.plot_widget = PlotWidget(figsize=(7, 5))
        splitter.addWidget(self.plot_widget)

        # 右側：純文字結果摘要
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        splitter.addWidget(self.result_text)

        # 設定左右拉伸比例為 3:1（圖佔較大空間）
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)

    # ================================================================
    #  UI 狀態切換
    # ================================================================
    def _on_mode_changed(self, index):
        """
        切換 Top-K / Top-P 模式時的回調。

        當選擇 Top-K (index=0) 時，啟用 spin_topk 並禁用 spin_topp。
        當選擇 Top-P (index=1) 時，反之。
        """
        self.spin_topk.setEnabled(index == 0)
        self.spin_topp.setEnabled(index == 1)

    def _on_network_loaded(self, name):
        """
        網路成功載入後的回調：啟用「Start Simulation」及「Compute SIR Ranking」按鈕。
        """
        self.btn_start.setEnabled(True)
        self.btn_rank_start.setEnabled(True)

    def _on_cleared(self):
        """
        網路被清除（用戶重置或關閉）時的回調：
        禁用所有按鈕、清空結果、清空圖表。
        """
        self.btn_start.setEnabled(False)
        self.btn_save_result.setEnabled(False)
        self.btn_save_plot.setEnabled(False)
        self.btn_rank_start.setEnabled(False)
        self.btn_rank_save.setEnabled(False)
        self._sir_ranking_result = None
        self.plot_widget.clear()
        self.result_text.clear()

    # ================================================================
    #  SIR 傳播模擬（對應 experiment1.py）
    # ================================================================
    def _start_simulation(self):
        """
        按下「Start Simulation」時的處理邏輯。

        流程：
        1. 檢查是否已載入網路、是否已計算節點屬性
        2. 收集使用者勾選的中心性指標
        3. 確認所選指標的屬性已在 net_attr 中存在
        4. 建立 SIRPropagationWorker（背景執行緒），傳入所有參數
        5. 顯示進度對話框，開始模擬
        """
        # 前置檢查：必須先載入網路
        if not self.manager.has_network():
            QMessageBox.warning(self, "Warning", "Please load a network first.")
            return
        # 前置檢查：必須先計算節點屬性（degree、betweenness 等）
        if not self.manager.has_attributes():
            QMessageBox.warning(self, "Warning", "Please compute node attributes first.")
            return

        # 收集使用者勾選的指標鍵名（例如 ['node_degree', 'node_mv17', ...]）
        measures = [k for k, cb in self.measure_checks.items() if cb.isChecked()]
        if not measures:
            QMessageBox.warning(self, "Warning", "Please select at least one measure.")
            return

        # 檢查所選指標是否已經計算過（存在於 net_attr 中）
        # 取任意一個節點的屬性字典作為樣本
        sample_node = next(iter(self.manager.net_attr.values()))
        missing = [m for m in measures if m not in sample_node]
        if missing:
            QMessageBox.warning(self, "Warning",
                                f"Missing attributes: {', '.join(missing)}.\n"
                                "Please compute them first.")
            return

        # mode=1 表示 Top-K 模式，mode=2 表示 Top-P 模式
        mode = 1 if self.combo_mode.currentIndex() == 0 else 2

        # 建立進度對話框並啟動背景執行緒
        self._progress = ProgressDialog("Running SIR Simulation...", self)
        self._worker = SIRPropagationWorker(
            self.manager.G,           # NetworkX 圖物件
            self.manager.net_attr,    # 節點屬性字典 {node_id: {attr_key: value}}
            sorted(measures),         # 排序後的指標列表（確保結果一致）
            top_k=self.spin_topk.value(),
            top_p=self.spin_topp.value(),
            mode=mode,
            num_round=self.spin_rounds.value(),
            num_time_step=self.spin_timesteps.value(),
            rate_infection=self.spin_beta.value(),
            rate_recovery=self.spin_gamma.value()
        )
        self._progress.set_worker(self._worker)
        self._worker.progress.connect(self._progress.update_progress)
        self._worker.finished.connect(self._on_simulation_done)
        self._worker.error.connect(self._on_error)
        self._progress.show()

        # 禁用 Start、啟用 Stop，防止重複點擊
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._worker.start()

    def _stop_simulation(self):
        """取消正在執行的 SIR 傳播模擬。"""
        if self._worker:
            self._worker.cancel()
        self.btn_stop.setEnabled(False)

    def _on_simulation_done(self, result):
        """
        SIR 傳播模擬完成後的回調。

        參數:
            result: dict[str, list[float]]
                鍵為指標名稱（如 'node_degree'），
                值為長度等於 num_time_step 的列表，
                每個元素是該時間步的平均恢復密度 (R/N)
        """
        self._progress.close()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

        # 若模擬被取消或無結果，僅顯示提示訊息
        if not result:
            self.result_text.setText("Simulation cancelled or no results.")
            return

        # 將結果存入 manager 以供其他分頁使用
        self.manager.set_propagation_results(result)

        # 繪製傳播曲線圖並顯示文字摘要
        self._draw_propagation(result)
        self._show_summary(result)

        # 啟用「儲存結果」和「儲存圖表」按鈕
        self.btn_save_result.setEnabled(True)
        self.btn_save_plot.setEnabled(True)

    # ================================================================
    #  繪製傳播曲線（對應 experiment1_draw plot.py → draw_propagation_result()）
    # ================================================================
    def _draw_propagation(self, result):
        """
        繪製 SIR 傳播曲線圖。

        對應舊版 experiment1_draw plot.py 中的 draw_propagation_result() 函式。

        繪圖邏輯：
        - 按指標名稱排序後，依序繪製各指標的時間序列曲線
        - 線條顏色使用 algo.COLOR_LIST（與舊版一致），以索引取餘來循環使用
        - 圖例標籤邏輯：
            * 若指標名為 'node_mv17'，則圖例顯示 'proposed'（代表本研究方法）
            * 否則，取指標名 split('_') 後的最後一部分
              例如 'node_degree' → 'degree'、'node_k-core' → 'k-core'

        參數:
            result: dict[str, list[float]]
                各指標的傳播曲線資料
        """
        ax = self.plot_widget.get_axes()
        ax.clear()
        ax.grid(True, alpha=0.3)

        for idx, key in enumerate(sorted(result.keys())):
            # 從 COLOR_LIST 取顏色，取餘以處理指標數超過顏色數的情況
            color = algo.COLOR_LIST[idx % len(algo.COLOR_LIST)]

            # 圖例標籤：node_mv17 → 'proposed'，其餘取最後一段
            label = 'proposed' if key == 'node_mv17' else key.split('_')[-1]

            ax.plot(result[key], color=color, linewidth=1.5, label=label)

        ax.set_xlabel('Time step')
        ax.set_ylabel('Recovered density (R/N)')
        ax.set_title(f"SIR Propagation: {self.manager.network_name}")
        ax.legend(loc='lower right', fontsize='small')
        self.plot_widget.refresh()

    def _show_summary(self, result):
        """
        在右側文字區域顯示模擬結果的摘要。

        包含：
        - 當前使用的參數設定（網路名、輪數、時間步、beta、gamma）
        - 依最終恢復密度 (R/N) 排名的各指標成績
        """
        lines = ["=== SIR Simulation Summary ===\n"]
        lines.append(f"Network: {self.manager.network_name}")
        lines.append(f"Rounds: {self.spin_rounds.value()}")
        lines.append(f"Time steps: {self.spin_timesteps.value()}")
        lines.append(f"Beta: {self.spin_beta.value()}")
        lines.append(f"Gamma: {self.spin_gamma.value()}\n")

        # 依最後一個時間步的恢復密度由高到低排序
        ranked = sorted(result.items(), key=lambda x: x[1][-1] if x[1] else 0, reverse=True)
        lines.append("Ranking by final density (R/N):")
        for rank, (key, vals) in enumerate(ranked, 1):
            final = vals[-1] if vals else 0
            # 標籤邏輯與繪圖一致
            label = 'proposed' if key == 'node_mv17' else key.split('_')[-1]
            lines.append(f"  {rank}. {label}: {final:.6f}")

        self.result_text.setText('\n'.join(lines))

    # ================================================================
    #  儲存模擬結果與圖表
    # ================================================================
    def _save_results(self):
        """將傳播模擬結果以文字格式儲存到檔案。"""
        if not self.manager.has_propagation():
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Results", "", "Text Files (*.txt)")
        if path:
            algo.write_propagation_result(path, self.manager.propagation_results)

    def _save_plot(self):
        """將傳播曲線圖匯出為 PNG 或 PDF 影像檔。"""
        path, _ = QFileDialog.getSaveFileName(self, "Save Plot", "", "PNG (*.png);;PDF (*.pdf)")
        if path:
            self.plot_widget.get_figure().savefig(path, dpi=300, bbox_inches='tight')

    # ================================================================
    #  逐節點 SIR 排名（對應 code/sir_ranking_file_writer.py）
    # ================================================================
    def _start_sir_ranking(self):
        """
        按下「Compute SIR Ranking」時的處理邏輯。

        對應 code/sir_ranking_file_writer.py 的功能：
        - 對網路中的每一個節點，分別以該節點為唯一初始感染源
        - 執行多輪 SIR 模擬，計算其平均傳播影響力
        - 支援多組感染率，以逗號分隔輸入（例如 "0.05, 0.1, 0.2"）

        當節點數 N > 500 時會彈出確認對話框：
        因為需要執行 N 次完整的 SIR 模擬（每次 rounds * time_steps），
        時間複雜度為 O(N * rounds * time_steps)，在大型網路上非常耗時。
        """
        # 前置檢查：必須先載入網路
        if not self.manager.has_network():
            QMessageBox.warning(self, "Warning", "Please load a network first.")
            return

        # 解析使用者輸入的感染率列表
        # 例如 "0.05, 0.1" → [0.05, 0.1]
        try:
            rates_text = self.edit_rank_rates.text().strip()
            rate_list = [float(r.strip()) for r in rates_text.split(',') if r.strip()]
            if not rate_list:
                raise ValueError("empty")
        except ValueError:
            QMessageBox.warning(self, "Warning", "Invalid infection rates. Use comma-separated numbers.")
            return

        # N > 500 時的警告：因為逐節點 SIR 的計算量 = N * rounds * time_steps
        n = len(self.manager.G.nodes())
        if n > 500:
            reply = QMessageBox.question(
                self, "Confirm",
                f"Per-node SIR ranking will run SIR {n} times (one per node). "
                f"This may take a very long time for N={n}. Continue?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply != QMessageBox.Yes:
                return

        # 建立進度對話框並啟動背景執行緒
        self._progress = ProgressDialog("Computing SIR Ranking...", self)
        self._rank_worker = SIRRankingWorker(
            self.manager.G,
            num_round=self.spin_rank_rounds.value(),
            num_time_step=self.spin_timesteps.value(),  # 共用上方的時間步設定
            rate_infection_list=rate_list,               # 多組感染率
            rate_recovery=self.spin_gamma.value())       # 共用上方的恢復率設定
        self._progress.set_worker(self._rank_worker)
        self._rank_worker.progress.connect(self._progress.update_progress)
        self._rank_worker.finished.connect(self._on_ranking_done)
        self._rank_worker.error.connect(self._on_error)
        self._progress.show()

        # 禁用 Start、啟用 Stop
        self.btn_rank_start.setEnabled(False)
        self.btn_rank_stop.setEnabled(True)
        self._rank_worker.start()

    def _stop_sir_ranking(self):
        """取消正在執行的逐節點 SIR 排名計算。"""
        if self._rank_worker:
            self._rank_worker.cancel()
        self.btn_rank_stop.setEnabled(False)

    def _on_ranking_done(self, result):
        """
        逐節點 SIR 排名計算完成後的回調。

        參數:
            result: dict[node_id, dict[rate_str, float]]
                外層鍵為節點 ID，內層鍵為感染率字串（如 "0.05"），
                值為該節點在該感染率下的平均傳播影響力
        """
        self._progress.close()
        self.btn_rank_start.setEnabled(True)
        self.btn_rank_stop.setEnabled(False)

        # 若被取消或無結果
        if not result:
            self.result_text.setText("SIR ranking cancelled or no results.")
            return

        # 儲存結果，並啟用「Save Ranking...」按鈕
        self._sir_ranking_result = result
        self.btn_rank_save.setEnabled(True)

        # 在文字區域顯示摘要：每個感染率下排名前 10 的節點
        lines = ["=== Per-Node SIR Ranking ===\n"]
        lines.append(f"Network: {self.manager.network_name}")
        lines.append(f"Nodes: {len(result)}\n")

        if result:
            # 取任一節點的屬性來獲得所有感染率的鍵名
            sample_node = next(iter(result.values()))
            rates = sorted(sample_node.keys())
            for rate_key in rates:
                lines.append(f"--- Infection rate = {rate_key} ---")
                # 依該感染率下的分數由高到低排序，顯示前 10 名
                ranked = sorted(result.items(),
                                key=lambda x: x[1].get(rate_key, 0), reverse=True)
                for rank, (node_id, vals) in enumerate(ranked[:10], 1):
                    lines.append(f"  {rank}. Node {node_id}: {vals[rate_key]:.6f}")
                lines.append("")

        self.result_text.setText('\n'.join(lines))

    def _save_sir_ranking(self):
        """
        將逐節點 SIR 排名結果儲存為空格分隔的文字檔。

        輸出格式：
            第一行：node_id rate1 rate2 ...（表頭）
            後續每行：node_id val1 val2 ...

        例如：
            node_id 0.05 0.1
            0 0.123456 0.234567
            1 0.111111 0.222222
            ...
        """
        if not self._sir_ranking_result:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save SIR Ranking", "", "Text Files (*.txt)")
        if not path:
            return
        with open(path, mode="w") as f:
            # 取任一節點來取得所有感染率鍵名，排序確保欄位順序一致
            sample_node = next(iter(self._sir_ranking_result.values()))
            rates = sorted(sample_node.keys())
            header = "node_id " + " ".join(rates)
            f.write(header + "\n")
            # 按節點 ID 排序輸出
            for ni in sorted(self._sir_ranking_result.keys()):
                vals = [str(self._sir_ranking_result[ni].get(r, 0)) for r in rates]
                f.write(f"{ni} " + " ".join(vals) + "\n")

    # ================================================================
    #  共用錯誤處理
    # ================================================================
    def _on_error(self, msg):
        """
        SIR 傳播模擬或逐節點排名發生錯誤時的回調。

        關閉進度對話框，重新啟用按鈕，並以對話框顯示錯誤訊息。
        """
        self._progress.close()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_rank_start.setEnabled(True)
        self.btn_rank_stop.setEnabled(False)
        QMessageBox.critical(self, "Error", msg)
