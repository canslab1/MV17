"""
背景執行緒工作者模組 (Worker Threads Module)
============================================

本檔案提供多個 QThread 工作者類別，用於在背景執行緒中執行耗時的運算任務，
避免 GUI 主執行緒被阻塞而導致介面凍結（卡死）。

設計模式
--------
所有工作者類別皆遵循統一的信號介面：
  - progress(int, str)  : 回報進度百分比（0~100）與目前步驟的文字描述
  - finished(object)    : 運算完成時發射，攜帶運算結果（通常是 dict）
  - error(str)          : 運算過程中發生例外時發射，攜帶錯誤訊息字串

使用方式（在 GUI 端）：
  1. 建立工作者實例，傳入所需參數
  2. 連接 progress / finished / error 信號到對應的槽函式
  3. 呼叫 worker.start() 啟動背景執行緒
  4. 若需中途取消，呼叫 worker.cancel()

取消機制
--------
每個工作者都內建 _cancelled 旗標，呼叫 cancel() 會將其設為 True。
對於短時間即可完成的任務（如載入網路、提取 GCC），取消旗標僅作為標記，
任務會自然結束後由 GUI 端判斷是否採用結果。
對於長時間迭代的任務（如 SIR 傳播模擬），會透過 cancel_check 回呼函式
在每輪迭代中檢查旗標，一旦偵測到取消便提前中止運算。
"""

from PySide6.QtCore import QThread, Signal
from core import algorithm_adapter as algo


class NetworkLoadWorker(QThread):
    """
    網路載入工作者

    對應舊版手動流程：
      讀取邊列表檔案 (read_edge_list) → 建立網路圖 (create_network)
      → 計算彈簧佈局 (spring_layout)

    執行內容：
      1. 從指定的 edgelist 檔案路徑讀取邊資料並建立 NetworkX 圖物件
      2. 使用彈簧佈局演算法（spring layout）計算各節點的二維座標

    finished 信號發射內容：
      dict，包含：
        'G'   : NetworkX 圖物件
        'pos' : 節點座標字典 {node_id: (x, y)}
    """

    # ── 信號定義 ──
    progress = Signal(int, str)   # (進度百分比, 步驟描述)
    finished = Signal(object)     # 運算結果 dict
    error = Signal(str)           # 錯誤訊息

    def __init__(self, filepath, parent=None):
        """
        參數：
            filepath : str — edgelist 檔案的完整路徑
            parent   : QObject — 父物件（可選）
        """
        super().__init__(parent)
        self.filepath = filepath
        self._cancelled = False  # 取消旗標

    def cancel(self):
        """設定取消旗標，通知工作者應中止運算。"""
        self._cancelled = True

    def run(self):
        """
        執行緒主體：依序執行「載入邊列表」與「計算佈局」兩個步驟。
        任何步驟發生例外時，透過 error 信號回傳錯誤訊息。
        """
        try:
            # 步驟一：讀取 edgelist 檔案並建立網路圖
            self.progress.emit(20, "Loading edgelist...")
            G = algo.create_network_from_edgelist(self.filepath)

            # 步驟二：計算彈簧佈局座標
            self.progress.emit(60, "Computing layout...")
            pos = algo.compute_spring_layout(G)

            # 完成：發射結果
            self.progress.emit(100, "Done.")
            self.finished.emit({'G': G, 'pos': pos})
        except Exception as e:
            self.error.emit(str(e))


class GCCExtractWorker(QThread):
    """
    最大連通子圖（GCC）提取工作者

    對應舊版手動流程：
      從完整網路中提取最大連通分量 (extract_gcc)
      → 重新計算彈簧佈局 (spring_layout)

    執行內容：
      1. 從傳入的圖 G 中提取最大連通子圖（Giant Connected Component）
      2. 對提取後的子圖重新計算彈簧佈局座標

    finished 信號發射內容：
      dict，包含：
        'G'   : 最大連通子圖的 NetworkX 圖物件
        'pos' : 子圖節點座標字典 {node_id: (x, y)}
    """

    # ── 信號定義 ──
    progress = Signal(int, str)
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, G, parent=None):
        """
        參數：
            G      : NetworkX 圖物件 — 原始完整網路
            parent : QObject — 父物件（可選）
        """
        super().__init__(parent)
        self.G = G
        self._cancelled = False  # 取消旗標

    def cancel(self):
        """設定取消旗標，通知工作者應中止運算。"""
        self._cancelled = True

    def run(self):
        """
        執行緒主體：依序執行「提取 GCC」與「計算佈局」兩個步驟。
        """
        try:
            # 步驟一：提取最大連通子圖
            self.progress.emit(30, "Extracting GCC...")
            Gcc = algo.extract_gcc(self.G)

            # 步驟二：對 GCC 重新計算佈局
            self.progress.emit(70, "Computing layout...")
            pos = algo.compute_spring_layout(Gcc)

            # 完成：發射結果
            self.progress.emit(100, "Done.")
            self.finished.emit({'G': Gcc, 'pos': pos})
        except Exception as e:
            self.error.emit(str(e))


class AttributeComputeWorker(QThread):
    """
    網路屬性計算工作者

    對應舊版手動流程：
      依序計算各種節點屬性（k-core、PageRank、聚集係數、
      k-core entropy、neighbor-core、neighbor-degree、MV17 等），
      在舊版中需逐一呼叫多個函式。

    執行內容：
      呼叫 algo.compute_all_attributes() 一次性計算所有節點屬性，
      並透過 progress_callback 即時回報各步驟的進度。

    finished 信號發射內容：
      net_attr : 節點屬性字典（由 algorithm_adapter 定義，
                 結構為 {node_id: {attr_key: value}}，包含 12 種節點屬性）
    """

    # ── 信號定義 ──
    progress = Signal(int, str)
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, G, parent=None):
        """
        參數：
            G      : NetworkX 圖物件 — 要計算屬性的網路
            parent : QObject — 父物件（可選）
        """
        super().__init__(parent)
        self.G = G
        self._cancelled = False  # 取消旗標

    def cancel(self):
        """設定取消旗標，通知工作者應中止運算。"""
        self._cancelled = True

    def run(self):
        """
        執行緒主體：計算全部網路屬性。
        透過 lambda 將演算法內部的進度回報轉發為 Qt 信號。
        """
        try:
            net_attr = algo.compute_all_attributes(
                self.G,
                progress_callback=lambda p, m: self.progress.emit(p, m))
            self.finished.emit(net_attr)
        except Exception as e:
            self.error.emit(str(e))


class CentralityComputeWorker(QThread):
    """
    中心性指標計算工作者

    對應舊版手動流程：
      分別呼叫 compute_betweenness()、compute_closeness() 等函式，
      逐一計算各種中心性指標。

    執行內容：
      根據使用者勾選的選項，選擇性地計算以下中心性指標：
        - 介數中心性（betweenness centrality）：精確計算
        - 近似介數中心性（approximate betweenness）：使用 epsilon 參數控制精度
        - 接近中心性（closeness centrality）

    finished 信號發射內容：
      dict，可能包含以下鍵值（視使用者選擇而定）：
        'betweenness'        : dict {node_id: float} — 精確介數中心性
        'approx_betweenness' : dict {node_id: float} — 近似介數中心性
        'closeness'          : dict {node_id: float} — 接近中心性
    """

    # ── 信號定義 ──
    progress = Signal(int, str)
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, G, compute_betweenness=True, compute_closeness=True,
                 compute_approx_betweenness=False, approx_epsilon=0.1, parent=None):
        """
        參數：
            G                        : NetworkX 圖物件 — 要計算中心性的網路
            compute_betweenness      : bool — 是否計算精確介數中心性（預設 True）
            compute_closeness        : bool — 是否計算接近中心性（預設 True）
            compute_approx_betweenness : bool — 是否計算近似介數中心性（預設 False）
            approx_epsilon           : float — 近似演算法的 epsilon 參數（預設 0.1）
            parent                   : QObject — 父物件（可選）
        """
        super().__init__(parent)
        self.G = G
        self._compute_bet = compute_betweenness
        self._compute_clos = compute_closeness
        self._compute_approx_bet = compute_approx_betweenness
        self._approx_epsilon = approx_epsilon
        self._cancelled = False  # 取消旗標

    def cancel(self):
        """設定取消旗標，通知工作者應中止運算。"""
        self._cancelled = True

    def run(self):
        """
        執行緒主體：依據初始化時的設定，選擇性地計算各項中心性指標。
        各步驟以不同的進度百分比回報，讓 GUI 端可更新進度條。
        """
        try:
            result = {}

            # 若需要計算精確介數中心性
            if self._compute_bet:
                self.progress.emit(15, "Computing betweenness...")
                result['betweenness'] = algo.compute_betweenness(self.G)

            # 若需要計算近似介數中心性（適用於大型網路）
            if self._compute_approx_bet:
                self.progress.emit(40, "Computing approx. betweenness...")
                result['approx_betweenness'] = algo.compute_approx_betweenness(
                    self.G, epsilon=self._approx_epsilon)

            # 若需要計算接近中心性
            if self._compute_clos:
                self.progress.emit(70, "Computing closeness...")
                result['closeness'] = algo.compute_closeness(self.G)

            # 完成：發射結果
            self.progress.emit(100, "Done.")
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class SIRPropagationWorker(QThread):
    """
    SIR 傳播模擬工作者

    對應舊版手動流程：
      在舊版腳本中，使用者需手動設定 SIR 模型參數，然後呼叫
      run_sir_experiment() 執行大量模擬輪次，最後收集結果。
      這個過程非常耗時，舊版在命令列中執行時會長時間阻塞。

    執行內容：
      執行 SIR（Susceptible-Infected-Recovered）傳播實驗：
        - 根據指定的測量指標（measurement_list）選出重要節點
        - 以選出的 top-k 或 top-p 比例節點作為初始感染源
        - 執行多輪（num_round）SIR 模擬，每輪模擬若干時間步（num_time_step）
        - 統計各指標的傳播效果

    finished 信號發射內容：
      result : SIR 實驗結果資料結構（由 algorithm_adapter 定義，
               通常包含各測量指標對應的傳播曲線與統計量）

    取消機制：
      此工作者支援即時取消。cancel_check 回呼函式會在每輪模擬開始前被呼叫，
      若偵測到 _cancelled 為 True，演算法會提前中止並回傳已完成的部分結果。
    """

    # ── 信號定義 ──
    progress = Signal(int, str)
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, G, net_attr, measurement_list,
                 top_k=1, top_p=0.01, mode=1,
                 num_round=1000, num_time_step=50,
                 rate_infection=0.1, rate_recovery=1, parent=None):
        """
        參數：
            G                : NetworkX 圖物件 — 傳播模擬所用的網路
            net_attr         : 網路屬性資料 — 由 AttributeComputeWorker 計算得到
            measurement_list : list[str] — 要使用的測量指標名稱清單
                               （如 ['degree', 'betweenness', 'closeness'] 等）
            top_k            : int — 選取排名前 k 個節點作為初始感染源（mode=1 時使用）
            top_p            : float — 選取排名前 p 比例的節點（mode=2 時使用）
            mode             : int — 選取模式（1=固定數量 top_k, 2=固定比例 top_p）
            num_round        : int — SIR 模擬的總輪次（預設 1000）
            num_time_step    : int — 每輪模擬的時間步數（預設 50）
            rate_infection   : float — 感染率 beta（預設 0.1）
            rate_recovery    : float — 恢復率 gamma（預設 1）
            parent           : QObject — 父物件（可選）
        """
        super().__init__(parent)
        self.G = G
        self.net_attr = net_attr
        self.measurement_list = measurement_list
        self.top_k = top_k
        self.top_p = top_p
        self.mode = mode
        self.num_round = num_round
        self.num_time_step = num_time_step
        self.rate_infection = rate_infection
        self.rate_recovery = rate_recovery
        self._cancelled = False  # 取消旗標

    def cancel(self):
        """
        設定取消旗標。
        演算法內部會在每輪迭代透過 cancel_check 回呼檢查此旗標，
        偵測到取消後會安全地提前結束模擬。
        """
        self._cancelled = True

    def run(self):
        """
        執行緒主體：執行 SIR 傳播實驗。
        透過 progress_callback 回報進度，透過 cancel_check 支援中途取消。
        """
        try:
            result = algo.run_sir_experiment(
                self.G, self.net_attr, self.measurement_list,
                top_k=self.top_k, top_p=self.top_p, mode=self.mode,
                num_round=self.num_round, num_time_step=self.num_time_step,
                rate_infection=self.rate_infection, rate_recovery=self.rate_recovery,
                progress_callback=lambda p, m: self.progress.emit(p, m),
                cancel_check=lambda: self._cancelled)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class SIRRankingWorker(QThread):
    """
    逐節點 SIR 排名工作者

    對應舊版手動流程：
      在舊版腳本中，需要為每個節點逐一執行 SIR 模擬，以該節點為唯一
      初始感染源，統計其傳播能力，最終得到所有節點的 SIR 傳播力排名。
      這是最耗時的運算之一，節點數越多所需時間越長。

    執行內容：
      對網路中的每個節點分別執行 SIR 模擬：
        - 依序以每個節點為初始感染源
        - 在指定的多組感染率（rate_infection_list）下各執行多輪模擬
        - 統計每個節點的平均傳播規模
        - 最終輸出節點的 SIR 傳播力排名

    finished 信號發射內容：
      result : SIR 排名結果資料結構（由 algorithm_adapter 定義，
               通常包含各節點在不同感染率下的平均傳播規模與排名）

    取消機制：
      此工作者支援即時取消。由於需要遍歷所有節點，運算時間可能極長，
      cancel_check 回呼會在處理每個節點前被呼叫，允許使用者隨時中止。
    """

    # ── 信號定義 ──
    progress = Signal(int, str)
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, G, num_round=1000, num_time_step=50,
                 rate_infection_list=None, rate_recovery=1, parent=None):
        """
        參數：
            G                    : NetworkX 圖物件 — 要進行排名的網路
            num_round            : int — 每個節點的 SIR 模擬輪次（預設 1000）
            num_time_step        : int — 每輪模擬的時間步數（預設 50）
            rate_infection_list  : list[float] — 要測試的感染率清單
                                   （預設 [0.1]，可指定多組如 [0.05, 0.1, 0.2]）
            rate_recovery        : float — 恢復率 gamma（預設 1）
            parent               : QObject — 父物件（可選）
        """
        super().__init__(parent)
        self.G = G
        self.num_round = num_round
        self.num_time_step = num_time_step
        self.rate_infection_list = rate_infection_list or [0.1]  # 若未指定則預設 [0.1]
        self.rate_recovery = rate_recovery
        self._cancelled = False  # 取消旗標

    def cancel(self):
        """
        設定取消旗標。
        由於逐節點 SIR 排名極為耗時，此取消機制尤為重要，
        允許使用者在不需等待全部節點完成的情況下中止運算。
        """
        self._cancelled = True

    def run(self):
        """
        執行緒主體：對每個節點執行 SIR 模擬以計算傳播力排名。
        透過 progress_callback 回報進度，透過 cancel_check 支援中途取消。
        """
        try:
            result = algo.compute_sir_ranking(
                self.G, self.num_round, self.num_time_step,
                self.rate_infection_list, self.rate_recovery,
                progress_callback=lambda p, m: self.progress.emit(p, m),
                cancel_check=lambda: self._cancelled)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class BatchAnalysisWorker(QThread):
    """
    批次網路分析工作者

    對應舊版手動流程：
      在舊版中，若要分析多個網路，使用者需逐一載入每個 edgelist 檔案，
      手動執行相同的分析流程（建立網路 → 計算屬性 → 輸出結果），
      然後彙整所有結果。此工作者將整個批次流程自動化。

    執行內容：
      對指定資料夾中的多個 edgelist 檔案批次執行網路分析：
        - 依序讀取每個 edgelist 檔案
        - 對每個網路執行標準的分析流程
        - 彙整所有網路的分析結果

    finished 信號發射內容：
      result : 批次分析結果資料結構（由 algorithm_adapter 定義，
               通常包含每個網路的屬性摘要與比較資料）

    取消機制：
      此工作者支援即時取消。cancel_check 回呼會在處理每個檔案前被呼叫，
      允許使用者在批次處理的中途中止運算。
    """

    # ── 信號定義 ──
    progress = Signal(int, str)
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, folder_edgelist, file_name_list, parent=None):
        """
        參數：
            folder_edgelist : str — edgelist 檔案所在的資料夾路徑
            file_name_list  : list[str] — 要分析的 edgelist 檔案名稱清單
            parent          : QObject — 父物件（可選）
        """
        super().__init__(parent)
        self.folder_edgelist = folder_edgelist
        self.file_name_list = file_name_list
        self._cancelled = False  # 取消旗標

    def cancel(self):
        """
        設定取消旗標。
        批次分析可能涉及大量檔案，此取消機制讓使用者可隨時中止。
        """
        self._cancelled = True

    def run(self):
        """
        執行緒主體：對檔案清單中的每個 edgelist 檔案執行網路分析。
        透過 progress_callback 回報整體進度，透過 cancel_check 支援中途取消。
        """
        try:
            result = algo.batch_network_analysis(
                self.folder_edgelist, self.file_name_list,
                progress_callback=lambda p, m: self.progress.emit(p, m),
                cancel_check=lambda: self._cancelled)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))
