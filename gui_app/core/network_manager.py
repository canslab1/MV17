"""
NetworkManager -- 全域共享資料中心（新元件，舊版無對應）

本模組是整個 GUI 應用程式的「中央資料倉庫」。所有分頁（Tab）不直接互相溝通，
而是統一透過 NetworkManager 來存取與共享資料。當某個分頁修改了共享資料後，
NetworkManager 會透過 Qt Signal 發出通知，其他監聽該信號的分頁便能即時更新。

這種設計模式的好處：
  - 分頁之間完全解耦，互不依賴
  - 新增或移除分頁時，不需要修改其他分頁的程式碼
  - 資料流向清晰：寫入 -> 發出信號 -> 監聽者回應

共享資料一覽：
  G                    -- NetworkX 圖物件（網路結構本體）
  network_name         -- 網路名稱（用於顯示）
  edgelist_path        -- 邊列表檔案的路徑
  net_attr             -- 節點屬性計算結果（dict，包含各種中心性指標等）
  propagation_results  -- SIR 傳播模擬結果
  pos                  -- 節點佈局座標（dict，用於繪圖時固定節點位置）
  basic_analysis       -- 基礎統計分析結果（節點數、邊數、密度等）
  project_root         -- 專案根目錄路徑
"""
from PySide6.QtCore import QObject, Signal


class NetworkManager(QObject):
    """
    全域共享資料中心，負責儲存網路資料並協調各分頁之間的通訊。

    所有分頁在初始化時都會從 MainWindow 取得同一個 NetworkManager 實例，
    並連接（connect）自己感興趣的信號。當資料更新時，對應的信號會自動發出，
    監聽者即可做出回應（例如重新繪圖、更新表格等）。
    """

    # ── Qt 信號定義 ──────────────────────────────────────────────────
    #
    # network_loaded(name: str)
    #   觸發時機：新的網路圖被成功載入或重新生成時
    #   發出者：  tab_network_io（網路 I/O 分頁）呼叫 set_network() 時觸發
    #   監聽者：  - MainWindow（更新視窗標題列與狀態列）
    #            - tab_network_io（更新檔案樹的選取狀態與按鈕啟用狀態）
    #            - tab_node_attributes（啟用屬性計算功能，重置舊的計算結果）
    #            - tab_network_viz（啟用視覺化功能，準備繪製新網路）
    #            - tab_statistics（重置統計結果，準備重新分析）
    #            - tab_sir_experiment（啟用 SIR 實驗功能，重置舊的模擬結果）
    network_loaded = Signal(str)

    # network_cleared()
    #   觸發時機：使用者清除目前載入的網路時
    #   發出者：  clear() 方法被呼叫時觸發
    #   監聯者：  - MainWindow（更新狀態列顯示「已清除」訊息）
    #            - tab_network_io（重置檔案選擇介面）
    #            - tab_node_attributes（清空屬性表格與繪圖）
    #            - tab_network_viz（清空視覺化圖表，停用操作按鈕）
    #            - tab_statistics（清空所有統計圖表與分析表格）
    #            - tab_sir_experiment（清空模擬結果，停用實驗按鈕）
    network_cleared = Signal()

    # attributes_computed()
    #   觸發時機：節點屬性（如各種中心性指標）計算完成時
    #   發出者：  - tab_node_attributes（節點屬性分頁）呼叫 set_attributes() 時觸發
    #            - tab_network_io（從 JSON 檔匯入屬性時）呼叫 set_attributes() 時觸發
    #            - tab_node_attributes 也會直接 emit 此信號（重新計算後通知）
    #   監聽者：  - MainWindow（更新狀態列提示屬性已就緒）
    #            - tab_node_attributes（更新屬性表格與分布圖）
    #            - tab_network_viz（在下拉選單中填入可用的色彩/大小映射屬性）
    #            - tab_statistics（更新散佈圖可用的屬性選項）
    attributes_computed = Signal()

    # propagation_completed()
    #   觸發時機：SIR 傳播模擬實驗執行完畢時
    #   發出者：  tab_sir_experiment（SIR 實驗分頁）呼叫 set_propagation_results() 時觸發
    #   監聽者：  - MainWindow（更新狀態列提示模擬已完成）
    #            - tab_statistics（載入模擬曲線，繪製傳播結果圖表）
    propagation_completed = Signal()

    def __init__(self, parent=None):
        """
        初始化 NetworkManager，將所有共享資料欄位設為預設值。

        參數：
            parent: Qt 父物件，用於物件生命週期管理（可為 None）
        """
        super().__init__(parent)

        # ── 網路基礎資料 ──
        self.G = None                   # NetworkX 圖物件（Graph / DiGraph）
        self.network_name = ""          # 網路的顯示名稱（例如 "karate_club"）
        self.edgelist_path = ""         # 邊列表來源檔案的完整路徑

        # ── 分析與模擬結果 ──
        self.net_attr = None            # 節點屬性 dict（由 tab_node_attributes 計算）
        self.propagation_results = None # SIR 傳播模擬結果（由 tab_sir_experiment 產生）

        # ── 視覺化相關 ──
        self.pos = None                 # 節點佈局座標 dict {node_id: (x, y)}

        # ── 統計分析 ──
        self.basic_analysis = None      # 基礎分析結果（由 tab_statistics 計算）

        # ── 路徑 ──
        self.project_root = ""          # 專案根目錄路徑

    # ═══════════════════════════════════════════════════════════════════
    # 資料寫入方法（Setter）
    # ═══════════════════════════════════════════════════════════════════

    def set_network(self, G, name, path, pos=None):
        """
        設定新的網路圖，並通知所有監聽者。

        當 tab_network_io 成功載入或生成一個新的網路後，會呼叫此方法。
        此方法會：
          1. 儲存網路圖物件及其元資料
          2. 清除舊的分析結果（屬性、傳播、基礎分析），因為它們已不適用於新網路
          3. 發出 network_loaded 信號，通知所有分頁進行更新

        參數：
            G:    NetworkX 圖物件
            name: 網路名稱（str），用於介面顯示
            path: 邊列表檔案路徑（str）
            pos:  節點佈局座標（dict 或 None），若為 None 則由繪圖時自動計算
        """
        self.G = G
        self.network_name = name
        self.edgelist_path = path
        self.pos = pos
        # 載入新網路時，舊的分析結果不再適用，必須清除
        self.net_attr = None
        self.propagation_results = None
        self.basic_analysis = None
        self.network_loaded.emit(name)

    def set_attributes(self, net_attr):
        """
        設定節點屬性計算結果，並通知所有監聽者。

        當 tab_node_attributes 完成中心性等指標的計算，或 tab_network_io 從
        JSON 檔匯入屬性資料後，會呼叫此方法。

        參數：
            net_attr: dict，鍵為屬性名稱（如 "degree_centrality"），
                      值為 {node_id: value} 的字典
        """
        self.net_attr = net_attr
        self.attributes_computed.emit()

    def set_propagation_results(self, results):
        """
        設定 SIR 傳播模擬結果，並通知所有監聽者。

        當 tab_sir_experiment 完成 SIR 模擬實驗後，會呼叫此方法。

        參數：
            results: 模擬結果物件，包含各時間步的 S/I/R 數量等資訊
        """
        self.propagation_results = results
        self.propagation_completed.emit()

    def set_basic_analysis(self, analysis):
        """
        設定基礎統計分析結果（不發出信號）。

        由 tab_statistics 在進行基礎網路分析後呼叫。此方法不發出信號，
        因為基礎分析結果僅供 tab_statistics 自身使用，不需要通知其他分頁。

        參數：
            analysis: 分析結果物件，包含節點數、邊數、密度、
                      平均路徑長度等基礎統計量
        """
        self.basic_analysis = analysis

    # ═══════════════════════════════════════════════════════════════════
    # 清除方法
    # ═══════════════════════════════════════════════════════════════════

    def clear(self):
        """
        清除所有共享資料，將 NetworkManager 重置回初始狀態，並通知所有監聽者。

        呼叫此方法後，所有分頁都會收到 network_cleared 信號，
        各自清空介面元素並停用需要網路資料的操作按鈕。
        """
        self.G = None
        self.network_name = ""
        self.edgelist_path = ""
        self.net_attr = None
        self.propagation_results = None
        self.pos = None
        self.basic_analysis = None
        self.network_cleared.emit()

    # ═══════════════════════════════════════════════════════════════════
    # 狀態查詢方法（Getter / Predicate）
    # ═══════════════════════════════════════════════════════════════════

    def has_network(self):
        """
        查詢是否已載入網路圖。

        各分頁在執行操作前會先呼叫此方法，確認目前有可用的網路資料。

        回傳：
            bool -- 若已載入網路則為 True，否則為 False
        """
        return self.G is not None

    def has_attributes(self):
        """
        查詢是否已計算節點屬性。

        tab_network_viz 和 tab_statistics 在需要依據屬性繪圖時，
        會呼叫此方法確認屬性資料是否可用。

        回傳：
            bool -- 若已計算屬性則為 True，否則為 False
        """
        return self.net_attr is not None

    def has_propagation(self):
        """
        查詢是否已有 SIR 傳播模擬結果。

        tab_statistics 在繪製傳播曲線前，會呼叫此方法確認模擬資料是否可用。

        回傳：
            bool -- 若已有模擬結果則為 True，否則為 False
        """
        return self.propagation_results is not None
