"""
Matplotlib 畫布元件 — 將 matplotlib 圖表嵌入 PySide6 視窗中。

本模組提供兩個類別，用於在 Qt 介面中嵌入 matplotlib 圖表：
    - MatplotlibCanvas：底層畫布，繼承自 FigureCanvasQTAgg，
      是 matplotlib 與 Qt 之間的橋接層
    - PlotWidget：高層封裝元件，將畫布 (Canvas) 與工具列 (Toolbar)
      組合成一個可直接使用的 QWidget

舊版程式沒有 GUI，圖表都是用 plt.show() 在獨立視窗中顯示，
或用 plt.savefig() 儲存為檔案。新版 GUI 需要將圖表嵌入分頁中，
因此使用 matplotlib 的 Qt 後端 (QtAgg) 來實現。

使用模式：
    各分頁透過 PlotWidget 的 get_axes() 和 get_figure() 方法
    取得 matplotlib 的 Axes 和 Figure 物件，然後用標準的 matplotlib API
    進行繪圖。繪圖完成後呼叫 refresh() 來更新畫面。

    典型用法：
        plot_widget = PlotWidget(figsize=(8, 6))
        ax = plot_widget.get_axes()
        ax.plot(x, y)
        plot_widget.refresh()

colorbar 清理注意事項：
    如果繪圖中使用了 colorbar（例如網路節點著色），在重新繪製時
    不能只呼叫 ax.clear()，因為 colorbar 是附加在 Figure 上的獨立 Axes。
    正確的做法是呼叫 fig.clf()（清除整個 Figure），然後重新建立子圖：
        fig = plot_widget.get_figure()
        fig.clf()                        # 清除整個 Figure（包含 colorbar）
        ax = fig.add_subplot(111)        # 重新建立子圖
        # ... 繪製新圖 ...
        plot_widget.refresh()
    如果只用 ax.clear()，舊的 colorbar 會殘留在畫面上，
    每次重繪都會多出一個 colorbar，最終佔滿整個圖表空間。
"""
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtWidgets import QWidget, QVBoxLayout


class MatplotlibCanvas(FigureCanvasQTAgg):
    """
    Matplotlib 畫布元件，繼承自 FigureCanvasQTAgg。

    FigureCanvasQTAgg 是 matplotlib 提供的 Qt 後端畫布，
    它同時是一個 QWidget 和一個 matplotlib FigureCanvas，
    負責將 matplotlib 的繪圖結果渲染到 Qt 視窗中。

    本類別在 FigureCanvasQTAgg 之上提供了簡便的存取方法：
    - get_figure()：取得 Figure 物件，用於需要操作整個圖表的場景
    - get_axes()：取得預設的 Axes 物件，用於一般繪圖
    - clear()：清除 Axes 並重繪畫面
    - refresh()：套用 tight_layout 並重繪畫面

    參數：
        figsize (tuple): 圖表尺寸，格式為 (寬, 高)，單位為英吋。預設 (8, 6)
        dpi (int): 解析度（每英吋點數）。預設 100
    """

    def __init__(self, figsize=(8, 6), dpi=100):
        """
        初始化畫布。

        建立一個 matplotlib Figure 物件和一個預設的子圖 (subplot)。
        Figure 的背景色設為白色 ('w')，與一般論文圖表的風格一致。
        """
        # 建立 Figure 物件，設定尺寸、解析度和背景色
        self.fig = Figure(figsize=figsize, dpi=dpi, facecolor='w')
        # 建立預設的子圖（1x1 網格中的第 1 個）
        self.axes = self.fig.add_subplot(111)
        # 呼叫父類別建構子，將 Figure 傳給 FigureCanvasQTAgg
        super().__init__(self.fig)

    def get_figure(self):
        """
        取得 Figure 物件。

        當需要操作整個圖表時使用，例如：
        - 呼叫 fig.clf() 清除整個圖表（包含 colorbar）
        - 呼叫 fig.add_subplot() 建立多個子圖
        - 呼叫 fig.colorbar() 加入色彩條

        返回：
            matplotlib.figure.Figure: 此畫布的 Figure 物件
        """
        return self.fig

    def get_axes(self):
        """
        取得預設的 Axes 物件。

        Axes 是 matplotlib 中實際進行繪圖的區域。
        大多數繪圖操作（plot、scatter、bar 等）都是在 Axes 上呼叫的。

        注意：如果之前呼叫了 fig.clf()，這個 Axes 物件會失效，
        需要重新透過 fig.add_subplot(111) 建立新的 Axes。

        返回：
            matplotlib.axes.Axes: 預設的子圖 Axes 物件
        """
        return self.axes

    def clear(self):
        """
        清除 Axes 上的所有繪圖內容並重繪畫面。

        只清除 Axes 的內容，不影響 Figure 上的其他元素。
        如果有 colorbar 需要清除，請改用 fig.clf()。
        """
        self.axes.clear()
        self.draw()

    def refresh(self):
        """
        套用 tight_layout 並重繪畫面。

        tight_layout() 會自動調整子圖的邊距，
        避免標題、軸標籤被裁切。
        繪圖完成後應呼叫此方法來更新畫面。
        """
        self.fig.tight_layout()
        self.draw()


class PlotWidget(QWidget):
    """
    繪圖元件 — 將 MatplotlibCanvas 和導航工具列組合成一個完整的元件。

    這是各分頁實際使用的繪圖元件。它包含：
    - 上方：NavigationToolbar2QT（matplotlib 的導航工具列），
      提供平移、縮放、儲存圖片等功能
    - 下方：MatplotlibCanvas（matplotlib 畫布），
      顯示實際的圖表

    使用方式：
        # 在分頁中建立繪圖元件
        self.plot = PlotWidget(figsize=(10, 6))
        layout.addWidget(self.plot)

        # 取得 Axes 進行繪圖
        ax = self.plot.get_axes()
        ax.plot([1, 2, 3], [4, 5, 6])
        self.plot.refresh()

    本類別提供與 MatplotlibCanvas 相同的 get_figure()、get_axes()、
    clear()、refresh() 方法，作為便捷的委派介面。

    參數：
        figsize (tuple): 圖表尺寸，格式為 (寬, 高)，單位為英吋。預設 (8, 6)
        parent (QWidget, optional): 父元件。預設 None
    """

    def __init__(self, figsize=(8, 6), parent=None):
        """
        初始化繪圖元件。

        建立 MatplotlibCanvas 和 NavigationToolbar2QT，
        以垂直佈局 (QVBoxLayout) 排列，工具列在上、畫布在下。
        佈局的邊距設為 0，讓圖表佔滿可用空間。
        """
        super().__init__(parent)
        # 建立底層畫布
        self.canvas = MatplotlibCanvas(figsize=figsize)
        # 建立 matplotlib 導航工具列，連結到畫布
        # 工具列提供：Home、Back/Forward、Pan、Zoom、Save 等按鈕
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        # 使用垂直佈局，邊距設為 0
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)  # 工具列在上方
        layout.addWidget(self.canvas)   # 畫布在下方

    def get_figure(self):
        """
        取得 Figure 物件（委派給內部的 MatplotlibCanvas）。

        返回：
            matplotlib.figure.Figure: 畫布的 Figure 物件
        """
        return self.canvas.get_figure()

    def get_axes(self):
        """
        取得 Axes 物件（委派給內部的 MatplotlibCanvas）。

        返回：
            matplotlib.axes.Axes: 畫布的預設 Axes 物件
        """
        return self.canvas.get_axes()

    def clear(self):
        """
        清除繪圖內容（委派給內部的 MatplotlibCanvas）。
        """
        self.canvas.clear()

    def refresh(self):
        """
        刷新畫面（委派給內部的 MatplotlibCanvas）。
        """
        self.canvas.refresh()
