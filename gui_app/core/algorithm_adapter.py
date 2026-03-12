"""
Algorithm adapter (演算法轉接層)
================================
將舊版專案中散落於多個獨立腳本的核心演算法，統一封裝為無狀態(stateless)函式，
並針對 Python 3 / NetworkX 2+ 進行相容性修正。

舊版對應檔案總覽:
  - measure_node_attribute.py   : 節點屬性計算 (k-core, PageRank, clustering, entropy, MV17)
  - experiment1.py              : SIR 傳播實驗 (Top-K/Top-P 初始節點選取 + 繪圖)
  - code/sir_ranking_file_writer.py : 逐節點 SIR ranking (每個節點各跑一次 SIR)
  - code/analysis1.py           : 網路基礎統計分析 (N, E, <k>, k-core, assortativity...)
  - code/network_attribute_scatter.py : 多面板散佈圖矩陣
  - util/sir_model.py           : SIR 傳播模型核心 (S->I, I->R, 時間序列密度)
  - util/read_write_*.py        : 各類檔案 I/O (edgelist, pairvalue, pos, umsgpack)

主要修正:
  - G.selfloop_edges()         -> nx.selfloop_edges(G)         (NetworkX 2+ API)
  - nx.degree(G).values()      -> dict(G.degree())             (回傳型態改變)
  - G.neighbors(ni) 回傳 iterator -> list(G.neighbors(ni))     (不可直接 len/shuffle)
  - closeness_centrality 移除 normalized 參數 (新版 NetworkX 預設即為 normalized)
"""
import os
import networkx as nx
import numpy as np
import random
import math

# ============================================================================
# 常數定義
# ============================================================================
# 節點屬性鍵值 (與舊版 measure_node_attribute.py 第 24~44 行完全對應)
# 每個常數對應 net_attr[node_id] 字典中的一個 key
NODE_ID = 'node_id'                     # 節點編號 (整數)
NODE_POS_X = 'node_pos_x'              # 節點布局 X 座標 (來自 _pos.txt)
NODE_POS_Y = 'node_pos_y'              # 節點布局 Y 座標 (來自 _pos.txt)
NODE_DEGREE = 'node_degree'            # 節點度數 (degree)
NODE_CC = 'node_cc'                    # 節點群聚係數 (clustering coefficient)
NODE_NEIGHBOR_CORE = 'node_neighbor-core'      # 鄰居的鄰居 k-core 總和取 log2
NODE_NEIGHBOR_DEGREE = 'node_neighbor-degree'  # 鄰居的鄰居 degree 總和取 log2
NODE_BETWEENNESS = 'node_betweenness'  # 介數中心性 (betweenness centrality)
NODE_CLOSENESS = 'node_closeness'      # 接近中心性 (closeness centrality)
NODE_KCORE = 'node_k-core'            # k-core 數
NODE_KCORE_ENTROPY = 'node_k-core-entropy'     # 鄰居 k-core 分佈的 Shannon 熵
NODE_PAGERANK = 'node_pagerank'        # PageRank 值
NODE_MV17 = 'node_mv17'               # 本論文提出的指標: k-core entropy * neighbor-degree
NODE_APPROX_BETWEENNESS = 'node_approx-betweenness'  # 近似介數中心性 (新版新增)

# PageRank 參數 (舊版 measure_node_attribute.py 第 47~48 行)
PAGERANK_ALPHA = 0.85   # Google 使用的阻尼係數
PAGERANK_MAX_ITER = 100  # 迭代上限

# 繪圖常數 (舊版 experiment1.py 第 24~43 行)
# 用於 SIR 傳播曲線，每條曲線一個顏色，最多 8 種量測指標
COLOR_LIST = ['gray', 'orange', 'y', 'b', 'c', 'm', 'r', 'k']
MARKER_LIST = ['o', '^', 's', 'h', 'x', '+', '.', '*']


# ============================================================================
# 網路建立 (對應舊版 measure_node_attribute.py 第 53~60 行 create_network())
# ============================================================================

def create_network_from_edgelist(filepath):
    """從 edgelist 文字檔建立無向圖。

    對應舊版: measure_node_attribute.py -> create_network(edgelist)
    以及     util/read_write_edgelist.py -> read_edge_list(filename)

    舊版分兩步: (1) read_edge_list 讀入字串列表, (2) create_network 建圖。
    新版合併為一步，直接讀檔並建圖。

    修正:
      - 舊版 G.remove_edges_from(G.selfloop_edges())
        新版 G.remove_edges_from(list(nx.selfloop_edges(G)))
        因為 NetworkX 2+ 移除了 G.selfloop_edges() 方法。

    Args:
        filepath: edgelist 檔案路徑，每行格式為 "node1 node2"

    Returns:
        nx.Graph: 無向圖 (自迴路已移除)
    """
    edge_list = []
    with open(filepath, mode="r") as f:
        for line in f:
            edge_list.append(line.strip())
    G = nx.parse_edgelist(edge_list, nodetype=int)
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    G = G.to_undirected()
    return G


def extract_gcc(G):
    """提取最大連通子圖 (Giant Connected Component, GCC)。

    舊版中此步驟由使用者手動在檔名加 '_gcc' 來區分，
    程式假設輸入已是 GCC。新版提供 GUI 按鈕讓使用者即時提取。

    Args:
        G: NetworkX 圖

    Returns:
        nx.Graph: 最大連通子圖的副本
    """
    if G is None or len(G.nodes()) == 0:
        return G
    gcc_nodes = max(nx.connected_components(G), key=len)
    return G.subgraph(gcc_nodes).copy()


def compute_spring_layout(G, iterations=50):
    """計算彈簧布局 (spring layout) 供視覺化使用。

    舊版中節點座標由外部 _pos.txt 檔案提供 (util/read_write_pos.py)。
    新版可即時計算，也可從檔案載入。

    Args:
        G: NetworkX 圖
        iterations: 力導向演算法迭代次數 (越多越穩定，但越慢)

    Returns:
        dict: {node_id: np.array([x, y]), ...}
    """
    return nx.spring_layout(G, iterations=iterations)


# ============================================================================
# 中心性計算 (對應舊版 measure_node_attribute.py 主流程中的步驟 3)
# ============================================================================
# 舊版中 betweenness 和 closeness 是預先計算好存成 _tbet.txt / _clos.txt，
# 再由 append_new_attribute() 合併到 net_attr。
# 新版可在 GUI 中即時計算，也可從檔案載入。

def compute_betweenness(G):
    """計算正規化介數中心性 (normalized betweenness centrality)。

    對應舊版: 預先計算並存入 edgelist/*_tbet.txt
    新版在 GUI 的 Node Attributes tab 中由 CentralityComputeWorker 呼叫。

    Args:
        G: NetworkX 圖

    Returns:
        dict: {node_id: float, ...}
    """
    return nx.betweenness_centrality(G, normalized=True, weight=None)


def compute_closeness(G):
    """計算接近中心性 (closeness centrality)。

    對應舊版: 預先計算並存入 edgelist/*_clos.txt
    新版在 GUI 的 Node Attributes tab 中由 CentralityComputeWorker 呼叫。

    注意: NetworkX 新版已移除 closeness_centrality 的 normalized 參數，
    預設行為即為正規化 (normalized=True)，無須額外指定。

    Args:
        G: NetworkX 圖

    Returns:
        dict: {node_id: float, ...}
    """
    return nx.closeness_centrality(G)


def compute_approx_betweenness(G, epsilon=0.1):
    """計算近似介數中心性 (approximate current flow betweenness)。

    舊版無此功能，為新版新增。適用於大型網路 (N > 5000) 時，
    精確 betweenness 太慢的情況。

    注意: 此演算法要求圖必須是連通的 (connected)，否則會拋出 ValueError。
    使用者應先提取 GCC。

    Args:
        G: NetworkX 圖 (必須連通)
        epsilon: 精度參數，越小越精確但越慢

    Returns:
        dict: {node_id: float, ...}

    Raises:
        ValueError: 若圖不連通
    """
    if not nx.is_connected(G):
        raise ValueError("Approximate betweenness requires a connected graph. "
                         "Please extract GCC first.")
    return nx.approximate_current_flow_betweenness_centrality(
        G, epsilon=epsilon, normalized=True, weight=None, solver='full')


# ============================================================================
# 節點屬性計算 (對應舊版 measure_node_attribute.py 第 70~156 行)
# ============================================================================

def _compute_kcore_entropy(G, node_core, normalize=False):
    """計算每個節點的 k-core 鄰居熵 (Shannon entropy)。

    對應舊版: measure_node_attribute.py -> compute_kcore_entropy()

    演算法: 對節點 ni 的所有鄰居，統計其 k-core 值的分佈，
    計算該分佈的 Shannon 熵 H = -sum(p * log2(p))。
    鄰居數 <= 1 時，熵定義為 0。

    修正:
      - 舊版 neighbor_list = G.neighbors(ni) 可直接 len()
        新版 neighbor_list = list(G.neighbors(ni)) 因為回傳 iterator

    Args:
        G: NetworkX 圖
        node_core: {node_id: k-core_number} 字典
        normalize: 是否以 log2(max_kcore) 正規化

    Returns:
        dict: {node_id: entropy_value, ...}
    """
    entropy_dict = {}
    for ni in G.nodes():
        neighbor_list = list(G.neighbors(ni))
        if len(neighbor_list) > 1:
            neighbors_core_list = [node_core[x] for x in neighbor_list]
            counts = np.bincount(neighbors_core_list)
            probs = counts[np.nonzero(counts)] / len(neighbors_core_list)
            entropy = -np.sum(probs * np.log2(probs))
            if normalize:
                max_kcore = max(node_core.values())
                if max_kcore > 1:
                    entropy_dict[ni] = entropy / np.log2(max_kcore)
                else:
                    entropy_dict[ni] = 0.0
            else:
                entropy_dict[ni] = float(entropy)
        else:
            entropy_dict[ni] = 0.0
    return entropy_dict


def _compute_neighbor_attribute(G, node_attr, is_cc=False):
    """計算鄰居的鄰居屬性聚合值。

    對應舊版: measure_node_attribute.py -> compute_neighbor_attribute()

    有兩種模式:
      (a) is_cc=True  (NODE_CC 模式):
          neighbor_attr[ni] = 1 - mean(鄰居的 clustering coefficient)
          用於計算「鄰居的群聚係數互補值」(舊版中被註解未使用)

      (b) is_cc=False (預設, NODE_NEIGHBOR_CORE / NODE_NEIGHBOR_DEGREE 模式):
          對節點 ni 的每個鄰居 nj，加總 nj 的所有鄰居的屬性值，
          最後取 log2: neighbor_attr[ni] = log2(sum of all 2nd-hop attr)
          這是本論文 MV17 指標的組成元件之一。

    修正:
      - 舊版 nj_neighbor_list = G.neighbors(nj) 可直接迭代加總
        新版需 list(G.neighbors(nj)) 以相容 NetworkX 2+

    Args:
        G: NetworkX 圖
        node_attr: {node_id: attribute_value} 字典
        is_cc: True 表示計算群聚係數互補值模式

    Returns:
        dict: {node_id: aggregated_value, ...}
    """
    neighbor_attr = {}
    for ni in G.nodes():
        neighbor_list = list(G.neighbors(ni))
        if is_cc:
            if len(neighbor_list) == 0:
                neighbor_attr[ni] = 0.0
            else:
                vals = [node_attr[n] for n in neighbor_list]
                neighbor_attr[ni] = float(1.0 - np.mean(vals))
        else:
            attr_value = 0
            for nj in neighbor_list:
                nj_neighbors = list(G.neighbors(nj))
                attr_value += sum(node_attr[n] for n in nj_neighbors)
            neighbor_attr[ni] = float(np.log2(attr_value)) if attr_value > 0 else 0.0
    return neighbor_attr


def compute_all_attributes(G, progress_callback=None):
    """計算所有基本節點屬性 (不含 betweenness / closeness)。

    對應舊版: measure_node_attribute.py -> measure_node_attribute(G, net_attr)
    舊版第 111~156 行的完整流程。

    計算順序 (與舊版一致):
      1. k-core number         -> NODE_KCORE
      2. k-core entropy        -> NODE_KCORE_ENTROPY
      3. PageRank              -> NODE_PAGERANK
      4. Clustering coeff.     -> NODE_CC
      5. Degree                -> NODE_DEGREE
      6. Neighbor-core (2-hop) -> NODE_NEIGHBOR_CORE
      7. Neighbor-degree (2-hop) -> NODE_NEIGHBOR_DEGREE
      8. MV17 (proposed)       -> NODE_MV17 = entropy * neighbor_degree

    注意: Betweenness 和 Closeness 由 CentralityComputeWorker 另外計算，
    因為它們耗時較長，允許使用者選擇性計算。

    Args:
        G: NetworkX 圖
        progress_callback: 回呼函式 (percent: int, message: str)

    Returns:
        dict: {node_id: {attr_key: value, ...}, ...} 巢狀字典
    """
    net_attr = {}
    for ni in G.nodes():
        net_attr[ni] = {NODE_ID: ni}

    if progress_callback:
        progress_callback(10, "Computing k-core...")
    node_core = nx.core_number(G)
    node_kcore_entropy = _compute_kcore_entropy(G, node_core)

    if progress_callback:
        progress_callback(25, "Computing PageRank...")
    node_pagerank = nx.pagerank(G, alpha=PAGERANK_ALPHA, max_iter=PAGERANK_MAX_ITER)

    if progress_callback:
        progress_callback(40, "Computing clustering coefficient & degree...")
    node_cc = nx.clustering(G)
    node_degree = dict(G.degree())

    if progress_callback:
        progress_callback(55, "Computing neighbor-core...")
    node_neighbor_core = _compute_neighbor_attribute(G, node_core)

    if progress_callback:
        progress_callback(70, "Computing neighbor-degree...")
    node_neighbor_degree = _compute_neighbor_attribute(G, node_degree)

    if progress_callback:
        progress_callback(85, "Assembling attributes...")
    for ni in G.nodes():
        net_attr[ni][NODE_KCORE] = node_core[ni]
        net_attr[ni][NODE_PAGERANK] = node_pagerank[ni]
        net_attr[ni][NODE_CC] = node_cc[ni]
        net_attr[ni][NODE_DEGREE] = node_degree[ni]
        net_attr[ni][NODE_NEIGHBOR_CORE] = node_neighbor_core[ni]
        net_attr[ni][NODE_NEIGHBOR_DEGREE] = node_neighbor_degree[ni]
        net_attr[ni][NODE_KCORE_ENTROPY] = node_kcore_entropy[ni]
        # MV17: 本論文提出的核心指標 = k-core entropy * neighbor-degree
        # 對應舊版第 155 行
        net_attr[ni][NODE_MV17] = node_kcore_entropy[ni] * node_neighbor_degree[ni]

    if progress_callback:
        progress_callback(100, "Done.")
    return net_attr


def append_attribute(net_attr, attr_dict, attr_name):
    """將單一屬性字典合併到 net_attr 中。

    對應舊版: measure_node_attribute.py -> append_new_attribute()

    舊版用於將預先計算好的 betweenness / closeness / pos 加入 net_attr。
    新版在 GUI 載入 _tbet.txt / _clos.txt 檔案時使用。

    Args:
        net_attr: 現有的 {node_id: {attrs}} 巢狀字典
        attr_dict: {node_id: value} 待合併的單一屬性
        attr_name: 屬性名稱 (如 NODE_BETWEENNESS)

    Returns:
        dict: 更新後的 net_attr (原地修改)
    """
    for ni in net_attr:
        if ni in attr_dict:
            net_attr[ni][attr_name] = attr_dict[ni]
    return net_attr


def dict_normalized(attr_dict):
    """將屬性字典做 Min-Max 正規化到 [0, 1] 範圍。

    對應舊版: measure_node_attribute.py -> dict_normalized()

    修正:
      - 舊版在 attr_max == attr_min 時會產生除以零 (NaN)
        新版加入 denom == 0 的防護，此時回傳全 0.0

    Args:
        attr_dict: {node_id: numeric_value} 字典

    Returns:
        dict: {node_id: normalized_value} 正規化後的字典
    """
    if not attr_dict:
        return {}
    attr_max = max(attr_dict.values())
    attr_min = min(attr_dict.values())
    denom = attr_max - attr_min
    if denom == 0:
        return {k: 0.0 for k in attr_dict}
    return {k: (v - attr_min) / denom for k, v in attr_dict.items()}


# ============================================================================
# SIR 傳播模型 (對應舊版 util/sir_model.py)
# ============================================================================
# SIR 模型有兩個版本:
#   (1) sir_propagation()          - 回傳時間序列密度 (用於傳播曲線繪圖)
#       對應 util/sir_model.py -> propagation()
#   (2) _sir_ranking_propagation() - 回傳純量平均密度 (用於逐節點排名)
#       對應 code/sir_ranking_file_writer.py -> propagation()
# 兩者的關鍵差異: (1) 有 neighbor shuffle, (2) 沒有 shuffle

def _sir_convert_s_to_i(G, node_susceptible, node_infected, rate_infection):
    """SIR 模型: S -> I 轉換 (含鄰居隨機洗牌)。

    對應舊版: util/sir_model.py -> convert_susceptible_to_infected()

    對每個已感染節點 ni，遍歷其鄰居 (隨機順序)，
    若鄰居在易感集合中且隨機數 < rate_infection，則感染之。

    修正:
      - 舊版 neighbor_list = G.neighbors(ni); np.random.shuffle(neighbor_list)
        新版 neighbor_list = list(G.neighbors(ni)); random.shuffle(neighbor_list)
        因為 NetworkX 2+ 的 neighbors() 回傳 iterator

    Args:
        G: NetworkX 圖
        node_susceptible: 易感節點列表
        node_infected: 已感染節點列表
        rate_infection: 感染機率 (beta)

    Returns:
        list: 本時間步新增的感染節點
    """
    target_set = set(node_susceptible)
    infected_list = []
    for ni in node_infected:
        neighbor_list = list(G.neighbors(ni))
        random.shuffle(neighbor_list)
        for nb in neighbor_list:
            if nb in target_set and random.random() < rate_infection:
                infected_list.append(nb)
    return infected_list


def _sir_convert_i_to_r(node_infected, rate_recovery):
    """SIR 模型: I -> R 轉換。

    對應舊版: util/sir_model.py -> convert_infected_to_recovered()

    rate_recovery == 1 時，所有感染節點立即康復 (SIR 特例)。
    否則，每個感染節點以 rate_recovery 機率康復。

    Args:
        node_infected: 已感染節點列表
        rate_recovery: 康復機率 (gamma)

    Returns:
        list: 本時間步新增的康復節點
    """
    if rate_recovery == 1:
        return list(node_infected)
    return [ni for ni in node_infected if random.random() < rate_recovery]


def sir_propagation(G, initial_nodes, num_round=1000, num_time_step=50,
                    rate_infection=0.1, rate_recovery=1,
                    progress_callback=None, cancel_check=None):
    """SIR 傳播模擬 (時間序列版本，用於繪製傳播曲線)。

    對應舊版: util/sir_model.py -> propagation()

    流程:
      1. 執行 num_round 輪模擬
      2. 每輪有 num_time_step+1 個時間步 (t=0 為初始化)
      3. t=0: 將 initial_nodes 設為感染
      4. t>0: S->I (含 shuffle), I->R
      5. 每個時間步記錄累計康復人數
      6. 最終取平均: round_density[t] / (num_round * N)

    與舊版的差異:
      - 舊版 num_time_step += 1 直接修改參數
        新版 num_steps = num_time_step + 1 避免副作用
      - 新增 cancel_check 支援 GUI 取消
      - 新增 progress_callback 支援進度回報

    Args:
        G: NetworkX 圖
        initial_nodes: 初始感染節點列表
        num_round: 模擬輪數 (預設 1000)
        num_time_step: 時間步數 (預設 50，實際執行 51 步含 t=0)
        rate_infection: 感染機率 beta (預設 0.1)
        rate_recovery: 康復機率 gamma (預設 1)
        progress_callback: 進度回呼 (percent, message)
        cancel_check: 取消檢查回呼，回傳 True 表示取消

    Returns:
        list: 長度為 num_time_step+1 的密度時間序列 [d(0), d(1), ..., d(T)]
              每個值為 R(t) / N 的平均
        None: 若被取消或 initial_nodes 為空
    """
    if not initial_nodes:
        return None
    num_steps = num_time_step + 1
    round_density = [0.0] * num_steps
    n_nodes = len(G.nodes())

    for i in range(num_round):
        if cancel_check and cancel_check():
            return None

        node_susceptible = list(G.nodes())
        node_infected = []
        node_recovered = []

        for t in range(num_steps):
            if t == 0:
                node_infected.extend(list(initial_nodes))
                node_susceptible = list(set(node_susceptible) - set(node_infected))
            else:
                current_infected = _sir_convert_s_to_i(
                    G, node_susceptible, node_infected, rate_infection)
                current_recovery = _sir_convert_i_to_r(node_infected, rate_recovery)
                node_recovered.extend(current_recovery)
                node_infected.extend(current_infected)
                node_infected = list(set(node_infected) - set(current_recovery))
                node_susceptible = list(set(node_susceptible) - set(current_infected))

            round_density[t] += len(node_recovered)

        if progress_callback and (i + 1) % max(1, num_round // 20) == 0:
            progress_callback(int((i + 1) / num_round * 100),
                              f"Round {i + 1}/{num_round}")

    round_density = [d / (num_round * n_nodes) for d in round_density]
    return round_density


def _sir_ranking_propagation(G, initial_nodes, num_round=1000, num_time_step=50,
                              rate_infection=0.1, rate_recovery=1):
    """SIR 傳播模擬 (純量版本，用於逐節點排名)。

    對應舊版: code/sir_ranking_file_writer.py -> propagation()

    與 sir_propagation() 的關鍵差異:
      (1) 不做鄰居 shuffle (直接按 G.neighbors 順序遍歷)
      (2) 不記錄時間序列，只回傳最終的 R/N 純量
      (3) 時間步數不加 1 (range(num_time_step) 而非 range(num_time_step+1))
          這與舊版 sir_ranking_file_writer.py 行為一致

    此差異是舊版兩個 SIR 實作之間的真實差異，並非 bug。

    Args:
        G: NetworkX 圖
        initial_nodes: 初始感染節點列表 (通常只有一個節點)
        num_round: 模擬輪數
        num_time_step: 時間步數
        rate_infection: 感染機率 beta
        rate_recovery: 康復機率 gamma

    Returns:
        float: 平均康復密度 R/N
    """
    if not initial_nodes:
        return 0.0
    n_nodes = len(G.nodes())
    average_density = 0

    for i in range(num_round):
        node_susceptible = list(G.nodes())
        node_infected = []
        node_recovered = []

        for t in range(num_time_step):
            if t == 0:
                node_infected.extend(list(initial_nodes))
                node_susceptible = list(set(node_susceptible) - set(node_infected))
            else:
                # S -> I (不做 shuffle，忠實對應舊版 sir_ranking_file_writer.py)
                target_set = set(node_susceptible)
                current_infected = []
                for ni in node_infected:
                    for nb in G.neighbors(ni):
                        if nb in target_set and random.random() < rate_infection:
                            current_infected.append(nb)
                # I -> R
                if rate_recovery == 1:
                    current_recovery = list(node_infected)
                else:
                    current_recovery = [ni for ni in node_infected if random.random() < rate_recovery]

                node_recovered.extend(current_recovery)
                node_infected.extend(current_infected)
                node_infected = list(set(node_infected) - set(current_recovery))
                node_susceptible = list(set(node_susceptible) - set(current_infected))

        average_density += len(node_recovered)

    average_density = average_density / (num_round * n_nodes)
    return average_density


def compute_sir_ranking(G, num_round=1000, num_time_step=50,
                        rate_infection_list=None, rate_recovery=1,
                        progress_callback=None, cancel_check=None):
    """計算逐節點 SIR 排名 (每個節點各自作為唯一初始感染者)。

    對應舊版: code/sir_ranking_file_writer.py -> compute_SIR_ranking()

    流程:
      1. 對每個感染率 rate，對每個節點 ni:
         以 [ni] 為初始感染節點，執行 _sir_ranking_propagation()
      2. 轉置結果: 從 {rate: {node: density}} 轉為 {node: {rate: density}}

    注意: 此函式的時間複雜度為 O(N * num_round * num_time_step * avg_degree)，
    對大型網路 (N > 500) 可能非常耗時。GUI 會在 N > 500 時顯示警告。

    Args:
        G: NetworkX 圖
        num_round: 每個節點的模擬輪數
        num_time_step: 每輪的時間步數
        rate_infection_list: 感染率列表 (可多個)
        rate_recovery: 康復率
        progress_callback: 進度回呼
        cancel_check: 取消檢查回呼

    Returns:
        dict: {node_id: {str(rate): density, ...}, ...}
        None: 若被取消
    """
    if rate_infection_list is None:
        rate_infection_list = [0.1]

    nodes = sorted(G.nodes())
    total_work = len(rate_infection_list) * len(nodes)
    work_done = 0
    sir_result = {}

    for rate in rate_infection_list:
        rate_key = str(rate)
        temp_result = {}
        for ni in nodes:
            if cancel_check and cancel_check():
                return None
            temp_result[ni] = _sir_ranking_propagation(
                G, [ni], num_round, num_time_step, rate, rate_recovery)
            work_done += 1
            if progress_callback and work_done % max(1, total_work // 50) == 0:
                progress_callback(
                    int(work_done / total_work * 100),
                    f"Rate={rate}, Node {ni} ({work_done}/{total_work})")
        sir_result[rate_key] = temp_result

    # 轉置: 從 per-rate 變為 per-node (與舊版第 108~114 行一致)
    ranking_result = {}
    for ni in nodes:
        node_result = {}
        for rate_key in sorted(sir_result.keys()):
            node_result[rate_key] = sir_result[rate_key][ni]
        ranking_result[ni] = node_result

    if progress_callback:
        progress_callback(100, "Done.")
    return ranking_result


def retrieve_topk_nodes(network_attr, specified_attr, top_k=1):
    """取出指定屬性值最高的前 K 個節點。

    對應舊版: experiment1.py -> retrieve_topk_nodes()

    用於 SIR 實驗: 選取某指標排名最高的節點作為初始感染者。

    修正:
      - 舊版 sorted_list[i][1][str(specified_attr)] 可能 KeyError
        新版使用 .get(specified_attr, 0) 防護
      - 舊版不檢查 top_k 是否超過節點數
        新版使用 min(top_k, len(sorted_list)) 防護

    Args:
        network_attr: {node_id: {attr_key: value}} 巢狀字典
        specified_attr: 排序依據的屬性 key
        top_k: 取前幾名

    Returns:
        list: top-K 節點 ID 列表
    """
    sorted_list = sorted(network_attr.items(),
                         key=lambda x: x[1].get(specified_attr, 0), reverse=True)
    return [sorted_list[i][1][NODE_ID] for i in range(min(top_k, len(sorted_list)))]


def run_sir_experiment(G, net_attr, measurement_list, top_k=1, top_p=0.01, mode=1,
                       num_round=1000, num_time_step=50,
                       rate_infection=0.1, rate_recovery=1,
                       progress_callback=None, cancel_check=None):
    """執行 SIR 傳播實驗: 對多種量測指標比較傳播效果。

    對應舊版: experiment1.py -> network_propagation()

    流程:
      對 measurement_list 中的每個指標:
        1. 以該指標選取 top-K 或 top-P 節點作為初始感染者
        2. 執行 sir_propagation() 取得時間序列
        3. 儲存結果供繪圖比較

    兩種模式 (與舊版一致):
      - mode=1: Top-K 模式，取排名前 top_k 個節點
      - mode=2: Top-P 模式，取排名前 top_p * N 個節點

    Args:
        G: NetworkX 圖
        net_attr: 節點屬性字典
        measurement_list: 量測指標名稱列表 (如 ['node_degree', 'node_mv17', ...])
        top_k: Top-K 模式的 K 值
        top_p: Top-P 模式的比例
        mode: 1=Top-K, 2=Top-P
        num_round: 模擬輪數
        num_time_step: 時間步數
        rate_infection: 感染率 beta
        rate_recovery: 康復率 gamma
        progress_callback: 進度回呼
        cancel_check: 取消檢查回呼

    Returns:
        dict: {measure_name: [density_time_series], ...}
    """
    propagation_result = {}
    total = len(measurement_list)

    for idx, measure in enumerate(measurement_list):
        if cancel_check and cancel_check():
            return propagation_result

        if progress_callback:
            progress_callback(int(idx / total * 100),
                              f"[{idx + 1}/{total}] {measure}")

        if mode == 1:
            node_initial = retrieve_topk_nodes(net_attr, measure, top_k)
        else:
            num_initial = max(1, int(len(G.nodes()) * top_p))
            node_initial = retrieve_topk_nodes(net_attr, measure, num_initial)

        result = sir_propagation(
            G, node_initial, num_round, num_time_step,
            rate_infection, rate_recovery,
            cancel_check=cancel_check)

        if result is None:
            return propagation_result
        propagation_result[measure] = result

    if progress_callback:
        progress_callback(100, "Done.")
    return propagation_result


# ============================================================================
# 網路基礎統計分析 (對應舊版 code/analysis1.py)
# ============================================================================

def compute_basic_analysis(G):
    """計算網路的 11 項基礎統計量。

    對應舊版: code/analysis1.py -> network_basic_analysis() 中的內部計算

    統計量:
      N          : 節點數
      E          : 邊數
      <c>        : 平均群聚係數
      k_max      : 最大度數
      <k>        : 平均度數
      <k^2>      : 平均度數平方 (舊版計算但未寫入檔案，新版新增)
      ks_max     : 最大 k-core
      <ks>       : 平均 k-core
      r          : 度度相關性 (assortativity)
      H          : 度數異質性 (heterogeneity) = <k^2> / <k>^2
      beta_thd   : 傳播閾值 = <k> / <k^2>

    修正:
      - 舊版 nx.degree(G).values() 在 NetworkX 2+ 會報錯
        新版使用 [d for _, d in G.degree()]
      - 新增空圖防護 (num_nodes == 0 時回傳全零)

    Args:
        G: NetworkX 圖

    Returns:
        dict: 包含 11 項統計量的字典
    """
    num_nodes = len(G.nodes())
    num_edges = len(G.edges())
    if num_nodes == 0:
        return {k: 0 for k in ['N', 'E', '<c>', 'k_max', '<k>', '<k^2>',
                                'ks_max', '<ks>', 'r', 'H', 'beta_thd']}
    avg_cc = float(np.mean(list(nx.clustering(G).values())))
    degrees = [d for _, d in G.degree()]
    max_degree = max(degrees)
    avg_degree = float(np.mean(degrees))
    avg_degree_sq = float(np.mean(np.power(degrees, 2)))
    core_numbers = list(nx.core_number(G).values())
    max_kcore = max(core_numbers)
    avg_kcore = float(np.mean(core_numbers))
    assortativity = float(nx.degree_pearson_correlation_coefficient(G))
    heterogeneity = avg_degree_sq / (avg_degree ** 2) if avg_degree > 0 else 0
    beta_threshold = avg_degree / avg_degree_sq if avg_degree_sq > 0 else 0

    return {
        'N': num_nodes,
        'E': num_edges,
        '<c>': round(avg_cc, 4),
        'k_max': max_degree,
        '<k>': round(avg_degree, 4),
        '<k^2>': round(avg_degree_sq, 4),
        'ks_max': max_kcore,
        '<ks>': round(avg_kcore, 4),
        'r': round(assortativity, 4),
        'H': round(heterogeneity, 4),
        'beta_thd': round(beta_threshold, 4),
    }


def batch_network_analysis(folder_edgelist, file_name_list,
                           progress_callback=None, cancel_check=None):
    """批次分析多個網路的基礎統計量。

    對應舊版: code/analysis1.py -> network_basic_analysis()
    舊版的主流程: 對 file_name_list 中的每個網路名稱，
    讀取 edgelist，建圖，計算統計量。

    新版新增: progress_callback / cancel_check 支援 GUI 非同步操作。

    Args:
        folder_edgelist: edgelist 檔案所在資料夾路徑
        file_name_list: 網路名稱列表 (不含 .txt 副檔名)
        progress_callback: 進度回呼
        cancel_check: 取消檢查回呼

    Returns:
        dict: {network_name: {statistics_dict}, ...}
    """
    analysis_result = {}
    total = len(file_name_list)
    for idx, name in enumerate(file_name_list):
        if cancel_check and cancel_check():
            return analysis_result
        if progress_callback:
            progress_callback(int(idx / total * 100), f"Analyzing {name}...")
        filepath = os.path.join(folder_edgelist, name + '.txt')
        if not os.path.exists(filepath):
            continue
        G = create_network_from_edgelist(filepath)
        analysis_result[name] = compute_basic_analysis(G)
    if progress_callback:
        progress_callback(100, "Done.")
    return analysis_result


def write_analysis_result(filepath, analysis_result):
    """將批次分析結果寫入文字檔。

    對應舊版: code/analysis1.py -> write_analysis_result()

    輸出格式: 空白分隔的表格，第一行為標頭。
    與舊版的差異: 新版多輸出 <k^2> 欄位。

    Args:
        filepath: 輸出檔案路徑
        analysis_result: {network_name: {stats_dict}} 字典
    """
    headers = ['Network', 'Nodes', 'Edges', 'Avg._c.c.', 'Max_degree', 'Avg._degree',
               '<k^2>', 'Max_k-core', 'Avg._k-core', 'assortativity',
               'degree_heterogeneity', 'beta_threshold']
    keys = ['N', 'E', '<c>', 'k_max', '<k>', '<k^2>', 'ks_max', '<ks>', 'r', 'H', 'beta_thd']
    with open(filepath, mode="w") as f:
        f.write(' '.join(headers) + '\n')
        for name in sorted(analysis_result.keys()):
            row = analysis_result[name]
            vals = [str(row.get(k, '')) for k in keys]
            f.write(name + ' ' + ' '.join(vals) + '\n')


# ============================================================================
# 檔案 I/O (對應舊版 util/ 目錄下的各 read_write_*.py)
# ============================================================================

def read_pairvalue_file(filepath):
    """讀取 node_id - value 配對檔案。

    對應舊版: util/read_write_pairvalue.py -> read_pairvalue_file()

    檔案格式: 每行 "node_id value"，例如 "42 0.123456"
    用於載入預先計算好的 betweenness (_tbet.txt) 或 closeness (_clos.txt)。

    Args:
        filepath: 檔案路徑

    Returns:
        dict: {node_id(int): value(float), ...}
    """
    pairvalue = {}
    with open(filepath, mode="r") as f:
        for line in f:
            pair = line.strip().split()
            if len(pair) >= 2:
                pairvalue[int(pair[0])] = float(pair[1])
    return pairvalue


def read_pos_file(filepath):
    """讀取節點座標檔案。

    對應舊版: util/read_write_pos.py -> read_pos_file()

    檔案格式: 每行 "node_id x y"
    用於載入預先計算好的布局座標 (_pos.txt)。

    Args:
        filepath: 檔案路徑

    Returns:
        dict: {node_id(int): np.array([x, y]), ...}
    """
    pos = {}
    with open(filepath, mode="r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                pos[int(parts[0])] = np.array([float(parts[1]), float(parts[2])])
    return pos


def write_propagation_result(filepath, propagation_result):
    """將 SIR 傳播結果寫入文字檔。

    對應舊版: experiment1.py -> write_propagation_result()

    檔案格式: 每行 "measure_name val1 val2 val3 ..."

    Args:
        filepath: 輸出檔案路徑
        propagation_result: {measure_name: [density_values]} 字典
    """
    with open(filepath, mode="w") as f:
        for key in sorted(propagation_result.keys()):
            vals = ' '.join(str(v) for v in propagation_result[key])
            f.write(f"{key} {vals}\n")


def read_propagation_result(filepath):
    """從文字檔讀取 SIR 傳播結果。

    對應舊版: code/experiment1_draw plot.py -> read_propagation_result()

    修正:
      - 舊版將 value 存為字串列表 (content[1:len(content)])
        新版將 value 轉為 float 列表，避免繪圖時的型態問題

    Args:
        filepath: 檔案路徑

    Returns:
        dict: {measure_name: [float_values]} 字典
    """
    result = {}
    with open(filepath, mode="r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                name = parts[0]
                values = [float(v) for v in parts[1:]]
                result[name] = values
    return result


def write_edgelist(filepath, G):
    """將圖寫出為 edgelist 格式。

    對應舊版: util/read_write_edgelist.py -> write_edge_list()

    Args:
        filepath: 輸出檔案路徑
        G: NetworkX 圖
    """
    nx.write_edgelist(G, path=filepath, data=False)


def write_pairvalue_file(filepath, G, pairvalue):
    """寫出 node_id - value 配對檔案。

    對應舊版: util/read_write_pairvalue.py -> write_pairvalue_file()

    用於儲存計算結果 (如 betweenness, SIR ranking)。
    按照圖中節點順序寫出，不在字典中的節點寫 0。

    Args:
        filepath: 輸出檔案路徑
        G: NetworkX 圖 (決定寫出哪些節點)
        pairvalue: {node_id: value} 或 {node_id: {rate: value}} 字典
    """
    with open(filepath, mode="w") as f:
        for n in G.nodes():
            val = pairvalue.get(n, 0)
            f.write(f"{n} {val}\n")


def write_pos_file(filepath, pos):
    """寫出節點座標檔案。

    對應舊版: util/read_write_pos.py -> write_pos_file()

    Args:
        filepath: 輸出檔案路徑
        pos: {node_id: np.array([x, y])} 字典
    """
    with open(filepath, mode="w") as f:
        for nid, coord in pos.items():
            f.write(f"{nid} {coord[0]} {coord[1]}\n")


# umsgpack I/O (對應舊版 util/read_write_umsgpack.py)
# umsgpack 為可選依賴，未安裝時提供 fallback 錯誤訊息
try:
    import umsgpack

    def read_umsgpack_data(filepath):
        """讀取 umsgpack 二進位格式的屬性資料。

        對應舊版: util/read_write_umsgpack.py -> read_umsgpack_data()

        umsgpack 格式用於儲存完整的 net_attr 字典，包含所有已計算的屬性。
        比純文字格式更快且更緊湊。

        Args:
            filepath: .umsgpack 檔案路徑

        Returns:
            dict: 反序列化後的 Python 物件 (通常是 net_attr 字典)
        """
        with open(filepath, 'rb') as f:
            data = f.read()
        return umsgpack.unpackb(data)

    def write_umsgpack_data(filepath, data):
        """將資料寫入 umsgpack 二進位格式。

        對應舊版: util/read_write_umsgpack.py -> write_umsgpack_data()

        Args:
            filepath: 輸出的 .umsgpack 檔案路徑
            data: 要序列化的 Python 物件
        """
        packed = umsgpack.packb(data)
        with open(filepath, 'wb') as f:
            f.write(packed)

except ImportError:
    def read_umsgpack_data(filepath):
        raise ImportError("umsgpack not installed")

    def write_umsgpack_data(filepath, data):
        raise ImportError("umsgpack not installed")
