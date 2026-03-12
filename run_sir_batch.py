"""Batch SIR experiment runner - parallel-friendly single network"""
import sys, os, json, time
gui_app_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gui_app')
sys.path.insert(0, gui_app_dir)
from core import algorithm_adapter as algo

def run_one(net_name, path, beta):
    t0 = time.time()
    G = algo.create_network_from_edgelist(path)
    net_attr = algo.compute_all_attributes(G)
    base = path.replace('.txt', '')
    tbet_path = base + '_tbet.txt'
    clos_path = base + '_clos.txt'
    if os.path.exists(tbet_path):
        net_attr = algo.append_attribute(net_attr, algo.read_pairvalue_file(tbet_path), 'node_betweenness')
    else:
        net_attr = algo.append_attribute(net_attr, algo.compute_betweenness(G), 'node_betweenness')
    if os.path.exists(clos_path):
        net_attr = algo.append_attribute(net_attr, algo.read_pairvalue_file(clos_path), 'node_closeness')
    else:
        net_attr = algo.append_attribute(net_attr, algo.compute_closeness(G), 'node_closeness')
    measures = sorted(['node_degree','node_betweenness','node_closeness','node_k-core','node_neighbor-core','node_pagerank','node_mv17'])
    result = algo.run_sir_experiment(G, net_attr, measures, top_k=1, mode=1, num_round=5000, num_time_step=50, rate_infection=beta, rate_recovery=1)
    rho50 = {m: result[m][-1] for m in measures if m in result}
    elapsed = time.time() - t0
    print(f'{net_name}: N={len(G.nodes())}, E={len(G.edges())}, beta={beta}, time={elapsed:.1f}s')
    for m in sorted(rho50.keys()):
        print(f'  {m}: {rho50[m]:.4f}')
    return rho50

if __name__ == '__main__':
    net_name = sys.argv[1]
    path = sys.argv[2]
    beta = float(sys.argv[3])
    outfile = sys.argv[4]
    result = run_one(net_name, path, beta)
    with open(outfile, 'w') as f:
        json.dump({net_name: result}, f)
