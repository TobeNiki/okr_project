import networkx as nx
import numpy as np

G = nx.Graph()

G.add_edge('A', 'B')
G.add_edge('B', 'C')

#ラプラシアン行列
laplacian = nx.laplacian_matrix(G)

print(laplacian)


#ラプラシアン行列の固有値
laplacians = laplacian.toarray()
eig = np.linalg.eigh(laplacians)
print(eig)
#https://blog.uni-3.app/2020/02/19/network-spectral-clustering
#https://qiita.com/hoshi921/items/0e19e392bb2f1086bfbe
#https://qiita.com/kazetof/items/3d5bd3241dcf349fb24e
