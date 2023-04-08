import crayons
import numpy as np
from tqdm import tqdm
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from WLM import WeisfeilerLehmanMachine

import os , sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import geppy_hry as gep

import networkx as nx

def sur_encode(cnf,pop_G2V):
    for ind in pop_G2V:
        ind_enc = []
        for gene in range(len(ind)):
            for i in range(ind.head_length):
                ind_enc.append(ind[gene].head[i]._encode_name)
            for i in range(ind.tail_length):
                ind_enc.append(ind[gene].tail[i]._encode_name)
            ind.sur_encode = np.array(ind_enc)
    return pop_G2V

def G2V_initialization(cnf,pop_G2V):
    # produce Val_pop timinig is cgp intializer
    print(crayons.blue("### Start Initialize to"),end="")
    print(crayons.red(" G2V ###"))
    document_collections = []
    for name , obj in tqdm(enumerate(pop_G2V)):
        document_collections.append(graph2doc(ind=obj,name=name))
    G2V = Doc2Vec(document_collections,
                        vector_size=cnf.vector_size, 
                        window=cnf.window,       
                        min_count=cnf.min_count, 
                        dm=cnf.dm,
                        sample=cnf.sample,
                        workers=cnf.workers,
                        epochs=cnf.epochs,
                        alpha=cnf.alpha)
    return G2V

def graph2doc(ind,name):
    # initialization
    feature_dict = {}
    nodes, edges, labels = gep.graph(ind)
    ind.graph.add_edges_from(edges)
    ind.graph.add_nodes_from(nodes)

    for i in range(len(nodes)):
        feature_dict[i] = labels[i]

    ### VIZUALIZATION ###
    #from networkx.drawing.nx_agraph import graphviz_layout
    #pos = graphviz_layout(ind.graph, prog="dot")
    #nx.draw_networkx_nodes(ind.graph, pos)
    #nx.draw_networkx_edges(ind.graph, pos)
    #nx.draw_networkx_labels(ind.graph, pos, labels)
    #import matplotlib.pyplot as plt
    #plt.show()

    machine = WeisfeilerLehmanMachine(ind.graph, feature_dict, 1) # arity2 is dependence
    doc = TaggedDocument(words=machine.extracted_features, tags=["g_{}".format(name)])
    return doc
