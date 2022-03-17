def search_above_node(node, digraph):
    for i in digraph:
        if node in digraph[i]:
            return i
def digraph_to_graph(digraph):
    graph = {}
    for i in digraph:
        above_node = search_above_node(i, digraph)
        graph[i] = copy(digraph[i] + [above_node,]) if above_node!=None else copy(digraph[i]) #copy
    return graph
def graph_to_digraph(graph, root_node):
    digraph = {}
    stack = [root_node,]
    node = root_node
    while stack:
        if node not in digraph:
            digraph[node] = [i for i in graph[node] if i not in  digraph]
            stack.extend(digraph[node])
            node = stack.pop()
    return digraph
def hash_graph(graph):
    for i in graph:
        graph[i] = sorted(graph[i])
    return hash(json.dumps(graph, sort_keys=True))
def move_root(digraph, root_node):
    graph = digraph_to_graph(digraph)
    return graph_to_digraph(graph, root_node)
def copy_graph(graph):
    new_graph = {}
    for node in graph:
        new_graph[node] = copy(graph[node])
    return new_graph
def centers_of_graph(graph):
    graph = copy_graph(graph)
    que = []
    for i in graph:
        if len(graph[i])==1:
            que.append(i)
    while len(graph) > 2:
        node = que.pop(0)
        graph.pop(node)
        for i in graph:
            if node in  graph[i]:
                graph[i].remove(node)
            if len(graph[i])==1:
                que.append(i)
    return list(graph)
def trees(digraph):
    graph = digraph_to_graph(digraph)
    centers = centers_of_graph(graph)
    hs = []
    for c in centers:
        hs.append(hash_graph(graph_to_digraph(graph, c)))
    return hash(json.dumps(sorted(hs), sort_keys=True))