import heapq

# Knapsack Problem
def knapsack(weights, values, capacity):
    n = len(weights)
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            if weights[i - 1] <= w:
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - weights[i - 1]] + values[i - 1])
            else:
                dp[i][w] = dp[i - 1][w]
    return dp[n][capacity]

# Job Sequencing with Deadline
def job_sequencing(jobs):
    jobs = sorted(jobs, key=lambda x: x[2], reverse=True)
    n = len(jobs)
    result = [False] * n
    profit = 0
    slot = [False] * n

    for i in range(n):
        for j in range(min(n, jobs[i][1]) - 1, -1, -1):
            if not slot[j]:
                result[j] = jobs[i][0]
                slot[j] = True
                profit += jobs[i][2]
                break

    return result, profit

# Kruskal's Algorithm
def kruskal(graph):
    def find(parent, i):
        if parent[i] == i:
            return i
        return find(parent, parent[i])

    def union(parent, rank, x, y):
        xroot = find(parent, x)
        yroot = find(parent, y)

        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot
        else:
            parent[yroot] = xroot
            rank[xroot] += 1

    graph = sorted(graph, key=lambda item: item[2])
    parent = [i for i in range(len(graph))]
    rank = [0] * len(graph)
    minimum_spanning_tree = []

    for edge in graph:
        u, v, weight = edge
        x = find(parent, u)
        y = find(parent, v
        if x != y:
            minimum_spanning_tree.append(edge)
            union(parent, rank, x, y)

    return minimum_spanning_tree

# Prim's Algorithm
def prim(graph):
    mst = []
    visited = [False] * len(graph)
    min_edge = [float('inf')] * len(graph)
    min_edge[0] = 0

    for _ in range(len(graph)):
        min_val = min(min_edge)
        u = min_edge.index(min_val)
        visited[u] = True

        for v in range(len(graph)):
            if graph[u][v] != 0 and not visited[v] and graph[u][v] < min_edge[v]:
                min_edge[v] = graph[u][v]
                mst.append((u, v, min_edge[v]))

    return mst

# Dijkstra's Algorithm
def dijkstra(graph, start):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    priority_queue = [(0, start)]

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        if current_distance > distances[current_node]:
            continue

        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))

    return distances

if __name__ == "__main__":
    # Example usage for each algorithm
    weights = [2, 2, 3, 4, 5]
    values = [3, 4, 4, 5, 6]
    capacity = 5
    knapsack_result = knapsack(weights, values, capacity)
    print("Knapsack Result:", knapsack_result)

    jobs = [(1, 2, 100), (2, 1, 19), (3, 2, 27), (4, 1, 25), (5, 3, 15)]
    job_sequence, job_profit = job_sequencing(jobs)
    print("Job Sequence:", job_sequence)
    print("Job Profit:", job_profit)

    graph_kruskal = [(0, 1, 7), (0, 3, 5), (1, 2, 8), (1, 3, 9), (1, 4, 7), (2, 4, 5), (3, 4, 15)]
    kruskal_result = kruskal(graph_kruskal)
    print("Kruskal's Algorithm Result:", kruskal_result)

    graph_prim = {
        0: {1: 7, 3: 5},
        1: {0: 7, 2: 8, 3: 9, 4: 7},
        2: {1: 8, 4: 5},
        3: {0: 5, 1: 9, 4: 15},
        4: {1: 7, 2: 5, 3: 15}
    }
    prim_result = prim(graph_prim)
    print("Prim's Algorithm Result:", prim_result)

    graph_dijkstra = {
        'A': {'B': 1, 'C': 4},
        'B': {'A': 1, 'C': 2, 'D': 5},
        'C': {'A': 4, 'B': 2, 'D': 1},
        'D': {'B': 5, 'C': 1}
    }
    start_node = 'A'
    dijkstra_result = dijkstra(graph_dijkstra, start_node)
    print("Dijkstra's Algorithm Result:", dijkstra_result)
