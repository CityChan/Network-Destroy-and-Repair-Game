import numpy as np
import gym
import networkx as nx
import random as rd
import statistics

CUT = 0
ATTACKER = 0
CONNECT = 1
DEFENDER = 1

def average_degree(G):
    return np.mean(G.degree)

def average_degreesquare(G):
    degree = G.degree
    degree = np.array(degree)
    egreesquare = (degree[:,1])**2
    return np.mean(egreesquare)
def Kappa(G):
    return average_degreesquare(G)/average_degree(G)



class NetworkGameEnv(gym.Env):
    def __init__(self, n = 30, m = 12, budget = [100,100],coff = [100,100]):
        self.Graph0 = nx.barabasi_albert_graph(n,m, seed = 6324)
        self.Graph = self.Graph0.copy()
        self.Attack_budget0 = budget[0]
        self.Attack_budget = self.Attack_budget0
        self.Attack_coff = coff[0]
        self.Defend_budget0 = budget[1]
        self.Defend_budget = self.Defend_budget0
        self.Defend_coff = coff[1]
        self.board = nx.to_numpy_array(self.Graph0)
        self.player = ATTACKER
        self.pass_count = 0
        self.kappa = Kappa(self.Graph0)
        self.k = int(self.board.shape[0]/10)
        self.illegal = 11
        self.penality = 5

    def reset(self):
        self.Graph = self.Graph0.copy()
        self.board = nx.to_numpy_array(self.Graph0)
        self.player = ATTACKER
        self.Attack_budget = self.Attack_budget0
        self.Defend_budget = self.Defend_budget0
        self.pass_count = 0
        budget = [self.Attack_budget, self.Defend_budget]
        return self.board, self.Graph, self.player, budget, self.pass_count

#   action for attacker: 0. pass(do nothing) 1. betweenness first(cut (n/10) edges) 2. randomly pick(cut (n/10) edges)
#   3.remove the edges linked to the node with highest degree(cut (n/10) edges)
#   action for defender: 0. pass(do nothing) 1. betweenness first(connect (n/10) edges) 2. randomly pick(connect (n/10) edges)
#   3.connect the edges linked to the node with highest degree(cut (n/10) edges)

    def is_0_valid(self, state):
        _, _, player, budget, pass_count = state
        if pass_count < 3:
            return True
        else:
            return False

    def is_1_valid(self, state):
        board, Graph, player, budget, _ = state

        if player == ATTACKER:
            BN = nx.edge_betweenness_centrality(Graph)
            BN_sorted = sorted(BN.items(), key=lambda item: item[1], reverse=True)
            if self.k > len(BN_sorted):
                return False
            else:
                cost = 0
                for i in range(self.k):
                    cost = cost + self.Attack_coff*BN_sorted[i][1]
                if int(cost+1) > budget[ATTACKER]:
                    return False
                else:
                    return True

        if player == DEFENDER:
            BN = nx.betweenness_centrality(Graph)
            BN_sorted = sorted(BN.items(), key=lambda item: item[1], reverse=True)
            cost = 0
            count = 0
            Number_node = len(BN_sorted)
            for i in range(Number_node - 1):
                if count >= self.k:
                    break
                for j in range(i+1,Number_node):
                    if count >= self.k:
                        break
                    v1 = BN_sorted[i][0]
                    v2 = BN_sorted[j][0]
                    if board[v1,v2] == CONNECT:
                        continue
                    else:
                        cost = cost + self.Defend_coff*(BN_sorted[i][1] + BN_sorted[j][1])/2
                        count = count + 1
            if int(cost+1) > budget[DEFENDER] or count < self.k:
                return False
            else:
                return True

    def is_2_valid(self, state):
        board, Graph, player, budget, _ = state
        if player == ATTACKER:
            BN = nx.edge_betweenness_centrality(self.Graph).values()
            if self.k > len(BN):
                return False
            else:
                cost = self.k*self.Attack_coff*statistics.mean(BN)
            if int(cost+1) > self.Attack_budget:
                return  False
            else:
                return  True
        if player == DEFENDER:
            BN = nx.edge_betweenness_centrality(self.Graph).values()
            cost = self.k * self.Defend_coff*statistics.mean(BN)
            if int(cost+1) > self.Defend_budget:
                return  False
            else:
                return  True

    def is_2_valid(self, state):
        board, Graph, player, budget, _ = state
        if player == ATTACKER:
            BN = nx.edge_betweenness_centrality(Graph).values()
            if self.k > len(BN):
                return False
            else:
                cost = self.k*statistics.mean(BN)
            if int(cost+1) > budget[ATTACKER]:
                return  False
            else:
                return  True
        if player == DEFENDER:
            BN = nx.edge_betweenness_centrality(Graph).values()
            cost = self.k * statistics.mean(BN)
            if int(cost+1) > budget[DEFENDER]:
                return  False
            else:
                return  True

    def is_3_valid(self, state):
        board, Graph, player, budget, _ = state
        if player == ATTACKER:
            BN = nx.edge_betweenness_centrality(Graph)
            D = Graph.degree
            D_sorted = sorted(D, key=lambda x: x[1], reverse=True)
            if self.k > len(BN):
                return False
            else:
                cost = 0
                edges = list(Graph.edges(D_sorted[0][0]))
                number_edges = len(edges)
                if number_edges>= self.k:
                    for i in range(self.k):
                        remove_v1 = edges[i][1]
                        remove_v2 = edges[i][0]
                        if remove_v1 > remove_v2:
                            temp = remove_v1
                            remove_v1 = remove_v2
                            remove_v2 = temp
                        cost = cost + self.Attack_coff*BN[(remove_v1,remove_v2)]
                    if int(cost+1) > budget[ATTACKER]:
                        return False
                    else:
                        return True
                else:
                    for i in range(number_edges):
                        remove_v1 = edges[i][1]
                        remove_v2 = edges[i][0]
                        if remove_v1 > remove_v2:
                            temp = remove_v1
                            remove_v1 = remove_v2
                            remove_v2 = temp
                        cost = cost + self.Defend_coff * BN[(remove_v1, remove_v2)]
                    if int(cost+1) > budget[DEFENDER]:
                        return False
                    else:
                        return True

        if player == DEFENDER:
            BN = nx.betweenness_centrality(Graph)
            D = Graph.degree
            D_sorted = sorted(D, key=lambda x: x[1], reverse=True)
            count = 0
            cost = 0
            Node = D_sorted[0][0]
            for j in range(board.shape[0]):
                if j == Node:
                    continue
                if board[Node,j] == CONNECT:
                    continue
                remove_v1 = Node
                remove_v2 = j
                if remove_v1 > remove_v2:
                    temp = remove_v1
                    remove_v1 = remove_v2
                    remove_v2 = temp
                cost = cost + self.Defend_coff * (BN[remove_v1]+BN[remove_v2])/2
                count = count + 1
                if count >= self.k:
                    break
            if int(cost+1) > budget[DEFENDER]:
                return False
            else:
                return True

    def is_valid(self, state, action):
        if action == 0:
            return self.is_0_valid(state)
        if action == 1:
            return self.is_1_valid(state)
        if action == 2:
            return self.is_2_valid(state)
        if action == 3:
            return self.is_3_valid(state)
        if action not in [0,1,2,3]:
            return False

    def get_valid(self, state):
        valid = np.zeros(4)
        action = range(4)
        for i in range(4):
            valid[i] = self.is_valid(state,action[i])
        return valid

    def has_valid(self, state):
        action = range(4)
        for i in range(4):
            if self.is_valid(state, action[i]) == True:
                return  True
        return False

    def get_winner(self, state):
        board, Graph, player, budget, pass_count = state
        for player in [ATTACKER, DEFENDER]:
            if self.has_valid(state):
                return None
        New_Kappa = Kappa(Graph)
        if New_Kappa < self.kappa:
            return ATTACKER
        else:
            return DEFENDER

    def get_next_state(self, state, action):
        board, Graph, player, budget, pass_count = state
        if player == ATTACKER:
            if action == 0:
                pass_count = pass_count + 1
                player = -player + 1
                return board, Graph, player, budget, pass_count
            if action == 1:
                pass_count = 0
                BN = nx.edge_betweenness_centrality(Graph)
                BN_sorted = sorted(BN.items(), key=lambda item: item[1], reverse=True)
                cost = 0
                for i in range(self.k):
                    remove = BN_sorted[i][0]
                    v1 = remove[0]
                    v2 = remove[1]
                    Graph.remove_edge(v1,v2)
                    board[v1,v2] = CUT
                    board[v2,v1] = CUT
                    cost = cost + self.Attack_coff * BN_sorted[i][1]
                budget[ATTACKER] = budget[ATTACKER] - int(cost+1)

            if action == 2:
                pass_count = 0
                BN = nx.edge_betweenness_centrality(Graph)
                BN_sorted = sorted(BN.items(), key=lambda item: item[1], reverse=True)
                choices = rd.sample(BN_sorted, self.k)
                cost = 0
                for i in range(self.k):
                    remove = choices[i][0]
                    v1 = remove[0]
                    v2 = remove[1]
                    if v1 > v2:
                        temp = 1
                        v1 = v2
                        v2 = temp
                    Graph.remove_edge(v1,v2)
                    board[v1,v2] = CUT
                    board[v2,v1] = CUT
                    cost = cost + self.Attack_coff*BN[(v1,v2)]
                budget[ATTACKER] = budget[ATTACKER] - int(cost+1)
            if action == 3:
                pass_count = 0
                BN = nx.edge_betweenness_centrality(Graph)
                D = Graph.degree
                D_sorted = sorted(D, key=lambda x: x[1], reverse=True)
                cost = 0
                edges = list(Graph.edges(D_sorted[0][0]))
                number_edges = len(edges)
                if number_edges >= self.k:
                   for i in range(self.k):
                      remove_v1 = edges[i][1]
                      remove_v2 = edges[i][0]
                      if remove_v1 > remove_v2:
                         temp = remove_v1
                         remove_v1 = remove_v2
                         remove_v2 = temp
                      Graph.remove_edge(remove_v1,remove_v2)
                      board[remove_v1,remove_v2] = CUT
                      board[remove_v2,remove_v1]= CUT
                      cost = cost + self.Attack_coff * BN[(remove_v1, remove_v2)]
                   budget[ATTACKER] = budget[ATTACKER] - int(cost+1)
                else:
                   for i in range(number_edges):
                      remove_v1 = edges[i][1]
                      remove_v2 = edges[i][0]
                      if remove_v1 > remove_v2:
                         temp = remove_v1
                         remove_v1 = remove_v2
                         remove_v2 = temp
                      Graph.remove_edge(remove_v1, remove_v2)
                      board[remove_v1,remove_v2] = CUT
                      board[remove_v2,remove_v1] = CUT
                      cost = cost + self.Defend_coff * BN[(remove_v1, remove_v2)]
                   budget[ATTACKER] = budget[ATTACKER] - int(cost+1)
            if action == self.illegal:
                budget[ATTACKER] = budget[ATTACKER] - self.penality
            player = -player + 1
            return board, Graph, player, budget, pass_count

        if player == DEFENDER:
            if action == 0:
                pass_count = pass_count + 1
                player = -player + 1
                return board, Graph, player, budget, pass_count
            if action == 1:
                pass_count = 0
                BN = nx.betweenness_centrality(Graph)
                BN_sorted = sorted(BN.items(), key=lambda item: item[1], reverse=True)
                Number_node = len(BN_sorted)
                cost = 0
                count = 0
                for i in range(Number_node - 1):
                    if count >= self.k:
                        break
                    for j in range(i + 1, Number_node):
                        if count >= self.k:
                            break
                        v1 = BN_sorted[i][0]
                        v2 = BN_sorted[j][0]
                        if board[v1, v2] == CONNECT:
                            continue
                        else:
                            cost = cost + self.Defend_coff * (BN_sorted[i][1] + BN_sorted[j][1]) / 2
                            count = count + 1
                            Graph.add_edge(v1,v2)
                            board[v1,v2] = CONNECT
                            board[v2,v1] = CONNECT
                    budget[DEFENDER] = budget[DEFENDER] - int(cost+1)
            if action == 2:
                pass_count = 0
                BN = nx.betweenness_centrality(Graph)
                BN_sorted = sorted(BN.items(), key=lambda item: item[1], reverse=True)
                count = 0
                cost = 0
                while count < self.k:
                    choices = rd.sample(BN_sorted,2)
                    v1 = choices[0][0]
                    v2 = choices[1][0]
                    if board[v1,v2] == CONNECT:
                        continue
                    Graph.add_edge(v1, v2)
                    board[v1,v2] = CONNECT
                    board[v2,v1] = CONNECT
                    count = count + 1
                    cost =  cost + self.Defend_coff*(BN[v1]+BN[v2])/2
                budget[DEFENDER] = budget[DEFENDER] - int(cost+1)
            if action == 3:
                pass_count = 0
                BN = nx.betweenness_centrality(Graph)
                D = Graph.degree
                D_sorted = sorted(D, key=lambda x: x[1], reverse=True)
                count = 0
                cost = 0
                Node = D_sorted[0][0]
                for j in range(board.shape[0]):
                    if j == Node:
                        continue
                    if board[Node,j] == CONNECT:
                        continue
                    add_v1 = int(Node)
                    add_v2 = int(j)
                    if add_v1 > add_v2:
                        temp = add_v1
                        add_v1 = add_v2
                        add_v2 = temp
                    Graph.add_edge(add_v1,add_v2)
                    board[add_v1,add_v2] = CONNECT
                    board[add_v2,add_v1] = CONNECT
                    cost = cost + self.Defend_coff * (BN[add_v1]+BN[add_v2])/2
                    count = count + 1
                    if count >= self.k:
                        break
                budget[DEFENDER] = budget[DEFENDER] - int(cost+1)
            if action == self.illegal:
                budget[DEFENDER] = budget[DEFENDER] - self.penality
            player = -player + 1
            return board, Graph, player, budget, pass_count

    def next_step(self, state, action):
        board, Graph, player, budget, pass_count = state
        if not self.is_valid(state, action):
            action = self.illegal
            state = (board, Graph, player, budget, pass_count)
        while True:
            state = self.get_next_state(state, action)
            winner = self.get_winner(state)
            if winner is not None:
                return state, winner, True, {}
            if self.has_valid(state):
                break
        return state, -1, False, {}

    def step(self, action):
        Graph = self.Graph.copy()
        budget = [self.Attack_budget, self.Defend_budget]
        state = (self.board, Graph, self.player, budget, self.pass_count)
        next_state, winner, done, info = self.next_step(state, action)
        self.board, self.Graph, self.player, budget, self.pass_count = next_state
        self.Attack_budget = budget[0]
        self.Defend_budget = budget[1]
        return next_state, winner, done, info



# env = NetworkGameEnv()
# state = (env.board, env.Graph, env.player, [env.Attack_budget, env.Defend_budget], env.pass_count )
# state,_,_,_ = env.next_step(state,2)
# board, Graph, player, budget, pass_count = state
# vaild = env.is_valid(state,11)
