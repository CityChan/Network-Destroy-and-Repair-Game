import networkx as nx
import numpy as np
import collections
import matplotlib.pyplot as plt
from pylab import show

ATTACKER = 0
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

class Attacker:
    remove_max = 200
    Budget_attack = 0
    Budget_oneturn = 0
    Budget_exhaust = False
    Budget_exhaust_thisturn = False
    def __init__(self, m, b, o):
        self.remove_max = m
        self.Budget_attack = b
        self.Budget_oneturn = o
    def Set_remove_max(self,m):
        self.remove_max = m
    def Set_Budget_oneturn(self,o):
        self.Budget_fraction = o
    def Set_Budget_attack(self,b):
        self.Budget_attack = b
    def Set_flag(self, Bool):
        self.Budget_exhaust = Bool
class Defender:
    add_max = 200
    Budget_defend = 0
    Budget_oneturn = 0
    Budget_exhaust = False
    Budget_exhaust_thisturn = False
    def __init__(self, m, b, o):
        self.add_max = m
        self.Budget_defend = b
        self.Budget_oneturn = o
    def Set_add_max(self,m):
        self.add_max = m
    def Set_Budget_oneturn(self,o):
        self.Budget_fraction = o
    def Set_Budget_defend(self,b):
        self.Budget_defend = b
    def Set_flag(self, Bool):
        self.Budget_exhaust = Bool

class Game:
    coff_1 = 100
    coff_2 = 500
    Kappa = 0   # molloy-reed criterion
    Graph = []  # this network
    Fraction = 0.7
    Game_log = {}
    attack = []
    defender = []
    def __init__(self, n,m,seed):
        self.Graph = nx.barabasi_albert_graph(n,m,seed=seed)
        self.Kappa = Kappa(self.Graph)
        self.add_max = m
    def Set_coff_1(self, c1):
        self.coff_1 = c1
    def Set_coff_2(self, c2):
        self.coff_2 = c2
    def Set_fraction(self,f):
        self.Fraction = f
    def Initialization(self):
        print("initializing...")
        Graph2 = self.Graph.copy()
        BN = nx.edge_betweenness_centrality(Graph2)
        BN_sorted = sorted(BN.items(), key=lambda item: item[1], reverse=True)
        cost = 0
        Kappa_0 = Kappa(Graph2)
        while Kappa(Graph2) / Kappa_0 > self.Fraction:
            remove_edge = BN_sorted[0][0]
            cost = cost + BN_sorted[0][1]
            BN_sorted.remove(BN_sorted[0])
            a = remove_edge[0]
            b = remove_edge[1]
            Graph2.remove_edge(a,b)
        Budget_attack = self.coff_1*cost*4
        Budget_defend = Budget_attack
        print("Attacker's budget is: ", Budget_attack)
        print("Defender's budget is: ", Budget_defend)
        self.attacker = Attacker(10,Budget_attack,0.05*Budget_attack)
        self.defender = Defender(10,Budget_defend,0.05*Budget_defend)
    def Attacker_win(self):
        if Kappa(self.Graph)/self.Kappa < self.Fraction:
            print("Kappa: ", Kappa(self.Graph)/self.Kappa)
            return True
        else:
            print("Kappa: ", Kappa(self.Graph) / self.Kappa)
            return False
    def Defender_win(self, exhaust):
        if Kappa(self.Graph)/self.Kappa >= self.Fraction and self.attacker.Budget_exhaust == True:
            print("Kappa: ", Kappa(self.Graph) / self.Kappa)
            return True
        else:
            return False
    def Play_game(self):
        print("Let's play a game.")
        attack_turn = True
        turn = 1
        while not self.Attacker_win() or self.Defender_win():
            if attack_turn == True:
                print("Turn: ", turn)
                self.attacker.Budget_exhaust_thisturn = False
                Budget_oneturn = self.attacker.Budget_oneturn
                print("This is attacker turn.")
                print("Your budget", self.attacker.Budget_attack)
                print("You can remove max edges: ", self.attacker.remove_max)
                print("You can use max budget in one turn: ",Budget_oneturn)
                print("Game log is as follow:")
                print(self.Game_log)
                BN = nx.edge_betweenness_centrality(self.Graph)
                BN_sorted = sorted(BN.items(), key=lambda item: item[1], reverse=True)
                remove_all = []
                for i in range(self.attacker.remove_max):
                    remove = BN_sorted[0]
                    remove_edge = remove[0]
                    remove_cost = self.coff_1*remove[1]
                    while self.attacker.Budget_attack < remove_cost or Budget_oneturn < remove_cost:
                        if len(BN_sorted) == 1:
                            if self.attacker.Budget_attack < remove_cost:
                                print("Attacker's budget is exhausted. You can't afford a edge to remove.")
                                self.attacker.Budget_exhaust = True
                            else:
                                print("Attacker's budget is exhausted in this turn. You can't afford a edge to remove in this turn.")
                                self.attacker.Budget_exhaust_thisturn = True
                            break
                        BN_sorted.remove(remove)
                        remove = BN_sorted[0]
                        remove_edge = remove[0]
                        remove_cost = self.coff_1 * remove[1]
                    if self.attacker.Budget_exhaust == True or self.attacker.Budget_exhaust_thisturn == True:
                        break
                    v1 = remove_edge[0]
                    v2 = remove_edge[1]
                    remove_all.append(remove_edge)
                    self.Graph.remove_edge(v1,v2)
                    Budget_oneturn = Budget_oneturn - remove_cost
                    self.attacker.Budget_attack = self.attacker.Budget_attack - remove_cost
                    BN = nx.edge_betweenness_centrality(self.Graph)
                    BN_sorted = sorted(BN.items(), key=lambda item: item[1], reverse=True)
                if self.Attacker_win():
                    break
                if self.attacker.Budget_exhaust == True:
                    break
                print("The attacker action in this turn: ", remove_all)
                print("Kappa is: ", Kappa(self.Graph)/self.Kappa)
                self.Game_log.update({-turn: remove_all})
                attack_turn = False
                turn = turn + 1

            else:
                print("Turn: ", turn)
                self.defender.Budget_exhaust_thisturn = False
                Budget_oneturn = self.defender.Budget_oneturn
                print("This is defender turn.")
                print("Your budget", self.defender.Budget_defend)
                print("You can add max edges: ", self.defender.add_max)
                print("You can use max budget in one turn: ", Budget_oneturn)
                print("Game log is as follow:")
                print(self.Game_log)
                BN = nx.betweenness_centrality(self.Graph)
                BN_sorted = sorted(BN.items(), key=lambda item: item[1], reverse=True)
                add_all = []
                if self.defender.Budget_exhaust == True:
                    print("Nothing you can do. Next turn")
                    self.Game_log.update({turn: add_all})
                    attack_turn = True
                    turn = turn + 1
                    continue
                Number_node = len(BN_sorted)
                count = 0
                for i in range(Number_node-1):
                    if count == self.defender.add_max:
                        break
                    for j in range(i+1,Number_node):
                        if count == self.defender.add_max:
                            break
                        v1 = BN_sorted[i][0]
                        v2 = BN_sorted[j][0]
                        if self.Graph.has_edge(v1,v2) == False:
                            add_cost = self.coff_2*(BN_sorted[i][1]+BN_sorted[j][1])*0.5
                            if remove_cost > Budget_oneturn or remove_cost > self.defender.Budget_defend:
                                continue
                            else:
                                self.Graph.add_edge(v1,v2)
                                Budget_oneturn = Budget_oneturn - add_cost
                                self.defender.Budget_defend = self.defender.Budget_defend - add_cost
                                add_all.append((v1,v2))
                                count = count + 1
                        else:
                            continue

                if add_cost > self.defender.Budget_defend:
                    self.defender.Budget_exhaust = True
                    print("The defender's budget is exhausted.")
                    print("Kappa is: ", Kappa(self.Graph) / self.Kappa)
                    self.Game_log.update({turn: add_all})
                    attack_turn = True
                    turn = turn + 1
                    continue
                if add_cost > Budget_oneturn:
                    self.defender.Budget_exhaust_thisturn = True
                    print("The defender's budget in this turn is exhausted.")
                    print("Kappa is: ", Kappa(self.Graph) / self.Kappa)
                    self.Game_log.update({turn: add_all})
                    attack_turn = True
                    turn = turn + 1
                    continue
                print("The defender action in this turn: ", add_all)
                print("Kappa is: ", Kappa(self.Graph) / self.Kappa)
                self.Game_log.update({turn:add_all})
                attack_turn = True
                turn = turn + 1

        if self.Attacker_win():
            print("Attacker wins.")
        else:
            print("Defender wins.")
G = Game(50,18,6324)
G.Initialization()
G.Play_game()


