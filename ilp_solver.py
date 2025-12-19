import networkx as nx
import pulp
import random
import time
import matplotlib.pyplot as plt

path_to_cplex = r"C:\Program Files\IBM\ILOG\CPLEX_Studio2211\cplex\bin\x64_win64\cplex.exe"
solver = pulp.CPLEX_CMD(path=path_to_cplex, msg=0)

def analyser_structure(G, k=0):
    """Identifie les ponts et segmente les sommets."""
    ponts = list(nx.bridges(G))
    partitions = {"V01": [], "V2": [], "V3": []}
    for v in G.nodes():
        deg = G.degree(v)
        # Simulation simplifiée de la classification fournie
        if deg <= 2:
            partitions["V01"].append(v)
        else:
            partitions["V2"].append(v)
    return ponts, partitions


def resoudre_ilp_iteration(G, partitions, ponts, edges_forcees):
    prob = pulp.LpProblem("MBVST_Iteration", pulp.LpMinimize)
    all_edges = [tuple(sorted(e)) for e in G.edges()]

    # Variables
    x = {e: pulp.LpVariable(f"x_{e[0]}_{e[1]}", cat=pulp.LpBinary) for e in all_edges}
    y = {v: pulp.LpVariable(f"y_{v}", cat=pulp.LpBinary) for v in partitions['V2']}

    # Objectif
    prob += pulp.lpSum([y[v] for v in partitions['V2']])

    # Contrainte (1) : Somme des x_e = n - 1
    prob += pulp.lpSum([x[e] for e in all_edges]) == G.number_of_nodes() - 1

    # Contrainte (4) : Base de cycles fondamentaux
    # On empêche de prendre toutes les arêtes d'un cycle de la base
    cycles_base = nx.cycle_basis(G)
    for cycle in cycles_base:
        aretes_du_cycle = []
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            aretes_du_cycle.append(tuple(sorted((u, v))))

        # Somme x_e = |C| - 1
        prob += pulp.lpSum([x[e] for e in aretes_du_cycle]) <= len(aretes_du_cycle) - 1

    # Contrainte (2) : Ponts et arêtes de reconnexion
    for e in set(ponts + edges_forcees):
        if tuple(sorted(e)) in x:
            prob += x[tuple(sorted(e))] == 1

    # Contraintes (5) & (6) : Degré et Branchement
    for i in G.nodes():
        incidentes = [e for e in all_edges if i in e]
        if i in partitions['V2']:
            prob += pulp.lpSum([x[e] for e in incidentes]) <= 2 + (G.degree(i) - 2) * y[i]
        elif i in partitions['V01']:
            prob += pulp.lpSum([x[e] for e in incidentes]) <= 2
        else:
            prob += pulp.lpSum([x[e] for e in incidentes]) <= G.degree(i)

    # Résolution
    prob.solve(solver)

    if prob.status == 1:
        return [e for e in all_edges if pulp.value(x[e]) == 1]
    return None


def resoudre_mbvst(G_initial):
    print(f"Démarrage : {G_initial.number_of_nodes()} sommets, {G_initial.number_of_edges()} arêtes.")
    ponts, partitions = analyser_structure(G_initial)
    edges_forcees = []

    # Quota initial arbitraire
    n_quota = max(1, int(G_initial.number_of_nodes()**0.5 / 2))
    iteration = 0

    while True:
        iteration += 1
        sol_edges = resoudre_ilp_iteration(G_initial, partitions, ponts, edges_forcees)

        if not sol_edges:
            print("L'ILP est devenu infaisable.")
            return None

        T = nx.Graph()
        T.add_nodes_from(G_initial.nodes())
        T.add_edges_from(sol_edges)

        if nx.is_connected(T):
            print(f"Succès à l'itération {iteration} !")
            return T

        # Gestion de la reconnexion (Quota dégressif)
        composantes = list(nx.connected_components(T))
        print(f"It {iteration} : {len(composantes)} composantes. Quota actuel : {n_quota}")

        candidates = []
        for u, v in G_initial.edges():
            # Trouver si l'arête relie deux composantes différentes
            c_u = next(i for i, c in enumerate(composantes) if u in c)
            c_v = next(i for i, c in enumerate(composantes) if v in c)
            if c_u != c_v:
                candidates.append(tuple(sorted((u, v))))

        if candidates:
            # On prend n_quota arêtes au hasard
            # Ici on utilise un échantillonage pour forcer la reconnexion
            sample_size = min(len(candidates), n_quota)
            new_edges = random.sample(candidates, sample_size)
            edges_forcees.extend(new_edges)

        # Décrémentation du quota pour assurer la terminaison
        if n_quota > 1:
            n_quota -= 1


def charger_graphe_depuis_txt(chemin_fichier):
    """
    Lit un fichier .txt avec le format :
    N_sommets N_arêtes k
    u v poids
    """
    G = nx.Graph()
    with open(chemin_fichier, 'r') as f:
        lignes = f.readlines()

        # Lecture de la première ligne
        premiere_ligne = lignes[0].split()
        n_sommets = int(premiere_ligne[0])

        # Ajout des sommets (de 1 à n_sommets ou 0 à n-1)
        G.add_nodes_from(range(1, n_sommets + 1))

        # Lecture des arêtes
        for ligne in lignes[1:]:
            if ligne.strip():
                u, v, w = ligne.split()
                G.add_edge(int(u), int(v))

    print(f"Graphe chargé : {G.number_of_nodes()} sommets et {G.number_of_edges()} arêtes.")
    return G



def afficher_resultat(G_initial, T_final):
    """
    Affiche le graphe et l'arbre trouvé.
    """
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G_initial)  # Positionnement des sommets

    # Dessiner toutes les arêtes du graphe initial (en gris)
    nx.draw_networkx_edges(G_initial, pos, alpha=0.2, edge_color='grey', style='dashed')

    # Dessiner les arêtes de l'arbre final (en rouge)
    nx.draw_networkx_edges(T_final, pos, width=2, edge_color='red')

    # Identifier les sommets de branchement pour les colorer différemment
    noeuds_branchement = [v for v in T_final.nodes() if T_final.degree(v) > 2]
    noeuds_normaux = [v for v in T_final.nodes() if T_final.degree(v) <= 2]

    nx.draw_networkx_nodes(T_final, pos, nodelist=noeuds_normaux, node_color='lightblue', node_size=500)
    nx.draw_networkx_nodes(T_final, pos, nodelist=noeuds_branchement, node_color='orange', node_size=700)

    nx.draw_networkx_labels(T_final, pos)

    plt.title("Arbre de recouvrement MBVST (Sommets de branchement en orange)")
    plt.axis('off')
    plt.show()


# --- EXEMPLE D'UTILISATION ---
if __name__ == "__main__":
    # 1. Charger votre fichier (remplacez 'mon_graphe.txt' par votre nom de fichier)
    G_test = charger_graphe_depuis_txt('instances/Spd_RF2_250_369_4195.txt')

    # Pour le test immédiat, j'utilise le Karaté Club, mais vous utiliserez la ligne au-dessus
    #G_test = nx.karate_club_graph()

    # 2. Résoudre le problème
    arbre_final = resoudre_mbvst(G_test)

    if arbre_final:
        # 3. Affichage pratique : Liste des arêtes
        print("\n--- ARÊTES DE L'ARBRE FINAL ---")
        print("[")
        for u, v in arbre_final.edges():
            print(f"({u},{v}),")
        print("]")

        # 4. Statistique
        noeuds_branchement = [v for v in arbre_final.nodes() if arbre_final.degree(v) > 2]
        print(f"\nNombre de sommets de branchement : {len(noeuds_branchement)}")
        print(f"Sommets concernés : {noeuds_branchement}")

        # 5. Affichage visuel
        afficher_resultat(G_test, arbre_final)
        print(arbre_final)
        print("Le graphe est connexe :", nx.is_connected(arbre_final))