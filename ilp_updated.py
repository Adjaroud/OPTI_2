import networkx as nx
import pulp
import random
import time
import matplotlib.pyplot as plt
import os

# Configuration du solveur CPLEX
path_to_cplex = r"C:\Program Files\IBM\ILOG\CPLEX_Studio2211\cplex\bin\x64_win64\cplex.exe"
solver = pulp.CPLEX_CMD(path=path_to_cplex, msg=0)


# ==========================================
# FONCTIONS DE RÉSOLUTION
# ==========================================

def analyser_structure(G, k=0):
    ponts = list(nx.bridges(G))
    partitions = {"V01": [], "V2": [], "V3": []}
    for v in G.nodes():
        deg = G.degree(v)
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

    # Objectif (1)
    prob += pulp.lpSum([y[v] for v in partitions['V2']])

    # Contrainte (1) : Somme des x_e = n - 1
    prob += pulp.lpSum([x[e] for e in all_edges]) == G.number_of_nodes() - 1

    #  CONTRAINTE (4) : Base de cycles fondamentaux
    cycles_base = nx.cycle_basis(G)
    for cycle in cycles_base:
        aretes_du_cycle = []
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            aretes_du_cycle.append(tuple(sorted((u, v))))
        prob += pulp.lpSum([x[e] for e in aretes_du_cycle]) <= len(aretes_du_cycle) - 1

    # Contrainte (2) : Ponts et arêtes de reconnexion forcées
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
    n_quota = max(1, int(G_initial.number_of_nodes() ** 0.5 / 2))
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

        composantes = list(nx.connected_components(T))
        print(f"It {iteration} : {len(composantes)} composantes. Quota actuel : {n_quota}")

        candidates = []
        for u, v in G_initial.edges():
            # Trouver si l'arête relie deux composantes différentes
            c_u = next(i for i, c in enumerate(composantes) if u in c)
            c_v = next(i for i, c in enumerate(composantes) if v in c)
            if c_u != c_v:
                candidates.append(tuple(sorted((u, v))))

        # --- MODIFICATION : QUOTA INTELLIGENT ---
        if candidates:
            # Score chaque arête candidate :
            # On additionne les degrés actuels des deux extrémités dans l'arbre partiel T.
            # Plus le score est bas (0 ou 1), moins l'arête risque de créer un branchement (>2).
            def score_arete(e):
                u, v = e
                return T.degree(u) + T.degree(v)

            # On trie les candidates par score croissant (priorité aux feuilles et noeuds isolés)
            candidates.sort(key=score_arete)

            # On prend les 'n_quota' meilleures arêtes (celles de score minimal)
            sample_size = min(len(candidates), n_quota)
            new_edges = candidates[:sample_size]

            edges_forcees.extend(new_edges)
        # ----------------------------------------

        if n_quota > 1:
            n_quota -= 1


# ==========================================
# UTILITAIRES DE FICHIERS
# ==========================================

def charger_graphe_depuis_txt(chemin_fichier):
    G = nx.Graph()
    with open(chemin_fichier, 'r') as f:
        lignes = f.readlines()
        premiere_ligne = lignes[0].split()
        n_sommets = int(premiere_ligne[0])
        G.add_nodes_from(range(1, n_sommets + 1))
        for ligne in lignes[1:]:
            if ligne.strip():
                u, v, w = ligne.split()
                G.add_edge(int(u), int(v))
    return G


def sauvegarder_graphe(T, n_sommets, n_aretes, chemin_sortie):
    with open(chemin_sortie, 'w') as f:
        f.write(f"{n_sommets} {n_aretes} 0\n")
        for u, v in T.edges():
            f.write(f"{u} {v} 0\n")


# ==========================================
# BOUCLE DE GESTION DU DOSSIER
# ==========================================

if __name__ == "__main__":
    dossier_entree = "instances"
    dossier_sortie = "solutions_intelligent"

    if not os.path.exists(dossier_sortie):
        os.makedirs(dossier_sortie)

    for nom_fichier in os.listdir(dossier_entree):
        if nom_fichier.endswith(".txt"):
            print(f"\n--- Traitement de : {nom_fichier} ---")
            chemin_complet = os.path.join(dossier_entree, nom_fichier)

            try:
                G_test = charger_graphe_depuis_txt(chemin_complet)
                n_sommets_init = G_test.number_of_nodes()

                start_time = time.time()
                arbre_final = resoudre_mbvst(G_test)
                end_time = time.time()

                if arbre_final:
                    chemin_sortie = os.path.join(dossier_sortie, "sol_" + nom_fichier)
                    sauvegarder_graphe(arbre_final, n_sommets_init, arbre_final.number_of_edges(), chemin_sortie)

                    nb_branch = sum(1 for v in arbre_final.nodes() if arbre_final.degree(v) > 2)
                    duree = round(end_time - start_time, 2)
                    print(f"   Terminé en {duree}s. Branchements : {nb_branch}")
                else:
                    print(f"   Échec de la résolution pour {nom_fichier}")

            except Exception as e:
                print(f"   Erreur lors du traitement de {nom_fichier} : {e}")

    print("\n" + "=" * 40)
    print("Traitement terminé.")
    print("=" * 40)