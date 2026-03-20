import matplotlib.pyplot as plt
import numpy as np

# Catégories
categories = ["RF", "DT", "KNN", "DBSCAN"]  # Valeurs de X
print("Metric calculation")
# Getting Acc
acc_bin = [1.0, 1.0, 1.0, 1.0]
acc_mul = [0.9296, 0.9311, 0.9316, 0.9050]

precision_bin = [1.0, 1.0, 1.0, 1.0]
precision_mul = [0.9333, 0.9322, 0.9319, 0.8662]

recall_bin = [1.0, 1.0, 1.0, 1.0]
recall_mul = [0.9296, 0.9311, 0.9316, 0.8430]

f_1_bin = [1.0, 1.0, 1.0, 1.0]
f_1_mul = [0.9282, 0.9308, 0.9315, 0.8203]

size_bin = np.array([112, 3.4, 683677, 519000])
size_mul = np.array([1300, 442, 683677, 519000])
print("Wait for plotting")
# Plotting
# --------------------------------------------

# Largeur des barres
largeur = 0.4

# Position des barres sur l'axe X
x = np.arange(len(categories))

# Création du graphique
fig, ax = plt.subplots()
# change here to plot the different metrics
barres1 = ax.bar(x - largeur / 2, size_bin, largeur, label="Binaire")
barres2 = ax.bar(x + largeur / 2, size_mul, largeur, label="Multiclasse")

# Personnalisation du graphique
ax.grid()
ax.set_xlabel("Modèles")
ax.set_ylabel("Modèle Size (in Ko)")
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.set_yscale("log")
ax.legend()

# Affichage du graphique
plt.show()
