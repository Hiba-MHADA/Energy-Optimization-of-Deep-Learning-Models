# Energy Optimization of Deep Learning Models
### A Comparative Study of Optimization Algorithms on Fashion-MNIST

Projet de fin de module — *Optimization Algorithms*
Master DSBD & IA — Faculté des Sciences Ben M'Sick, Université Hassan II de Casablanca
Année universitaire : 2025-2026
Encadrante : Mme Faouzia Benabbou

**Équipe**
- Wijdane Bassiry (Master DS&BD)
- Hiba Mhada (Master DS&BD)
- Soukaina Grandi (Master DS&BD)
- Rahil Msouhli (Master DS&BD)
- Israa Sekkat (Master DS&BD)
- Mouad Tace (Master IA)

---

## 1. Description du projet

Ce projet étudie l'impact de différents algorithmes d'optimisation sur la performance prédictive **et** l'efficacité énergétique d'un modèle de Deep Learning, dans une optique de *Green AI* (IA plus sobre et plus durable).

Un CNN léger (`SimpleCNN`) a été entraîné sur le dataset **Fashion-MNIST** avec quatre optimiseurs — **SGD, Adam, AdamW et RMSProp** — évalués dans des conditions identiques. Les indicateurs comparés sont : l'accuracy, la vitesse de convergence, le temps d'entraînement, la mémoire GPU utilisée, les FLOPs et la consommation énergétique estimée (Wh / Joules).

## 2. Contenu de l'archive

| Fichier | Description |
|---|---|
| `optimisation.pdf` | Rapport complet du projet (24 pages) : revue de littérature, méthodologie, architecture du CNN, résultats expérimentaux, analyse énergétique et conclusion générale |
| `projet_optimisation_ipynb (2).ipynb` | Notebook Jupyter/Colab contenant l'ensemble du pipeline (exploration des données, prétraitement, entraînement comparatif, mesure des ressources, visualisations) |
| `Optimisation Énergétique des Modèles Deep Learning.pptx` | Support de présentation orale du projet |

## 3. Architecture du modèle (SimpleCNN)

```
Conv2D(1→32, 3x3) → MaxPool → Conv2D(32→64, 3x3) → MaxPool
→ Flatten → FC(64*7*7→256) → Dropout(0.3) → FC(256→10)
```

Hyperparamètres partagés : 15 époques, learning rate = 1e-3, `CrossEntropyLoss`, batch size = 64.

## 4. Résultats principaux

| Optimiseur | Test Accuracy | Temps total | Énergie (Wh) | Mém. GPU |
|---|---|---|---|---|
| **AdamW** | **92.11 %** | 223–244 s | 4.74 Wh | 101 MB |
| Adam | 92.01 % | 237–246 s | 4.79 Wh | 101 MB |
| RMSProp | 91.38–92 % | 206–243 s | 4.73 Wh | 98 MB |
| SGD | 88.31–88.6 % | 234–236 s | **4.56 Wh** | 98 MB |

**Conclusions clés :**
- 🏆 Meilleure accuracy : **AdamW** (~92.1 %)
- ⚡ Convergence la plus rapide : **RMSProp / Adam** (époque 8–12)
- 🔋 Moins d'énergie consommée : **SGD** (mais accuracy plus faible)
- ⚖️ Meilleure efficacité globale (Acc/Wh) : **RMSProp** (~19.5 Acc%/Wh)

Les optimiseurs adaptatifs (AdamW, Adam, RMSProp) surpassent SGD en précision et en vitesse de convergence. RMSProp offre le meilleur compromis performance/énergie, tandis qu'AdamW maximise la précision.

## 5. Environnement et dépendances

- Python 3
- `torch`, `torchvision`
- `numpy`, `pandas`, `matplotlib`
- *(optionnel)* `thop` — pour le calcul exact des FLOPs
- *(optionnel)* `lion_pytorch` — pour tester l'optimiseur Lion

> Le notebook a été exécuté sur GPU (CUDA) pour les mesures de temps, mémoire et énergie ; il reste exécutable sur CPU mais les métriques énergétiques ne seront pas calculées dans ce cas.

## 6. Utilisation

1. Ouvrir le notebook dans Jupyter ou Google Colab.
2. Exécuter les cellules dans l'ordre (le dataset Fashion-MNIST est téléchargé automatiquement via `torchvision`).
3. Les résultats (CSV, graphiques PNG) sont générés automatiquement : `comparaison_optimiseurs.csv`, `analyse_energetique.csv`, `courbes_loss_accuracy.png`, `m5_radar_chart.png`, etc.
4. Se référer à `optimisation.pdf` pour l'analyse complète et la revue de littérature.
