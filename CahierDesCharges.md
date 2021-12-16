# Conception d’une librairie de réseaux neuronaux

## Samedi 23 Octobre 2021

**Hugo EYNARD
Thomas BLUSSON
Romain MOREAU
Gabriel CHAVANON**

Responsable référent:
Gabriel CHAVANON

Dépot GITEA:
https://dwarves.iut-fbleau.fr/gitiut/blusson/PT21-22-Reseau-Neurones


## Sommaire

- Sommaire
- Cahier des charges fonctionnelles
   - Contexte
   - Etudes détaillées des objectifs (analyses des besoins)
   - Calendrier et priorisation des objectives
- Cahier des charges techniques et méthodologique
   - Bibliographie


## Cahier des charges fonctionnelles


### Contexte

Client : Pierre VALARCHER (tuteur)

Description:

Nous comptons concevoir notre propre librairie de réseaux neuronaux et l’optimiser
par la suite à travers différents tests (reconnaître des caractères manuscrites). Pour
ceci nous ne comptons pas nous appuyer sur des solutions déjà existantes (comme
TensorFlow) mais bien tout réaliser de A à Z. Pour ce qui est de l’optimisation (une
meilleure vitesse d'exécution et une meilleure précision des résultats) on pourra
utiliser des tests et des outils comparatifs (ancienne version de notre projet ou
encore des librairies déjà existantes).

Contraintes :

On réalisera ce projet en Python orienté objet et grâce à ses librairies, tout notre
travail sera disponible surnotre dépôt GITEA.

Existant :

Nous avons à notre disposition un jeu de données contenant des images de chiffres
annotés avec le chiffre correspondant. Ilnous permettra d'entraînernotre réseau
neuronal pour le tester.


### Etudes détaillées des objectifs (analyses des besoins)

Fonctionnalités :
-Choix de la fonction d’activation pour chaque couche de neurones
-Choix du nombre de couches de neurones
-Choix du type de neurones pour chaque couche
-Choix du nombre de neurones dans chaque couche
-Faire une prédiction à partir d’un réseau de neurone
-Entraîner le réseau de neurones
-Exporter et importer l’état (modèle, biais et poids) d’un réseau de neurones
-Visualiser l'entraînement d’un réseau de neurone à deux entrées

Bob le développeur possède un jeu de données comportant des images de chats et
de chiens annotés. Il commence par importer Sobek. Ilchoisit ensuitepour son
modèlederéseauneuronald’avoirunematrice 360 par 360 pourentrée.Ildécidede
mettre 2 premières couches de 180 neurones convolutifs,puis 2 couchesde 64
neuronesdenses(ouclassiques)etenfinunesortiede 2 neuronesdenses(Unpour
chienet un pourchat). Il sépareson jeude donnéesen 2 parties:ilutiliseles 3
premiers quarts pour entraîner son réseau neuronal et après quelques minutes
d'attente, il utilise le dernier quart pour estimer sa précision. Après quelques
modifications et tests de l'architecture de son réseaupour obtenir une meilleure
précision, Bob est satisfait et peut maintenant utiliserson réseaupour faire des
prédictions.


### Calendrier et priorisation des objectives

Jalon 0 : Cahier des charges (à signer) début novembre
Jalon 1 : Perceptron multicouche (première itération du projet), mi décembre M
1.1 : Partie prédiction du perceptron multicouche M
1.2 : Partie apprentissage du perceptron multicouche M
1.3 : Estimation de la précision du perceptron multicouche S
1.4 : Exportation et importation de l’état du réseau neuronal M
Jalon 2 : Utilisation concrète du réseau neuronal, fin décembre M
Jalon 3 : Visualisation 2D de l’entraînement du réseau neuronal M
Jalon 4 : Réseau de neurones convolutif (version optimisée du perceptron multicouche), fin janvier S
4.1 : Création d’un système de choix de type de couche M
4.2 : Implémentation du neurone convolutif M
Jalon 5 : Utilisation concrète du réseau convolutif, fin février C


## Cahier des charges techniques et méthodologique

Nous avons choisi d’utiliser le Python car c’est un langage plutôt abordable qui est
communément utilisé pour le machine learning. On compte également utiliser Scipy
(NumPy (gestion des tableaux), Matplotlib (visualisation de données sous forme
graphique).

Méthodologie de travail :
On envisage d’utiliser la méthode agile (1 sprint correspond à 1 jalon) ainsi que le
pair programming.

On compte utiliser très fréquemment GIT, et faire des tests unitaires dès que
possible. On utilisera make pour l’exécution des tests et des mises en application.


### Bibliographie

https://en.wikipedia.org/wiki/Types_of_artificial_neural_networks

https://youtu.be/bVQUSndDllU

https://youtu.be/aircAruvnKk

https://www.mygreatlearning.com/blog/open-source-python-libraries/

https://fr.wikipedia.org/wiki/Matplotlib

https://brilliant.org/wiki/backpropagation/

_Apprentissage machine Clé de l’intelligence artificielle_ - Rémi Gilleron, 2019

_Deep Learning with Python, 2nd Edition_ - FrançoisChollet, 2021