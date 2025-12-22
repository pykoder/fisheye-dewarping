# Fisheye Dewarping : Du Memory Leak à 6 Implémentations Comparées

*TL;DR : Face à un bug critique en prod, j'ai dû recoder un dewarper fisheye from scratch. Trois ans plus tard, je revisite ce problème pour comparer 6 approches différentes - FFmpeg ligne de commande, Python pur, NumPy vectorisé, OpenCV Python/C++, et une lib C++ custom optimisée.*

---

## L'incident Fisheye

**Janvier 2022.** Notre conteneur Docker plante en prod après quelques heures sur certains sites. Le diagnostic tombe : memory leak dans la bibliothèque propriétaire qui gère le dewarping des caméras fisheye 360°.

Contexte rapide : notre IA analyse des flux vidéo pour détecter des gestes. Nos modèles sont entraînés sur des vues plates, pas sur des vues circulaires déformées. Sans dewarping, **zéro détection** sur les caméras fisheye, juste 1% du parc, mais des clients importants.

La lib qui plante ? Fournie par le fabricant des caméras. On ne peut ni la patcher, ni attendre un correctif.

**Verdict** : on code notre propre version.

Quinze jours de plongée dans les bouquins de géométrie projective, une implémentation C++ à base de quaternions et projections sphériques, quelques semaines d'optimisation... et on avait notre solution en prod.

**Trois ans plus tard**, avec un peu de recul et du temps libre, une question me trotte dans la tête : **et si on l'avait fait autrement ?**

Quelle aurait été la meilleure approche avec nos contraintes ?
- Caméras fisheye **fixes** au plafond
- Besoin de **5 vues plates** par frame
- Points de vue des caméras virtuelles **statiques** (pas de rotation dynamique)
- **Performance critique** : traitement temps réel de multiples flux vidéo
- Qualité "suffisante" pour la détection (pas besoin de perfection photographique)

Cet article compare **6 implémentations différentes** pour ce cas d'usage précis, avec benchmarks à l'appui.

---

## Ce que vous allez découvrir

1. **Les bases théoriques** du dewarping fisheye (version courte, promis)
2. **Trois implémentations** du même algorithme :
   - FFmpeg en ligne de commande
   - Python natif (boucles, pas de lib)
   - NumPy vectorisé
3. **Benchmarks comparatifs** : temps d'exécution, RAM, complexité de mise en œuvre
4. Une mise en bouche pour la **partie 2** au cours de laquelle nous explorons OpenCV et une implémentation C++ ad hoc.

**Important** : Cette comparaison est spécifique à *notre* cas d'usage (caméras fixes, vues statiques, perf temps réel). Si vos besoins différent - caméras mobiles, recalibration dynamique, qualité maximale, etc. - adaptez en conséquence.

---

## Un peu de théorie (juste ce qu'il faut)

### Le problème en image

Une caméra fisheye 360° au plafond capture tout l'espace environant dans une image circulaire déformée.

Pour que nos algorithmes de détection puissent travailler, nous en tirons plusieurs vues plates rectangulaires :

![Schéma de principe](https://github.com/pykoder/fisheye-dewarping/blob/main/images/schema.png?raw=true)


### Comment ça marche (version simplifiée)

Le dewarping se décompose en deux phases :

**Phase 1 : Calcul du mapping (une seule fois au démarrage)**

On crée une table de correspondance : pour chaque pixel de nos 5 vues plates de sortie, on calcule quel pixel de l'image fisheye il faut aller chercher.

Concrètement :
1. On projette chaque point de l'image fisheye sur une demi-sphère virtuelle centrée sur la caméra
2. On définit 5 "caméras virtuelles" avec leurs positions et orientations fixes
3. Pour chaque caméra virtuelle, on calcule quelle portion de la sphère elle "voit"
4. On stocke tout ça dans une lookup table

Cette phase utilise des projections sphériques et dans notre cas initial, des quaternions pour les rotations. C'est du calcul assez lourd, mais on ne le fait **qu'une seule fois**.

**Phase 2 : Application du mapping (pour chaque frame vidéo)**

Pour chaque nouvelle image de la caméra fisheye :
1. On parcourt nos 5 vues de sortie
2. Pour chaque pixel, on regarde dans la lookup table d'où il vient
3. On copie la couleur du pixel source (avec interpolation optionnelle si on veut de la qualité)

C'est cette phase qu'on doit optimiser à fond. Elle tourne en boucle sur chaque frame.

### Note sur le calibrage

Les lentilles fisheye varient d'un modèle de caméra à l'autre. Un calibrage précis permet d'obtenir des vues parfaitement rectilignes. Dans notre cas, nous nous en passons : nos algos de détection tolèrent de légères déformations résiduelles. Cela simplifie le code et booste les performances.

---

## Implémentation 1: FFmpeg CLI - La solution rapide et parallèle

### L'approche

FFmpeg supporte nativement ce type de dewarping via son filtre `v360`. L'implémentation est directe mais révèle quelques subtilités intéressantes.

### Le code
```bash
#!/bin/bash
# unwarper_ffmpeg.sh

ffmpeg -y -i "fisheye.mp4" \
-vf "crop=1920:1920,v360=input=fisheye:output=flat:interp=near:yaw=0:pitch=45:roll=0:v_fov=90:w=960:h=960" "unwarped_1.mp4" \
-vf "rotate=4*72*PI/180,crop=1920:1920,v360=input=fisheye:output=flat:interp=near:yaw=0:pitch=45:roll=0:v_fov=90:w=960:h=960" "unwarped_2.mp4" \
-vf "rotate=3*72*PI/180,crop=1920:1920,v360=input=fisheye:output=flat:interp=near:yaw=0:pitch=45:roll=0:v_fov=90:w=960:h=960" "unwarped_3.mp4" \
-vf "rotate=2*72*PI/180,crop=1920:1920,v360=input=fisheye:output=flat:interp=near:yaw=0:pitch=45:roll=0:v_fov=90:w=960:h=960" "unwarped_4.mp4" \
-vf "rotate=72*PI/180,crop=1920:1920,v360=input=fisheye:output=flat:interp=near:yaw=0:pitch=45:roll=0:v_fov=90:w=960:h=960" "unwarped_5.mp4"
```

### Détails d'implémentation

**Paramètres du filtre v360 :**
- `yaw`, `pitch`, `roll` : rotations décrivant l'orientation de la caméra fisheye (dome au plafond)
- `v_fov=90` : champ de vision vertical de la vue de sortie
- `w=960:h=960` : résolution des vues dewarpées
- `interp=near` : interpolation plus proche voisin (vs `linear` par défaut)

**Choix d'optimisation :**

1. **Interpolation minimale** : On force `interp=near` au lieu de l'interpolation linéaire par défaut. La qualité d'image est légèrement dégradée mais les performances sont meilleures. Pour de la détection d'objets, c'est largement suffisant.

2. **Rotation de l'image source** : Le contrôle du point de vue via `yaw/pitch/roll` est limité et ne permet d'obtenir le résultat souhaité que dans une seule direction. Nous contournons le problème en appliquant une rotation préalable de l'image fisheye (multiples de 72° pour couvrir les 360°).

3. **Lecture unique du fichier source** : Toutes les vues sont générées en un seul passage de FFmpeg (une commande avec 5 outputs). Essentiel pour les perfs.

4. **Vidéo de 1024 frames** : Les benchmarks utilisent une vidéo suffisamment longue pour que le temps d'application du mapping domine largement le temps de calcul initial du mapping (la phase qui nous intéresse vraiment). 1024 répétitions du dewarping sera la référence pour les autres solution.

### Benchmark
```
Commande: ./unwarper_ffmpeg.sh fisheye_video.mp4

======================================================================
RESULTATS BENCHMARK
======================================================================
Wall time:              208.64s
CPU time (user+sys):     1411.84s
  - User time:           1401.84s
  - System time:         10.00s
CPU utilization:        676%
Cores used:             ~6.8
Peak memory:            1784.20 MB (1827016 KB)
Page faults:            387273 minor, 0 major
Context switches:       328327 vol, 712451 invol
Exit status:            0

Parallel speedup:       6.77x
(CPU time / Wall time = 1411.84s / 208.64s)
======================================================================
```

**Résultats :** 208 secondes pour traiter 1024 frames et générer 5 vues, soit 5120 images plates extraites, soit environ 40ms par image plate. 

FFmpeg exploite bien les 8 cores disponibles (~676% d'utilisation CPU), avec un speedup parallèle de 6.8x. L'utilisation mémoire grimpe toutefois à **1.7 GB**, ce qui est significatif. Cette consommation mémoire ne dépend pas de la longueur du film.

FFmpeg est aussi un peu pénalisé vis à vis des autres solutions car il procède aussi à un réencodage des vues en mp4 sous forme de film. Ce réencodage n'est en réalité pas nécessaire dans le cas d'usage présenté.


### Analyse

**✅ Points forts**
- **Setup immédiat** : une seule commande, aucune lib à installer (FFmpeg suffit)
- **Parallélisation native** : utilisation optimale du multi-core sans effort
- **Parallèlization excellente** : accélération de 6,77×
- **Robuste** : FFmpeg est battle-tested en prod partout
- **Pas de maintenance** : dépendance externe stable, bugs déjà corrigés par la communauté

**❌ Points faibles**
- **Boîte noire totale** : impossible d'auditer ou modifier l'algo de dewarping
- **Flexibilité limitée** : on est coincé avec les paramètres exposés par `v360`
- **Pas intégrable finement** : nécessite de spawner un process externe, impossible d'appeler directement comme une fonction Python
- **Consommation mémoire élevée** : 1.7 GB pour une vidéo 1920×1920, potentiellement problématique à grande échelle
- **Optimisations limitées** : on ne peut pas optimiser la phase de mapping spécifiquement pour notre cas d'usage (caméras fixes, vues statiques)

### Verdict

FFmpeg est **l'arme idéale pour un POC rapide** ou quand vous avez besoin d'un résultat qui marche immédiatement sans vous poser de questions. Parfait pour :
- Tester si le dewarping résout le problème métier
- Scripts one-shot ou batch processing occasionnel
- Situations où la RAM n'est pas une contrainte

**Mais inadapté si :**
- Vous devez intégrer le dewarping dans un pipeline Python complexe
- Vous voulez optimiser finement (pré-calcul du mapping une fois, réutilisation)
- La consommation mémoire est critique
- Vous avez besoin de comprendre ou adapter l'algorithme sous-jacent

Dans notre cas (memory leak de la lib propriétaire), FFmpeg aurait pu être une solution de secours acceptable... mais nous aurions vite été limités pour l'optimisation et l'intégration.


---
## ## Implementation 2: Python pur - Objectif Comprendre les maths

### L'approche

Maintenant qu'on a vu la solution "boîte noire" avec FFmpeg, plongeons dans les entrailles de l'algorithme. Cette implémentation en **Python pur** utilise uniquement les bibliothèques standard et NumPy pour la manipulation de tableaux, mais **sans aucune vectorisation** (ok, j'avoue j'ai laissé un produit matriciel pour ne pas le recoder à la main).

L'objectif ici n'est pas la performance, mais la **compréhension**. Chaque étape mathématique est explicite, documentée, compréhensible. C'est la référence pédagogique qui servira de baseline pour toutes les optimisations ultérieures.

### Le code
```python
#!/usr/bin/env python3
"""
Pure Python Fisheye Unwarper

Cette implémentation en Python pur du dewarping fisheye utilise uniquement
des bibliothèques standard et numpy pour manipuler des tableaux,
sans aucune optimisation vectorielle.
"""

def multiply_quaternion(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Multiply two quaternions using Python primitives.
    
    Args:
        a: First quaternion [w, x, y, z]
        b: Second quaternion [w, x, y, z]
        
    Returns:
        Result quaternion [w, x, y, z]
    """
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return np.array([w, x, y, z], dtype=np.float64)

def get_rotation_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """
    Generate rotation matrix from yaw, pitch, roll angles (in degrees).
    
    Args:
        yaw: Rotation around Y axis in degrees
        pitch: Rotation around X axis in degrees
        roll: Rotation around Z axis in degrees
    Returns:
        3x3 rotation matrix as numpy array
    """
    # Yaw quaternion, rotate view around Y axis
    yaw = np.deg2rad(0)
    yaw_q = np.array([np.cos(yaw/2.0), 0.0, np.sin(yaw/2.0), 0.0], dtype=np.float64)
    # Pitch quaternion, rotate view around X axis (look up 45 degrees)
    pitch = np.deg2rad(45)             
    pitch_q = np.array([np.cos(pitch/2.0), np.sin(pitch/2.0), 0.0, 0.0], dtype=np.float64)
    # Roll quaternion, rotate view around Z axis (look in different direction)
    roll = np.deg2rad(roll)
    roll_q = np.array([np.cos(roll/2.0), 0.0, 0.0, np.sin(roll/2.0)], dtype=np.float64)

    rq = multiply_quaternion(roll_q, multiply_quaternion(pitch_q, yaw_q))

    # Build spherical projection matrix from quaternions
    w, x, y, z = rq
    return np.array([
        [ (w*w + x*x - y*y - z*z),  2.0 * (x*y - z*w), 2.0 * (w*y + x*z)],
        [ 2.0 * (w*z + x*y), (w*w - x*x + y*y - z*z), 2.0 * (y*z - w*x)],
        [ 2.0 * (x*z - y*w), 2.0 * (w*x + y*z), (w*w - x*x - y*y + z*z)]], dtype=np.float64)


def project2D(xyz: np.ndarray) -> Tuple[int, int]:
    """Project 3D Dome point to 2D Fisheye image"""
    hs = np.hypot(xyz[0],xyz[1])
    phi = np.arctan2(hs, xyz[2])
    coeff = phi / (hs * np.pi)
    src_x = xyz[0] * coeff + 0.5
    src_y = xyz[1] * coeff + 0.5
    return src_x, src_y


class PythonDewarper:
    """
    Pure Python implementation of the fisheye dewarper.    
    """
    
    def __init__(self, width: int, height: int, zones: int = 3):
        """
        Initialize dewarper with image dimensions.
        
        Args:
            width: Image width in pixels
            height: Image height in pixels
        """
        self.width = width
        self.height = height
        self.output_width = self.width // 2
        self.output_height = self.height // 2
        self.zones = zones
        
        # Remapping tables for each view
        self.remap = self._dewarp_mapping()
        self.output_buffer = np.zeros((self.zones, self.output_height, self.output_width, 3), dtype=np.uint8)
    

    def _dewarp_mapping(self) -> List[List[List[Tuple[int, int]]]]:
        """
        Create pixel remapping table for specific view.
        
        This is the core dewarping algorithm using spherical projection.
        """
       
        remap = []
        for zone_id in range(self.zones):
    
            # Get rotation matrix for this zone
            R = get_rotation_matrix(0, 45, zone_id * (360.0 / self.zones))

            remap_zone = []
            for j in range(self.output_height):
                line = []
                for i in range(self.output_width):
                    v = np.array([i / (0.25 * self.width) - 1.0, j / (0.25 * self.height) - 1.0, 1.0])
                    xyz = R @ v.T
                    src_x, src_y = project2D(xyz)
                    map_y = int(src_y * self.height)
                    map_x = int(src_x * self.width)
                    if 0 <= map_y < self.height and 0 <= map_x < self.width:    
                        line.append((map_y, map_x))
                    else:
                        line.append((0, 0))
                remap_zone.append(line)
            remap.append(remap_zone)

        return remap
    
    def dewarp_frame(self, image: np.ndarray, zone_id: int = -1):
        """
        Apply dewarping transformation to image.
        
        Args:
            image: Input image as NumPy array (H, W, 3)
        """
        
        remap_table = self.remap[zone_id]
        output_buffer = self.output_buffer[zone_id]

        for i in range(self.output_height):
            for j in range(self.output_width):
                # Note: never out of bound as it is ensured when building remapping
                output_buffer[i, j] = image[remap_table[i][j]]

        return output_buffer.reshape((self.output_height, self.output_width, 3))
```

### Explication détaillée des maths

Maintenant qu'on a le code, décortiquons ce qui se passe sous le capot. C'est ici qu'on va plonger dans les détails mathématiques - on n'y reviendra plus dans les sections suivantes.

#### Pourquoi les quaternions ?

**Petit rappel historique** : Les quaternions ont été inventés en 1843 par William Hamilton pour représenter les rotations dans l'espace 3D. Ils ont depuis conquis la 3D, la robotique, l'aérospatiale et... le dewarping fisheye.

**Pourquoi pas des angles d'Euler classiques ?**

Les angles d'Euler (yaw/pitch/roll) sont intuitifs mais ont plusieurs défauts :
- **Gimbal lock** : certaines combinaisons de rotations causent une perte de degré de liberté
- **Interpolation non-linéaire** : difficile d'interpoler proprement entre deux orientations
- **Calculs plus lourds** : composer des rotations nécessite 3 multiplications matricielles 3×3

Les quaternions règlent tout ça :
- **Pas de gimbal lock** : toutes les orientations sont représentables sans singularité
- **Composition efficace** : multiplier deux quaternions = 16 multiplications + 12 additions (vs 27 mult + 18 add pour des matrices 3×3)
- **Compacts** : 4 nombres au lieu de 9 (matrice 3×3)
- **Normalisables facilement** : évite l'accumulation d'erreurs numériques lors de compositions successives

**Structure d'un quaternion** : `[w, x, y, z]` où :
- `w` : partie scalaire (cosinus de la demi-rotation)
- `(x, y, z)` : partie vectorielle (direction de l'axe de rotation)

Une rotation de θ autour d'un axe unitaire `(ax, ay, az)` s'écrit :
```
q = [cos(θ/2), ax·sin(θ/2), ay·sin(θ/2), az·sin(θ/2)]
```

#### Les trois rotations

Notre caméra virtuelle doit pouvoir regarder dans n'importe quelle direction. On décompose ça en 3 rotations.

**Convention importante** : Pour une caméra fisheye montée **au plafond** (regardant vers le bas), les axes sont différents de la convention classique des caméras horizontales :

**1. Yaw (rotation autour de Y)** : rotation de la caméra sur son axe optique
```python
yaw_q = [cos(yaw/2), 0, sin(yaw/2), 0]
```
Dans notre cas : `yaw=0°` (pas de rotation sur l'axe)

**2. Pitch (rotation autour de X)** : inclinaison du regard (regarder plus ou moins vers le bas)
```python
pitch_q = [cos(pitch/2), sin(pitch/2), 0, 0]
```
Ici : `pitch=45°` (on regarde vers le bas à 45° depuis l'horizontale, idéal pour voir le sol + une portion du plafond)

**3. Roll (rotation autour de Z)** : tourner la tête à gauche/droite, balayer l'horizon
```python
roll_q = [cos(roll/2), 0, 0, sin(roll/2)]
```
C'est cette rotation qu'on fait varier : `roll = 360° × zone_id / 5` pour couvrir les 360° en 5 vues (0°, 72°, 144°, 216°, 288°)

**Composition** : On combine les trois rotations en multipliant les quaternions :
```python
rq = roll_q × pitch_q × yaw_q
```

L'ordre compte ! Ici on applique d'abord le yaw, puis le pitch, puis le roll. L'ordre de multiplication des quaternions suit la règle : la rotation la plus à droite est appliquée en premier.

#### De quaternion à matrice de projection

Le quaternion nous donne l'orientation de la caméra, mais pour projeter les pixels on a besoin d'une **matrice 3×3 de projection sphérique**.

Cette matrice transforme un point `(x, y)` de l'image de sortie en un point `(X, Y, Z)` sur la demi-sphère virtuelle centrée sur la caméra fisheye.

Les formules proviennent de la conversion quaternion → matrice de rotation, adaptée pour la projection sphérique fisheye. Chaque coefficient de la matrice `m[i,j]` encode comment les coordonnées `(x, y)` de l'output contribuent aux coordonnées `(X, Y, Z)` du point 3D.

Les 9 coefficients de la matrice sont calculés selon ce même principe, chacun combinant les composantes du quaternion de manière spécifique pour encoder la rotation 3D complète.

#### Projection sphérique finale

Une fois qu'on a le point 3D `(X, Y, Z)` sur la demi-sphère, on doit le re-projeter dans l'image fisheye source :
```python
hs = sqrt(X² + Y²)           # Distance horizontale du point
phi = atan2(hs, Z)            # Angle depuis le zénith (0 à π/2)

src_x = width × (X × phi / (π × hs) + 0.5)
src_y = height × (Y × phi / (π × hs) + 0.5)
```

Cette formule implémente la **projection équidistante** (equidistant projection), le modèle standard pour les objectifs fisheye :
- L'angle `phi` (angle depuis le zénith) est proportionnel à la distance radiale dans l'image fisheye
- `X / hs` et `Y / hs` donnent la direction azimutale normalisée
- Le facteur `phi / (π × hs)` convertit l'angle en distance radiale normalisée [0, 0.5]
- Le `+0.5` centre l'image (passage de [-0.5, 0.5] à [0, 1])

Ce modèle équidistant signifie qu'un objet à 45° du centre apparaît à mi-chemin entre le centre et le bord de l'image fisheye, un objet à 90° est exactement au bord.

#### Résumé du pipeline complet

Pour chaque pixel `(i, j)` de l'image dewarpée de sortie :

1. **Appliquer l'offset** : `x = i - offset_width`, `y = j - offset_height`
2. **Projection par matrice** : `(X, Y, Z) = m × (x, y, 1)` → point 3D sur la demi-sphère
3. **Calcul angle sphérique** : `phi = atan2(sqrt(X²+Y²), Z)` → angle depuis le zénith
4. **Projection fisheye inverse** : `(src_x, src_y)` dans l'image source fisheye
5. **Copie du pixel** : `output[i,j] = input[src_y, src_x]` (nearest neighbor)

**Phases distinctes** :
- Étapes 1-4 = **Phase 1 - Calcul du mapping** (exécutée une seule fois à l'initialisation)
- Étape 5 = **Phase 2 - Application du mapping** (exécutée pour chaque frame vidéo)

Cette séparation est cruciale pour les performances : le calcul du mapping est coûteux mais ne se fait qu'une fois. L'application du mapping est répétée des milliers de fois et doit être ultra-optimisée - c'est là que les optimisations vont se concentrer dans les sections suivantes.

### Benchmark
```
🔍 Commande: uv run ./unwarper_python.py ../images/fisheye.jpg -r 1024

======================================================================
📈 RÉSULTATS BENCHMARK
======================================================================
⏱️  Wall time:              1889.36s
⚙️  CPU time (user+sys):     1889.34s
    ├─ User time:           1888.94s
    └─ System time:         0.40s
🔥 CPU utilization:        99%
💻 Cores utilisés:         ~1.0
🧠 Mémoire pic:            646 MB (662148 KB)
======================================================================

💡 Speedup parallèle:      1.00x
   (CPU time / Wall time = 1889.36s / 1889.34s)

```

**Résultats** : 1889 secondes (31 minutes 29 secondes) pour traiter 1024 frames × 5 vues. Soit environ **369 ms par frame et par vue**.

### Analyse

**Performance comparée à FFmpeg** :
- FFmpeg : 208.64s pour 1024 frames × 5 vues → **49.7 ms/frame/vue**
- Python pur : 211s pour 1024 frames × 5 vues → **369 ms/frame/vue**
- **Ratio : Python pur est 9.1× plus lent que FFmpeg**

Mais FFMpeg utilise 6.8 cores, tandis que python n'utilise qu'1 core. En termes de **consommation CPU globale**:
- FFmpeg: 1411.84s CPU time → **276 ms CPU time/view**
- Pure Python: 1889.34s CPU time → **369 ms CPU time/view**
- **Ratio: Pure Python uses only 1.3× more CPU than FFmpeg!**

C'est étonnament efficace pour du Python interprété! Bien entendu cela ne prend pas en compte le décodage et réencodage vidéo côté FFMpeg ce qui explique en partie les résultats.


**✅ Points forts**

- **Code lisible et compréhensible** : 200 lignes de Python clair où chaque étape mathématique est explicite. Idéal pour comprendre l'algorithme, le debugger, ou l'adapter à un nouveau cas d'usage.
- **Consommation mémoire modérée** : 650 MB vs 1800 MB pour FFmpeg, soit **2.8× moins**. La table de remapping pré-calculée est compacte (~9 MB pour 5 vues), et on ne charge qu'une frame à la fois.
- **Baseline de référence solide** : Implémentation correcte et vérifiée qu'on peut utiliser comme point de comparaison pour toutes les optimisations futures.
- **Facilement modifiable** : Besoin de changer l'angle de vue ? Les paramètres de calibration ? Tout est accessible et modifiable sans recompiler quoi que ce soit.
- **Mono-core total** : CPU à 100% signifie qu'on utilise **un seul core**. Le GIL (Global Interpreter Lock) de Python empêche le parallélisme. FFmpeg utilisait 7.7 cores en parallèle, on reste à 1. Dans notre cas d'usage on peut considérer cela comme une qualité, car les ressources CPU sont utilisées efficacement. Python ne consomme que 3 fois plus de ressources CPU que ffmpeg pour le même traitement.


**❌ Points faibles**

- **Dramatiquement lent** : 6.7× plus lent que FFmpeg, pas utilisable en production pour du temps réel.
- **Boucles Python catastrophiques** : On a 960 × 960 × 5 = 4,6 millions d'itérations de boucles Python par frame. Chaque itération implique du bytecode Python interprété (accès dictionnaire, indexation NumPy, assignation, gestion d'exceptions), ce qui est des ordres de grandeur plus lent que du code natif.
- **Aucune vectorisation** : NumPy est utilisé uniquement comme conteneur de données. On n'exploite **aucune** des optimisations SIMD ou des opérations vectorielles batch qu'il offre.

### Verdict

La version Python pur est un outil **pédagogique**, pas une solution de production.

**Ce code est parfait pour** :
- Comprendre exactement comment fonctionne le dewarping fisheye
- Servir de référence pour vérifier la correction des implémentations optimisées
- Prototyper rapidement des variations de l'algorithme (nouveaux angles, calibrations différentes)
- Apprendre les maths derrière (quaternions, projections sphériques)

**Inadapté pour** :
- Production ou temps réel (trop lent)
- Traitement de gros volumes de vidéos
- Tout cas d'usage où la performance compte

**La question maintenant** : peut-on garder la simplicité de Python tout en rattrapant FFmpeg ? La prochaine section explore la vectorisation NumPy - première étape vers des performances acceptables sans quitter Python.

**Code complet** : [github.com/pykoder/fisheye-dewarping](lien-à-adapter)

## 2.3 NumPy vectorisé - boost de performance majeur

### L'approche

On garde exactement le même algorithme que la version Python pur, mais on **élimine toutes les boucles Python** en utilisant les opérations vectorisées de NumPy. L'idée : laisser NumPy (écrit en C optimisé) gérer les millions d'itérations au lieu de l'interpréteur Python.

Le calcul du mapping reste identique (quaternions, matrice de projection), mais la phase d'application devient massivement parallèle grâce au broadcasting et à la vectorisation.

### Les modifications clés

Au lieu d'itérer pixel par pixel avec des boucles Python imbriquées, on calcule tout d'un coup en manipulant des tableaux entiers.

#### Phase 1 : Calcul du mapping vectorisé

**Avant (Python pur)** :
```python
remap_zone = []
for j in range(self.output_height):
    line = []
    y = j - offset_height
    for i in range(self.output_width):
        x = i - offset_width
        # Calculs pour ce pixel...
        line.append((src_y, src_x))
    remap_zone.append(line)
```

**Après (NumPy vectorisé)** :
```python
# Créer une grille de toutes les coordonnées d'un coup
i_coords, j_coords = np.meshgrid(
    np.arange(self.output_width),
    np.arange(self.output_height),
    indexing='xy')

# Aplatir et recentrer les coordonnées
x_coords = i_coords.flatten() * inv_width  - 1.0
y_coords = j_coords.flatten() * inv_height - 1.0

# Empiler en une matrice de coordonnées homogènes
coords = np.column_stack([x_coords, y_coords, np.ones_like(x_coords)]).T

# UNE multiplication matricielle pour TOUS les pixels
xyz = R @ coords

# Calculs vectorisés (appliqués à tous les pixels simultanément)
hs = np.hypot(xyz[0, :],xyz[1, :])
phi = np.arctan2(hs, xyz[2, :])
coeff = phi / (hs * np.pi)
src_x = (self.width * (xyz[0, :] * coeff + 0.5)).astype(np.int32)
src_y = (self.height * (xyz[1, :] * coeff + 0.5)).astype(np.int32)

# Clipper pour rester dans les bornes de l'image
src_x = np.clip(src_x, 0, self.width - 1)
src_y = np.clip(src_y, 0, self.height - 1)

# Reshape en 2D et stocker
zone_mapping = np.stack([
    src_x.reshape((self.output_height, self.output_width)),
    src_y.reshape((self.output_height, self.output_width))
], axis=-1)
```

**Gain** : Au lieu de 960×960 = 921,600 itérations de boucles Python, on a **une seule** multiplication matricielle optimisée en C + quelques opérations vectorisées. NumPy utilise les instructions SIMD du CPU (SSE, AVX) pour traiter plusieurs valeurs simultanément.

#### Phase 2 : Application du mapping vectorisé

**Avant (Python pur)** :
```python
for i in range(self.output_height):
    for j in range(self.output_width):
        try:
            output_buffer[i, j] = image[remap_table[i][j]]
        except IndexError:
            output_buffer[i, j] = [0, 0, 0]
```

**Après (NumPy vectorisé)** :
```python
# Extraire les coordonnées sources
src_x = remap_table[:, :, 0]
src_y = remap_table[:, :, 1]

# Indexation avancée NumPy : copie TOUS les pixels d'un coup
output_buffer = image[src_y, src_x]
```

**Détail critique** : Le piège du masque de validité

Une première version tentait de gérer explicitement les pixels hors-limites avec un masque booléen :
```python
valid_mask = ((src_y >= 0) & (src_y < self.height) &
              (src_x >= 0) & (src_x < self.width))
output_buffer[valid_mask] = image[src_y[valid_mask], src_x[valid_mask]]
```

**Impact désastreux** : le benchmark passe de **7.97s à 18.58s** ! Pourquoi un simple masque de validité ralentit-il de 2.3× ?

Le masque booléen casse la **localité mémoire**. Sans masque, NumPy accède aux pixels de façon relativement séquentielle, exploitant les caches CPU. Avec le masque, les accès deviennent aléatoires et dispersés - chaque pixel valide peut être n'importe où dans l'image source. Le CPU passe son temps à attendre des données du RAM au lieu de calculer.

**Solution élégante** : Clipper les coordonnées lors du calcul du mapping (Phase 1) :
```python
src_x = np.clip(src_x, 0, self.width - 1)
src_y = np.clip(src_y, 0, self.height - 1)
```

Les quelques pixels qui dépassent pointent maintenant vers le bord de l'image (artefact visuel négligeable sur quelques pixels) mais **tous les accès mémoire restent valides et séquentiels**. NumPy peut optimiser agressivement l'indexation.

### Benchmark
```
Commande: python3 unwarper_numpy.py -r 1024 ../images/fisheye.jpg

======================================================================
RESULTATS BENCHMARK
======================================================================
Wall time:              110.11s
CPU time (user+sys):     113.83s
  - User time:           113.14s
  - System time:         0.69s
CPU utilization:        103%
Cores utilises:         ~1.0
Memoire pic:            255.73 MB (261868 KB)
======================================================================

Speedup parallele:      1.03x
```

**Résultats** : 110 secondes pour traiter 1024 frames × 5 vues. Soit environ **21.5 ms par frame et par vue**.

**Performance comparée** :
- FFmpeg : 208s, , 1411.84s CPU (6.8 cores) → **276 ms/frame/vue/CPU**
- Python pur : 1889.36s wall, 1889.34s CPU (1.0 core) → **369 ms/frame/vue**
- NumPy vectorisé : 110.11s → **21.5 ms/frame/vue/CPU**

### Analyse

**Gains significatifs** :
- **17× plus rapide** que Python pur
- **1.9× plus rapide** que FFmpeg
- **12.4× moins de CPU** utilisé que FFmpeg


La vectorisation NumPy élimine le coût catastrophique des boucles Python.

**La magie de NumPy (partielle)** :

✅ **Code C optimisé** : Les opérations NumPy sont implémentées en C hautement optimisé avec `-O3`.

✅ **Vectorisation SIMD limitée** : Les calculs mathématiques (`hypot`, `arctan2`) exploitent les instructions AVX pour traiter 4-8 float64 simultanément. Gain réel sur la Phase 1.

✅ **Localité mémoire** : Les opérations vectorisées accèdent à la mémoire séquentiellement, maximisant l'efficacité du cache (sauf si on utilise le masque de validité !).

❌ **Parallélisme limité** : NumPy ne parallélise que les grosses multiplications matricielles. L'indexation avancée reste mono-thread. Contrairement à FFmpeg qui parallélise le décodage vidéo + les filtres sur tous les cores.

❌ **GIL partiellement présent** : Certaines opérations NumPy relâchent le GIL, d'autres non. L'indexation avancée garde souvent le GIL, limitant le parallélisme.

**✅ Points forts**

- **Performance acceptable** : 1.9× plus rapide que FFmpeg, 17× plus rapide que Python pur. Utilisable en production pour des volumes modérés.
- **Code toujours en Python** : Gardé la simplicité et la lisibilité de Python. Facile à modifier, debugger, intégrer dans un pipeline existant.
- **Pas de compilation** : Aucun toolchain C++, CMake ou dépendances système complexes. Juste `pip install numpy` et ça tourne.
- **Mémoire optimisée** : 256 MB vs 638 MB (Python pur) et 1761 MB (FFmpeg). La table de mapping NumPy est compacte (array dense contiguë en mémoire).

**❌ Points faibles**

- **Parallélisme décevant** : 1 core vs 6.8 pour FFmpeg. On n'exploite pas le potentiel multi-core de la machine. L'indexation avancée reste le goulot mono-thread. Mais dans notre cas d'usage ce n'est pas un inconvénient.
- **Pas de gain énorme sur FFmpeg** : 1.9× plus rapide, c'est bien mais pas spectaculaire. FFmpeg reste compétitif grâce à son parallélisme agressif.
- **Dépendance NumPy** : Nécessite NumPy + ses backends (OpenBLAS ou MKL). Packaging plus lourd qu'un simple script Python, possibles conflits de versions.
- **Courbe d'apprentissage** : Broadcasting, indexation avancée, pièges de performance (masque de validité) - il faut maîtriser ces concepts pour ne pas se tirer une balle dans le pied.
- **Optimisations limitées** : On ne peut pas tweaker finement les stratégies d'accès mémoire ou le threading. NumPy décide pour nous.

### Verdict

NumPy vectorisé apporte un **gain substantiel de 17× sur Python pur**, prouvant que la vectorisation fonctionne. Mais le gain modeste sur FFmpeg (1.4×) révèle les limites de cette approche : **on reste fondamentalement mono-thread** sur la partie critique (indexation).

**Cette implémentation est adaptée pour** :
- Pipelines Python existants où ajouter du C++ serait compliqué
- Prototypage rapide avec performances acceptables
- Situations où 1.4× plus rapide que FFmpeg suffit et où la simplicité de déploiement prime

**Limitations** :
- Pour exploiter vraiment le multi-core, il faudrait sortir de Python
- L'indexation avancée NumPy ne répartit pas les traitements entre plusieurs coeurs, mais utilise seulement le SIMD.

**Peut-on faire mieux en restant en Python ?** Oui - la prochaine section explore OpenCV Python, qui offre des fonctions dédiées au dewarping fisheye avec des optimisations spécifiques au traitement d'image.

**Code complet** : [github.com/pykoder/fisheye-dewarping](Code Source)

---

**Encore plus vite ?** La seconde partie de cet article explore trois implémentations supplémentaires : OpenCV Python (en utilisant la primitive `cv2.remap()`), OpenCV C++ (code OpenCV natif compilé), pour finir par une bibliothèque C++ personnalisée qui offre des performances **42× plus rapide** que FFmpeg. Comment ? A lire dans la seconde partie !

---

## Quoi de neuf dans la partie 2 2

Dans le prochain article nous explorons trois autres implémentations qui poussent le gain en performances toujours plus loin.

1. **OpenCV Python**: en utilisant `cv2.remap()` avec du parallèlisme multi-core (~4.8 cores)
2. **OpenCV C++**: code openCV natif compilé, pour éliminer l'overhead Python.
3. **Bibliothèque C++ personnalisée**: l'optimisation ultime - 42× plus rapide que FFmpeg en consommant seulement 80 MB de RAM

Nous verrons:
- Comment le multi-threading d'OpenCV permet de dépasser l'efficacité mono-core de NumPy's
- Si du code C++ natif permet des gains significatifs par rapport à Python
- Quelles optimisation permettent à une bibliothèque C++ ad-hoc d'atteindre des performances de 4,91s pour 5120 vues. (0.96 ms/vue !)

**Points à retenir de la partie 1**:
- FFmpeg: rapide mais coûteux en mémoire (1.78 GB), forte consommation CPU (1411s)
- Pur Python: lent mais efficace dans l'utilisation du CPU (seulement 1.3× pire que FFmpeg par core)
- NumPy: Champion de l'efficacité - 12.4× moins d'utilisation du CPU que FFmpeg, 7× moins de mémoire

**Spoiler alert**: The custom C++ library will process the same workload in just **4.91 seconds** using **1.1 cores** and **80 MB RAM**. That's:
- **42.5× faster** than FFmpeg in wall time
- **256× less CPU** consumption than FFmpeg
- **22× less memory** than FFmpeg

How is this possible? Find out in Part 2!


**Full code for all implementations**: [github.com/pykoder/fisheye-dewarping](https://github.com/pykoder/fisheye-dewarping)

*Article written in December 2025. Benchmarks performed on a Lenovo ThinkPad P14s - Ubuntu 25.04, Intel Core i7-1185G7 (4 physical cores, 8 threads), 16GB RAM. All tests process 1024 frames × 5 views = 5,120 dewarped images.*