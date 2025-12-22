# Fisheye Dewarping : Du Memory Leak à 6 Implémentations Comparées

*TL;DR : Face à un bug critique en prod, j'ai dû recoder un dewarper fisheye from scratch. Trois ans plus tard, je revisite ce problème pour comparer 6 approches différentes - FFmpeg ligne de commande, Python pur, NumPy vectorisé, OpenCV Python/C++, et une lib C++ custom optimisée.*

---

## L'incident qui a tout lancé

**Janvier 2022.** Notre conteneur Docker plante régulièrement en prod après quelques heures sur certains sites. Le diagnostic tombe : memory leak dans la bibliothèque propriétaire qui gère le dewarping des caméras fisheye 360°.

Contexte rapide : notre IA analyse des flux vidéo pour détecter des gestes. Nos modèles sont entraînés sur des vues plates, pas sur des vues circulaires déformées. Sans dewarping, **zéro détection** sur les caméras fisheye (environ 1% du parc, mais des clients importants).

La lib qui plante ? Fournie par le fabricant des caméras. On ne peut ni la patcher, ni attendre un correctif.

**Verdict** : on code notre propre version.

Quinze jours de plongée dans les bouquins de géométrie projective, une implémentation C++ à base de quaternions et projections sphériques, quelques semaines d'optimisation... et on avait notre solution en prod.

**Trois ans plus tard**, avec un peu de recul et du temps libre, une question me trotte dans la tête : **et si on l'avait fait autrement ?**

Quelle approche aurait été la plus efficace selon nos contraintes :
- Caméras fisheye **fixes** au plafond
- Besoin de **5 vues plates** par frame
- Points de vue des caméras virtuelles **statiques** (pas de rotation dynamique)
- **Performance critique** : traitement temps réel de multiples flux vidéo
- Qualité "suffisante" pour la détection (pas besoin de perfection photographique)

Cet article compare **6 implémentations différentes** pour ce cas d'usage précis, avec benchmarks à l'appui.

---

## Ce que vous allez découvrir

1. **Les bases théoriques** du dewarping fisheye (version courte, promis)
2. **Six implémentations** du même algorithme :
   - FFmpeg en ligne de commande
   - Python natif (boucles, pas de lib)
   - NumPy vectorisé
   - OpenCV fisheye (Python)
   - OpenCV C++ avec bindings
   - Lib C++ custom optimisée + wrapper Python
3. **Benchmarks comparatifs** : temps d'exécution, RAM, complexité de mise en œuvre
4. **Recommandations** selon votre contexte

**Important** : Cette comparaison est spécifique à *notre* cas d'usage (caméras fixes, vues statiques, perf temps réel). Vos besoins peuvent différer radicalement - caméras mobiles, recalibration dynamique, qualité maximale, etc. Adaptez en conséquence.

---

## Un peu de théorie (juste ce qu'il faut)

### Le problème en image

Une caméra fisheye 360° au plafond capture tout l'espace dans une image circulaire déformée :

Pour que nos algorithmes de détection puissent bosser, on transforme ça en plusieurs vues plates rectangulaires :

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

Cette phase utilise des projections sphériques (et dans notre cas initial, des quaternions pour les rotations). C'est du calcul assez lourd, mais on ne le fait **qu'une seule fois**.

**Phase 2 : Application du mapping (pour chaque frame vidéo)**

Pour chaque nouvelle image de la caméra fisheye :
1. On parcourt nos 5 vues de sortie
2. Pour chaque pixel, on regarde dans la lookup table d'où il vient
3. On copie la couleur du pixel source (avec interpolation optionnelle si on veut de la qualité)

C'est cette phase qu'on doit optimiser à fond - elle tourne en boucle sur chaque frame.

### Note sur le calibrage

Les lentilles fisheye varient d'un modèle à l'autre. Un calibrage précis permet d'obtenir des vues parfaitement rectilignes. Dans notre cas, on s'en passe : nos algos de détection tolèrent de légères déformations résiduelles. Ça simplifie le code et booste les perfs.

---

## 2.1 FFmpeg CLI - La solution rapide et parallèle

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

2. **Rotation de l'image source** : Le contrôle du point de vue via `yaw/pitch/roll` est limité et ne permet d'obtenir le résultat souhaité que dans une seule direction. On contourne ça en appliquant une rotation préalable de l'image fisheye (multiples de 72° pour couvrir les 360°).

3. **Lecture unique du fichier source** : Toutes les vues sont générées en un seul passage de FFmpeg (une commande avec 5 outputs). Essentiel pour les perfs.

4. **Vidéo de 128 frames** : Les benchmarks utilisent une vidéo suffisamment longue pour que le temps d'application du mapping domine largement le temps de calcul initial du mapping (la phase qui nous intéresse vraiment).

### Benchmark
```
Commande: ./unwarper_ffmpeg.sh ../images/fisheye_pharma_gde_1920.mp4

======================================================================
RESULTATS BENCHMARK
======================================================================
Wall time:              10.97s
CPU time (user+sys):     84.27s
  - User time:           81.63s
  - System time:         2.64s
CPU utilization:        767%
Cores utilises:         ~7.7
Memoire pic:            1761.50 MB (1803772 KB)
Page faults:            349553 minor, 0 major
Context switches:       44239 vol, 28518 invol
Exit status:            0

Speedup parallele:      7.68x
(CPU time / Wall time = 84.27s / 10.97s)
======================================================================
```

**Résultats :** 11 secondes pour traiter 128 frames et générer 5 vues. FFmpeg exploite à fond les 8 cores disponibles (~767% d'utilisation CPU), avec un speedup parallèle de 7.68x. 

L'utilisation mémoire grimpe toutefois à **1.7 GB**, ce qui est significatif.

### Analyse

**✅ Points forts**
- **Setup immédiat** : une seule commande, aucune lib à installer (FFmpeg suffit)
- **Parallélisation native** : utilisation optimale du multi-core sans effort
- **Performances brutes excellentes** : 10.97s pour 128 frames × 5 vues = ~85 ms/frame/vue
- **Robuste** : FFmpeg est battle-tested en prod partout
- **Pas de maintenance** : dépendance externe stable, bugs déjà corrigés par la communauté

**❌ Points faibles**
- **Boîte noire totale** : impossible d'auditer ou modifier l'algo de dewarping
- **Flexibilité limitée** : on est coincé avec les paramètres exposés par `v360`
- **Pas intégrable finement** : nécessite de spawner un process externe, impossible d'appeler directement comme une fonction Python
- **Consommation mémoire élevée** : 1.7 GB pour une vidéo 1920×1920, potentiellement problématique à grande échelle
- **Optimisations limitées** : on ne peut pas optimiser la phase de mapping spécifiquement pour notre cas d'usage (caméras fixes, vues statiques)

### Verdict

FFmpeg est **l'arme idéale pour un POC rapide** ou quand vous avez besoin d'un résultat qui marche *maintenant* sans vous poser de questions. Parfait pour :
- Tester si le dewarping résout votre problème métier
- Scripts one-shot ou batch processing occasionnel
- Situations où la RAM n'est pas une contrainte

**Mais inadapté si :**
- Vous devez intégrer le dewarping dans un pipeline Python complexe
- Vous voulez optimiser finement (pré-calcul du mapping une fois, réutilisation)
- La consommation mémoire est critique
- Vous avez besoin de comprendre ou tweaker l'algorithme sous-jacent

Dans notre cas (memory leak de la lib propriétaire), FFmpeg aurait pu être une solution de secours acceptable... mais on aurait vite été limités pour l'optimisation et l'intégration.

**Code complet :** [github.com/pykoder/fisheye-dewarping](lien-à-adapter)

---
## 2.2 Python pur - Comprendre les maths

### L'approche

Maintenant qu'on a vu la solution "boîte noire" avec FFmpeg, plongeons dans les entrailles de l'algorithme. Cette implémentation en **Python pur** utilise uniquement les bibliothèques standard et NumPy pour la manipulation de tableaux, mais **sans aucune vectorisation**.

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

import numpy as np
from PIL import Image
import math
from typing import Tuple, List


def multiply_quaternion(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Multiply two quaternions using Python primitives.
    
    Args:
        a: First quaternion [w, x, y, z]
        b: Second quaternion [w, x, y, z]
        
    Returns:
        Result quaternion [w, x, y, z]
    """
    w1, x1, y1, z1 = a[0], a[1], a[2], a[3]
    w2, x2, y2, z2 = b[0], b[1], b[2], b[3]
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return np.array([w, x, y, z], dtype=np.float64)


class PythonDewarper:
    """
    Pure Python implementation of the fisheye dewarper.    
    """
    
    def __init__(self, width: int, height: int, zones: int = 5):
        """
        Initialize dewarper with image dimensions.
        
        Args:
            width: Image width in pixels (1920)
            height: Image height in pixels (1920)
            zones: Number of perspective views to generate (5)
        """
        self.width = width
        self.height = height
        self.output_width = self.width // 2   # 960px
        self.output_height = self.height // 2  # 960px
        self.zones = zones
        
        # Pre-calculate remapping tables for all views (Phase 1)
        self.remap = self._dewarp_mapping()
    
    def _dewarp_mapping(self) -> List[List[List[Tuple[int, int]]]]:
        """
        Create pixel remapping tables for all views.
        
        This is Phase 1: calculating the mapping once at initialization.
        Returns a 3D lookup table: [zone_id][y][x] -> (src_y, src_x)
        """
       
        # Base orientation: yaw=0, pitch=45° (looking down at 45°)
        yaw = 0 * np.pi / 360.0
        sin_yaw, cos_yaw = np.sin(yaw), np.cos(yaw)
        yaw_q = np.array([cos_yaw, 0.0, sin_yaw, 0.0], dtype=np.float64)
        
        pitch = 45 * np.pi / 360.0             
        sin_pitch, cos_pitch = np.sin(pitch), np.cos(pitch)
        pitch_q = np.array([cos_pitch, sin_pitch, 0.0, 0.0], dtype=np.float64)

        remap = []
        for zone_id in range(self.zones):
            # Rotate view around Z axis to cover 360° in N zones
            roll = (360.0 * zone_id / self.zones) * np.pi / 360.0
            sin_roll, cos_roll = np.sin(roll), np.cos(roll)
            roll_q = np.array([cos_roll, 0.0, 0.0, sin_roll], dtype=np.float64)
    
            # Combine rotations: roll × pitch × yaw
            rq = multiply_quaternion(multiply_quaternion(roll_q, pitch_q), yaw_q)
            
            # Calibration constants to match FFmpeg's camera positioning
            expand = 1.269  # Zoom out slightly
            offset = 0.25   # Move camera back from fisheye center
         
            # Build 3×3 spherical projection matrix from quaternion
            m = np.array([[
                    expand * 4.0 * (rq[0]**2 + rq[1]**2 - rq[2]**2 - rq[3]**2) / self.width,
                    expand * 4.0 * (-rq[0] * rq[3] + rq[1] * rq[2] + rq[2] * rq[1] - rq[3] * rq[0]) / self.height,
                    4.0 * (rq[0] * rq[2] + rq[1] * rq[3] + rq[2] * rq[0] + rq[3] * rq[1]) / np.pi
                ],
                [
                    expand * 4.0 * (rq[0] * rq[3] + rq[1] * rq[2] + rq[2] * rq[1] + rq[3] * rq[0]) / self.width,
                    expand * 4.0 * (rq[0]**2 - rq[1]**2 + rq[2]**2 - rq[3]**2) / self.height,
                    4.0 * (-rq[0] * rq[1] - rq[1] * rq[0] + rq[2] * rq[3] + rq[3] * rq[2])  / np.pi
                ],
                [
                    expand * 4.0 * (-rq[0] * rq[2] + rq[1] * rq[3] - rq[2] * rq[0] + rq[3] * rq[1]) / self.width,
                    expand * 4.0 * (rq[0] * rq[1] + rq[1] * rq[0] + rq[2] * rq[3] + rq[3] * rq[2]) / self.height,
                    4.0 * (rq[0]**2 - rq[1]**2 - rq[2]**2 + rq[3]**2)  / np.pi
                ]], dtype=np.float64)

            # Build remapping table for this zone
            offset_width = offset * self.width
            offset_height = offset * self.height

            remap_zone = []
            for j in range(self.output_height):
                line = []
                y = j - offset_height
                for i in range(self.output_width):
                    x = i - offset_width
                    
                    # Apply projection matrix: get 3D point on hemisphere
                    xyz = ((m[0, 0] * x + m[0, 1] * y + m[0, 2]),
                           (m[1, 0] * x + m[1, 1] * y + m[1, 2]),
                           (m[2, 0] * x + m[2, 1] * y + m[2, 2]))
                    
                    # Convert 3D point to fisheye coordinates
                    hs = np.hypot(xyz[0], xyz[1])  # Horizontal distance
                    phi = np.arctan2(hs, xyz[2])    # Angle from zenith
                    
                    # Map to source pixel in fisheye image
                    src_x = int(self.width * (xyz[0] * phi / (np.pi * hs) + 0.5))
                    src_y = int(self.height * (xyz[1] * phi / (np.pi * hs) + 0.5))
                    
                    line.append((src_y, src_x))
                remap_zone.append(line)
            remap.append(remap_zone)

        return remap
    
    def dewarp_frame(self, image: np.ndarray, zone_id: int):
        """
        Apply dewarping transformation to image (Phase 2).
        
        Args:
            image: Input fisheye image as NumPy array (H, W, 3)
            zone_id: Which perspective view to generate (0-4)
        """
        
        remap_table = self.remap[zone_id]
        output_buffer = np.zeros((self.output_height, self.output_width, 3), dtype=np.uint8)

        # Apply lookup table - simple nearest neighbor
        for i in range(self.output_height):
            for j in range(self.output_width):
                try:
                    output_buffer[i, j] = image[remap_table[i][j]]
                except IndexError:
                    output_buffer[i, j] = [0, 0, 0]  # Black for out-of-bounds

        return output_buffer
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

Exemple pour `m[0,0]` (première ligne, première colonne) :
```python
m[0,0] = expand * 4.0 * (rq[0]² + rq[1]² - rq[2]² - rq[3]²) / width
```

Cette formule vient de la conversion standard quaternion → matrice de rotation, avec :
- Un facteur `4.0` pour la normalisation sphérique
- Division par `width/height` pour passer en coordonnées normalisées [-1, 1]
- Multiplication par `expand` (on y revient juste après)

Les 9 coefficients de la matrice sont calculés selon ce même principe, chacun combinant les composantes du quaternion de manière spécifique pour encoder la rotation 3D complète.

#### Les constantes magiques : `expand` et `offset`

Deux constantes empiriques apparaissent dans le code :

**`expand = 1.269`** : Facteur de zoom
- Sans ce facteur, la projection serait trop "serrée" et on ne verrait qu'une portion limitée du champ de vision
- Avec `expand=1.269`, on "recule" légèrement la caméra virtuelle pour capturer un champ de vision plus large
- Cette valeur a été ajustée pour **matcher exactement le rendu de FFmpeg** (qui applique sa propre calibration interne au filtre `v360`)

**`offset = 0.25`** : Décalage du centre de projection
- Par défaut, la caméra virtuelle serait exactement au centre du fisheye (point zénithal, directement sous la caméra)
- Avec `offset=0.25`, on déplace la caméra de 25% de l'image vers l'arrière
- **Effet visuel** : on voit à la fois le centre de l'image fisheye (la zone directement sous la caméra) ET une portion du plafond/bords
- Cette position "en retrait" donne une vue plus équilibrée et exploitable

Ces deux constantes sont le résultat d'un **reverse engineering empirique** de FFmpeg : on a ajusté les valeurs manuellement jusqu'à obtenir exactement les mêmes vues de sortie que le filtre `v360` avec les paramètres `yaw=0:pitch=45:v_fov=90`. L'objectif était d'avoir une baseline de référence identique à FFmpeg pour comparer équitablement les performances des différentes implémentations.

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
🔍 Commande: uv run ./unwarper_python.py ../images/fisheye.jpg -r 128

======================================================================
📈 RÉSULTATS BENCHMARK
======================================================================
⏱️  Wall time:              210.61s
⚙️  CPU time (user+sys):     211.16s
    ├─ User time:           210.91s
    └─ System time:         0.25s
🔥 CPU utilization:        100%
💻 Cores utilisés:         ~1.0
🧠 Mémoire pic:            646.98 MB (662508 KB)
📄 Page faults:            172455 minor, 19 major
🔄 Context switches:       51 vol, 2313 invol
✅ Exit status:            0
======================================================================

💡 Speedup parallèle:      1.00x
   (CPU time / Wall time = 211.16s / 210.61s)

```

**Résultats** : 211 secondes (3 minutes 31 secondes) pour traiter 128 frames × 5 vues. Soit environ **329 ms par frame et par vue**.

### Analyse

**Performance comparée à FFmpeg** :
- FFmpeg : 11s pour 128 frames × 5 vues → **17 ms/frame/vue**
- Python pur : 211s pour 128 frames × 5 vues → **330 ms/frame/vue**
- **Ratio : Python pur est 20× plus lent que FFmpeg**

**✅ Points forts**

- **Code lisible et compréhensible** : 200 lignes de Python clair où chaque étape mathématique est explicite. Idéal pour comprendre l'algorithme, le debugger, ou l'adapter à un nouveau cas d'usage.
- **Consommation mémoire modérée** : 650 MB vs 1800 MB pour FFmpeg, soit **2.8× moins**. La table de remapping pré-calculée est compacte (~9 MB pour 5 vues), et on ne charge qu'une frame à la fois.
- **Baseline de référence solide** : Implémentation correcte et vérifiée qu'on peut utiliser comme point de comparaison pour toutes les optimisations futures.
- **Facilement modifiable** : Besoin de changer l'angle de vue ? Les paramètres de calibration ? Tout est accessible et modifiable sans recompiler quoi que ce soit.
- **Mono-core total** : CPU à 100% signifie qu'on utilise **un seul core**. Le GIL (Global Interpreter Lock) de Python empêche le parallélisme. FFmpeg utilisait 7.7 cores en parallèle, on reste à 1. Dans notre cas d'usage on peut considérer cela comme une qualité, car les ressources CPU sont utilisées efficacement. Python ne consomme que 3 fois plus de ressources CPU que ffmpeg pour le même traitement.


**❌ Points faibles**

- **Dramatiquement lent** : 20× plus lent que FFmpeg, pas utilisable en production pour du temps réel.
- **Boucles Python catastrophiques** : On a 960 × 960 × 5 = 4,6 millions d'itérations de boucles Python par frame. Chaque itération implique du bytecode Python interprété (accès dictionnaire, indexation NumPy, assignation, gestion d'exceptions), ce qui est des ordres de grandeur plus lent que du code natif.
- **Aucune vectorisation** : NumPy est utilisé uniquement comme conteneur de données. On n'exploite **aucune** des optimisations SIMD ou des opérations vectorielles batch qu'il offre.

### Verdict

Python pur est **20× plus lent** que FFmpeg. C'est un outil **pédagogique**, pas une solution de production.

**Ce code est parfait pour** :
- Comprendre exactement comment fonctionne le dewarping fisheye
- Servir de référence pour vérifier la correction des implémentations optimisées
- Prototyper rapidement des variations de l'algorithme (nouveaux angles, calibrations différentes)
- Apprendre les maths derrière (quaternions, projections sphériques)

**Inadapté pour** :
- Production ou temps réel (20× trop lent, 3x trop consommateur de ressources)
- Traitement de gros volumes de vidéos
- Tout cas d'usage où la performance compte

**La question maintenant** : peut-on garder la simplicité de Python tout en rattrapant FFmpeg ? La prochaine section explore la vectorisation NumPy - première étape vers des performances acceptables sans quitter Python.

**Code complet** : [github.com/pykoder/fisheye-dewarping](lien-à-adapter)

## 2.3 NumPy vectorisé - Le premier boost de performance

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
j_coords, i_coords = np.meshgrid(
    np.arange(self.output_width),
    np.arange(self.output_height), 
    indexing='ij')

# Aplatir et appliquer l'offset
x_coords = i_coords.flatten() - offset_width
y_coords = j_coords.flatten() - offset_height

# Empiler en une matrice de coordonnées homogènes
coords = np.column_stack([x_coords, y_coords, np.ones_like(x_coords)])

# UNE multiplication matricielle pour TOUS les pixels
xyz = coords @ m.T

# Calculs vectorisés (appliqués à tous les pixels simultanément)
hs = np.hypot(xyz[:, 0], xyz[:, 1])
phi = np.arctan2(hs, xyz[:, 2])

# Coordonnées source pour tous les pixels
src_x = (self.width * (xyz[:, 0] * phi / (np.pi * hs) + 0.5)).astype(np.int32)
src_y = (self.height * (xyz[:, 1] * phi / (np.pi * hs) + 0.5)).astype(np.int32)

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
Commande: python3 unwarper_numpy.py -r 128 ../images/fisheye.jpg

======================================================================
RESULTATS BENCHMARK
======================================================================
Wall time:              7.97s
CPU time (user+sys):     10.69s
  - User time:           10.58s
  - System time:         0.11s
CPU utilization:        134%
Cores utilises:         ~1.3
Memoire pic:            256.19 MB (262336 KB)
======================================================================

Speedup parallele:      1.34x
(CPU time / Wall time = 10.69s / 7.97s)
```

**Résultats** : 7.97 secondes pour traiter 128 frames × 5 vues. Soit environ **12.5 ms par frame et par vue**.

**Performance comparée** :
- FFmpeg : 10.97s → **17 ms/frame/vue**
- Python pur : 274.32s → **428 ms/frame/vue**
- NumPy vectorisé : 7.97s → **12.5 ms/frame/vue**

### Analyse

**Gains significatifs** :
- **34× plus rapide** que Python pur (274.32s → 7.97s)
- **1.4× plus rapide** que FFmpeg (10.97s → 7.97s)

La vectorisation NumPy élimine le coût catastrophique des boucles Python, mais on n'obtient pas les gains astronomiques qu'on aurait pu espérer. Pourquoi ?

**Analyse du parallélisme** : CPU utilization à 134% signifie qu'on utilise ~1.3 cores. C'est mieux que Python pur (1.0 core) mais **très loin** de FFmpeg (7.7 cores). NumPy parallélise certaines opérations (multiplication matricielle, fonctions transcendantes) mais l'indexation avancée `image[src_y, src_x]` reste largement séquentielle.

**La magie de NumPy (partielle)** :

✅ **Code C optimisé** : Les opérations NumPy sont implémentées en C hautement optimisé avec `-O3`.

✅ **Vectorisation SIMD limitée** : Les calculs mathématiques (`hypot`, `arctan2`) exploitent les instructions AVX pour traiter 4-8 float64 simultanément. Gain réel sur la Phase 1.

✅ **Localité mémoire** : Les opérations vectorisées accèdent à la mémoire séquentiellement, maximisant l'efficacité du cache (sauf si on utilise le masque de validité !).

❌ **Parallélisme limité** : NumPy ne parallélise que les grosses multiplications matricielles. L'indexation avancée reste mono-thread. Contrairement à FFmpeg qui parallélise le décodage vidéo + les filtres sur tous les cores.

❌ **GIL partiellement présent** : Certaines opérations NumPy relâchent le GIL, d'autres non. L'indexation avancée garde souvent le GIL, limitant le parallélisme.

**✅ Points forts**

- **Performance acceptable** : 1.4× plus rapide que FFmpeg, 34× plus rapide que Python pur. Utilisable en production pour des volumes modérés.
- **Code toujours en Python** : Gardé la simplicité et la lisibilité de Python. Facile à modifier, debugger, intégrer dans un pipeline existant.
- **Pas de compilation** : Aucun toolchain C++, CMake ou dépendances système complexes. Juste `pip install numpy` et ça tourne.
- **Mémoire optimisée** : 256 MB vs 638 MB (Python pur) et 1761 MB (FFmpeg). La table de mapping NumPy est compacte (array dense contiguë en mémoire).
- **Léger multithreading** : ~1.3 cores utilisés vs 1.0 pour Python pur. NumPy parallélise automatiquement certaines opérations.

**❌ Points faibles**

- **Parallélisme décevant** : 1.3 cores vs 7.7 pour FFmpeg. On n'exploite pas le potentiel multi-core de la machine. L'indexation avancée reste le goulot mono-thread.
- **Pas de vrai gain sur FFmpeg** : 1.4× plus rapide, c'est bien mais pas spectaculaire. FFmpeg reste compétitif grâce à son parallélisme agressif.
- **Dépendance NumPy** : Nécessite NumPy + ses backends (OpenBLAS ou MKL). Packaging plus lourd qu'un simple script Python, possibles conflits de versions.
- **Courbe d'apprentissage** : Broadcasting, indexation avancée, pièges de performance (masque de validité) - il faut maîtriser ces concepts pour ne pas se tirer une balle dans le pied.
- **Optimisations limitées** : On ne peut pas tweaker finement les stratégies d'accès mémoire ou le threading. NumPy décide pour nous.

### Verdict

NumPy vectorisé apporte un **gain substantiel de 34× sur Python pur**, prouvant que la vectorisation fonctionne. Mais le gain modeste sur FFmpeg (1.4×) révèle les limites de cette approche : **on reste fondamentalement mono-thread** sur la partie critique (indexation).

**Cette implémentation est adaptée pour** :
- Pipelines Python existants où ajouter du C++ serait compliqué
- Prototypage rapide avec performances acceptables
- Situations où 1.4× plus rapide que FFmpeg suffit et où la simplicité de déploiement prime

**Limitations** :
- Pour exploiter vraiment le multi-core, il faut sortir de Python
- L'indexation avancée NumPy ne parallélise pas bien
- On plafonne à ~1.3 cores quoi qu'on fasse en pur NumPy

**Peut-on faire mieux en restant en Python ?** Oui - la prochaine section explore OpenCV Python, qui offre des fonctions dédiées au dewarping fisheye avec des optimisations spécifiques au traitement d'image.

**Code complet** : [github.com/pykoder/fisheye-dewarping](lien-à-adapter)

---
### OpenCV : La Puissance du C++

## 2.4 OpenCV Python - La fonction dédiée cv2.remap()

### L'approche

OpenCV est **la** bibliothèque de référence en computer vision. Elle offre une fonction spécialisée pour les transformations géométriques : `cv2.remap()`, conçue spécifiquement pour appliquer des tables de correspondance pixel à pixel.

On garde notre algorithme de mapping (quaternions, projection sphérique) mais on délègue la phase d'application à OpenCV. Bonus : on a pu simplifier les formules de projection en utilisant une approche plus classique basée sur la matrice de caméra, éliminant les constantes empiriques `expand` et `offset`.

### Les modifications clés

#### Projection simplifiée

**Avant (avec constantes magiques)** :
```python
expand = 1.269  # Facteur empirique
offset = 0.25   # Offset empirique
# ... formules complexes avec ces constantes
```

**Après (projection classique)** :
```python
# Matrice de caméra pour la vue perspective
K = np.array([
    [self.output_width / 2, 0, self.output_width / 2],
    [0, self.output_height / 2, self.output_height / 2],
    [0, 0, 1]
], dtype=np.float32)

# Projection inverse : pixels → rayons 3D
rays = np.linalg.inv(K) @ xyz

# Normalisation et rotation
rays = rays / np.linalg.norm(rays, axis=0, keepdims=True)
rays_fisheye = R @ rays

# Projection fisheye équidistante
theta = np.arccos(np.clip(rays_fisheye[2, :], -1, 1))
phi = np.arctan2(rays_fisheye[1, :], rays_fisheye[0, :])
r = theta * self.width / np.pi

x = r * np.cos(phi) + self.width / 2
y = r * np.sin(phi) + self.height / 2
```

Formules standard de projection perspective + fisheye équidistante. Plus besoin de reverse-engineer FFmpeg.

#### Application avec cv2.remap()

**NumPy vectorisé** :
```python
src_x = remap_table[:, :, 0]
src_y = remap_table[:, :, 1]
output_buffer = image[src_y, src_x]
```

**OpenCV** :
```python
map_x = remap_table[:, :, 0].astype(np.float32)
map_y = remap_table[:, :, 1].astype(np.float32)

output = cv2.remap(
    image, 
    map_x, 
    map_y, 
    cv2.INTER_NEAREST,              # Plus proche voisin
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=(0, 0, 0)
)
```

**Différences notables** :

1. **Type `float32` obligatoire** : OpenCV exige des maps en `float32` (vs `int32` pour NumPy). Conversion nécessaire.

2. **Clipping automatique** : `cv2.remap()` gère automatiquement les pixels hors-limite via `borderMode`. On ne peut pas le désactiver - OpenCV vérifie systématiquement les bornes. Avec `BORDER_CONSTANT`, les pixels hors-limite deviennent noirs.

3. **Interpolation explicite** : `INTER_NEAREST` pour performances maximales. `INTER_LINEAR` ou `INTER_CUBIC` disponibles pour meilleure qualité.

### Benchmark
```
Commande: python3 unwarper_opencv.py -r 1000 ../images/fisheye.jpg

======================================================================
RESULTATS BENCHMARK
======================================================================
Wall time:              12.35s
CPU time (user+sys):     56.00s
  - User time:           42.85s
  - System time:         13.15s
CPU utilization:        453%
Cores utilises:         ~4.5
Memoire pic:            288.39 MB (295316 KB)
======================================================================

Speedup parallele:      4.53x
(CPU time / Wall time = 56.00s / 12.35s)
```

**Comparaison NumPy** (même test avec 1000 répétitions) :
```
Commande: python3 unwarper_numpy.py -r 1000 ../images/fisheye.jpg

======================================================================
RESULTATS BENCHMARK
======================================================================
Wall time:              56.57s
CPU time (user+sys):     60.09s
CPU utilization:        106%
Cores utilises:         ~1.1
======================================================================

Speedup parallele:      1.06x
```

**Résultats** : 
- OpenCV : 12.35s wall time, 56.00s CPU time
- NumPy : 56.57s wall time, 60.09s CPU time

### Analyse

**Comparaison des ressources CPU** :

Sur une machine partagée, ce qui compte c'est le **CPU time total** consommé (charge sur la machine), pas seulement le wall clock time.

- NumPy : **60.09s CPU time** pour traiter 5000 vues (1000 frames × 5 vues)
- OpenCV : **56.00s CPU time** pour le même travail
- **Gain réel : 7% de CPU économisé** (60.09s → 56.00s)

**Pourquoi si peu de différence en CPU time ?**

Le multithreading d'OpenCV (4.5 cores) **ne réduit pas la charge CPU totale** - il la distribue juste sur plus de cores. Si on additionne le temps CPU de tous les cores, on arrive à un total similaire à NumPy.

**Wall time vs CPU time** :
- **Wall time** : OpenCV gagne 4.6× (56.57s → 12.35s) grâce au parallélisme
- **CPU time** : OpenCV gagne seulement 7% (60.09s → 56.00s)

Si la machine fait d'autres choses en parallèle, monopoliser 4.5 cores (OpenCV) vs 1.1 core (NumPy) peut être problématique. Le gain en wall time se paie par une occupation CPU plus élevée.

**Efficacité parallèle** :
- NumPy : 1.06× speedup (quasi linéaire, pas de surcoût de parallélisation)
- OpenCV : 4.53× speedup mais consomme 4.5 cores → efficacité = 4.53/4.5 = **100%** (excellent)

Le parallélisme d'OpenCV est bien implémenté (pas de surcoût significatif), mais ça ne réduit pas la charge CPU totale - juste la durée.

**✅ Points forts**

- **Wall time excellent** : 4.6× plus rapide que NumPy en temps réel. Idéal pour pipelines interactifs.
- **Multithreading automatique** : ~4.5 cores utilisés sans une ligne de code multithread.
- **Parallélisme efficace** : 100% d'efficacité (4.53× speedup sur 4.5 cores).
- **Projection simplifiée** : Formules classiques, plus de constantes magiques à ajuster.
- **Code Python propre** : Pas de compilation C++, juste `pip install opencv-python`.
- **Mémoire stable** : 288 MB, similaire à NumPy, bien mieux que FFmpeg (1761 MB).

**❌ Points faibles**

- **Gain CPU marginal** : Seulement 7% de CPU économisé vs NumPy. Si la machine est chargée, le gain peut être contre-productif (on monopolise plus de cores).
- **Dépendance OpenCV** : ~90 MB à installer (vs ~20 MB pour NumPy). Installation parfois capricieuse.
- **Clipping obligatoire** : `cv2.remap()` vérifie systématiquement les bornes. Impossible de le désactiver même si on a pré-clippé les coordonnées.
- **Conversion de types** : Maps en `float32` obligatoire. Conversion depuis `int32` à chaque appel.
- **Moins flexible** : Boîte noire. Impossible de tweaker l'implémentation de `remap()`.

### Verdict

OpenCV Python apporte un **gain en wall time de 4.6×** grâce au multithreading, mais consomme **presque autant de CPU total** que NumPy (7% d'économie seulement).

**Cette implémentation est adaptée pour** :
- Pipelines interactifs où la latence compte (wall time critique)
- Machines dédiées où monopoliser 4-5 cores n'est pas un problème
- Applications nécessitant aussi d'autres fonctions OpenCV (évite une dépendance supplémentaire)

**Inadaptée pour** :
- Serveurs partagés avec forte charge CPU (monopolise trop de cores pour peu de gain)
- Traitement batch où la charge CPU totale prime sur le wall time
- Environnements où OpenCV est difficile à déployer

**Question** : Peut-on faire mieux en sortant de Python ? La section suivante explore OpenCV C++ appelé via des bindings Python - même algorithme mais code natif compilé. Est-ce que ça améliore le CPU time ou seulement le wall time ?

**Code complet** : [github.com/TON_USER/fisheye-dewarping/tree/main/04_opencv_python](lien-à-adapter)

---


## 2.5 OpenCV C++ - Code natif compilé

### L'approche

Après avoir exploré les limites du Python, passons au **C++ natif**. Même algorithme, même `cv::remap()`, mais cette fois compilé directement en binaire sans interpréteur Python. 

Objectif : mesurer le coût réel de l'interprétation Python et voir si le C++ apporte un gain significatif.

### Le code C++

Le code reprend exactement la même logique que la version Python OpenCV, mais en C++ natif. L'implémentation est directe, sans surprises - quaternions, matrices de rotation, projection sphérique, puis appel à `cv::remap()`.

### Détails d'implémentation

**Impact de l'interpolation : différence Python vs C++**

Le choix d'interpolation a un impact **très différent** selon qu'on est en Python ou en C++ :

**OpenCV C++** :
- `INTER_NEAREST` : 7.00s wall, 23.01s CPU
- `INTER_LINEAR` : 12.87s wall, 58.32s CPU
- **Impact : 2.5× plus de CPU, 1.8× plus lent**

**OpenCV Python** :
- `INTER_NEAREST` : 12.35s wall, 56.00s CPU
- `INTER_LINEAR` : 14.46s wall, 72.32s CPU
- **Impact : 1.3× plus de CPU, 1.2× plus lent**

**Explication** : En Python, l'overhead de l'interpréteur et des conversions NumPy/OpenCV "noie" partiellement le coût de l'interpolation. Le temps passé dans les couches Python est incompressible et masque l'impact du choix d'interpolation.

En C++ pur, **100% du temps est dans le code critique** (`cv::remap()`). Chaque cycle CPU compte. Le surcoût de l'interpolation linéaire (4 accès mémoire + calculs vs 1 accès) devient dominant.

**Pour nos benchmarks**, on utilise `INTER_NEAREST` pour maximiser les performances et avoir une comparaison équitable. Pour de la production, `INTER_LINEAR` reste un choix valide si la qualité visuelle prime.

### Benchmark

**Configuration : `cv::INTER_NEAREST`** (tous les benchmarks suivants)
```
Commande: ./unwarper ../images/fisheye.jpg --repeat-dewarp 1000

======================================================================
RESULTATS BENCHMARK
======================================================================
Wall time:              7.00s
CPU time (user+sys):     23.01s
  - User time:           21.27s
  - System time:         1.74s
CPU utilization:        328%
Cores utilises:         ~3.3
Memoire pic:            109.95 MB (112584 KB)
======================================================================

Speedup parallele:      3.29x
(CPU time / Wall time = 23.01s / 7.00s)
```

**Comparaisons** (toutes sur 1000 frames × 5 vues, `INTER_NEAREST`) :
- NumPy vectorisé : 56.57s wall, 60.09s CPU, 1.1 cores
- OpenCV Python : 12.35s wall, 56.00s CPU, 4.5 cores
- **OpenCV C++ : 7.00s wall, 23.01s CPU, 3.3 cores**

### Analyse

**Comparaison avec OpenCV Python** :
- **Wall time** : 7.00s vs 12.35s → **1.8× plus rapide**
- **CPU time** : 23.01s vs 56.00s → **2.4× moins de CPU consommé**
- **Mémoire** : 110 MB vs 288 MB → **2.6× moins de RAM**
- **Cores** : 3.3 vs 4.5 → légèrement moins parallèle

**Comparaison avec NumPy** :
- **Wall time** : 7.00s vs 56.57s → **8.1× plus rapide**
- **CPU time** : 23.01s vs 60.09s → **2.6× moins de CPU**

**Le gain du C++ est-il significatif ?**

Par rapport à OpenCV Python, le gain est **modéré mais réel** :
- 1.8× en wall time (pas spectaculaire)
- 2.4× en CPU time (meilleur, économie significative sur serveur partagé)
- 2.6× en mémoire (le gain le plus notable)

**Pourquoi si peu de différence avec Python ?**

Dans les deux cas (Python et C++), on passe **la majorité du temps dans `cv::remap()`**, qui est du code C++ optimisé. L'overhead Python ne représente qu'une fraction du temps total :
- Appel de fonction Python → C (via l'API C de Python)
- Gestion des références et GC Python
- Conversions de types NumPy ↔ cv::Mat

Le ratio 12.35s / 7.00s ≈ 1.8× représente cet overhead Python de ~40% du temps total.

**Le multi-threading diffère légèrement** : OpenCV Python (4.5 cores) vs OpenCV C++ (3.3 cores). Différence probablement due aux configurations OpenMP ou au binding Python qui peut forcer plus de parallélisme.

**Les vrais gains : CPU et mémoire** :
- **CPU time** : 2.4× moins de charge (56s → 23s). Sur un serveur partagé qui traite N flux en parallèle, cette économie est significative.
- **Mémoire** : 2.6× moins (288 MB → 110 MB). Les objets Python (wrappers NumPy, compteurs de références, dictionnaires d'attributs) ajoutent un overhead de ~180 MB. En C++, les `cv::Mat` sont des structures compactes.

**✅ Points forts**

- **CPU time réduit** : 23.01s vs 56.00s (OpenCV Python). Gain de 2.4× sur la charge CPU totale - critique pour serveurs partagés.
- **Mémoire minimale** : 110 MB, 2.6× moins qu'OpenCV Python. Excellent pour environnements contraints ou traitement de nombreux flux simultanés.
- **Multi-threading efficace** : 3.3 cores, speedup de 3.29× → efficacité parallèle de 100%.
- **Pas d'overhead Python** : Pas de GC, pas d'interpréteur, pas de conversions de types. Code natif direct.
- **Performances prédictibles** : Comportement déterministe, pas de pauses GC aléatoires qui peuvent causer des latences.
- **Sensibilité claire aux optimisations** : L'impact de l'interpolation est mesurable (2.5×), permettant de choisir précisément le trade-off performance/qualité.

**❌ Points faibles**

- **Compilation nécessaire** : CMake, toolchain C++ (GCC/Clang), headers OpenCV dev. Beaucoup plus complexe que `pip install opencv-python`.
- **Gain modéré sur Python** : Seulement 1.8× en wall time. Si OpenCV Python suffit déjà, le C++ n'apporte pas un game changer.
- **Moins flexible** : Toute modification du code = recompilation complète. Cycle de développement plus lent qu'en Python.
- **Portabilité limitée** : Binaires spécifiques à la plateforme (Linux x64, Windows, macOS, ARM). Python est portable out-of-the-box.
- **Debugging plus lourd** : GDB, Valgrind, compilation en mode debug vs simple `print()` en Python.
- **Dépendances de build** : Nécessite OpenCV compilé avec les bonnes options (OpenMP, optimisations). Gestion de dépendances plus complexe.

### Verdict

OpenCV C++ apporte un **gain modeste mais réel** : **2.4× moins de CPU, 2.6× moins de RAM, 1.8× plus rapide**.

**Le gain n'est pas spectaculaire** car OpenCV Python passait déjà la majorité du temps dans du code C++. L'overhead Python représentait environ 40% du temps total - maintenant éliminé.

**Cette implémentation est adaptée pour** :
- **Serveurs partagés** où économiser 2.4× de CPU sur chaque flux compte (traitement de dizaines de flux simultanés)
- **Applications contraintes en mémoire** (110 MB vs 288 MB permet de traiter plus de flux en parallèle)
- **Environnements embedded ou edge** où Python est difficile à déployer ou trop lourd
- **Besoin de performances déterministes** (pas de GC qui pause aléatoirement)
- **Optimisation fine** : l'impact clair des choix d'interpolation permet de tuner précisément

**Pas nécessaire si** :
- OpenCV Python suffit déjà (gain de 1.8× en wall time peut ne pas justifier la complexité)
- La simplicité de déploiement et maintenance prime
- L'agilité du développement Python est critique (prototypage rapide, modifications fréquentes)
- Pas de contraintes fortes sur CPU ou mémoire

**Peut-on faire encore mieux ?** La section suivante explore une **bibliothèque C++ custom** écrite from scratch avec gestion mémoire fine, et algorithmes spécifiques à notre cas d'usage (caméras fixes, vues statiques). Peut-on descendre significativement sous 7 secondes ? À quel prix en complexité et maintenabilité ?

**Code complet** : [github.com/TON_USER/fisheye-dewarping/tree/main/05_opencv_cpp](lien-à-adapter)

---

## 2.6 Bibliothèque C++ custom optimisée - L'optimisation ultime

### L'approche

Après avoir exploré OpenCV, passons à une **bibliothèque C++ écrite from scratch** et optimisée spécifiquement pour notre cas d'usage. Plus de dépendance à OpenCV - juste du C++ pur avec gestion mémoire fine et algorithme minimal.

L'idée : garder la simplicité d'appel Python (via ctypes) tout en exploitant des optimisations impossibles avec OpenCV :
- Code minimal sans overhead de bibliothèque générique
- Gestion mémoire optimale (buffers réutilisables)
- Algorithme spécialisé pour notre cas (caméras fixes, pas de recalibration)
- Pas de multithreading (évite la contention, optimal pour mono-flux)

### Architecture

**Côté C++** : Une bibliothèque partagée (.so) exposant une API simple :
```cpp
extern "C" {
    // Créer le contexte de dewarping (calcule le mapping une fois)
    DewarpContext* create_dewarp_context(int width, int height, int zones);
    
    // Appliquer le dewarping (rapide, appelé en boucle)
    void dewarp_frame(DewarpContext* ctx, uint8_t* input, uint8_t* output, int zone_id);
    
    // Libérer le contexte
    void free_dewarp_context(DewarpContext* ctx);
}
```

**Côté Python** : Wrapper ctypes minimal :
```python
import ctypes
import numpy as np

# Charger la bibliothèque
lib = ctypes.CDLL('libunwarper_ctypes.so')

# Configurer les signatures
lib.create_dewarp_context.argtypes = [c_int, c_int, c_int]
lib.create_dewarp_context.restype = c_void_p

lib.dewarp_frame.argtypes = [c_void_p, POINTER(c_uint8), POINTER(c_uint8), c_int]
lib.dewarp_frame.restype = None

# Utilisation
ctx = lib.create_dewarp_context(1920, 1920, 5)
lib.dewarp_frame(ctx, input_ptr, output_ptr, zone_id)
```

### Optimisations clés

**1. Table de mapping compacte** : Stockage en `int16_t` au lieu de `float32` (OpenCV). Économie mémoire et meilleure localité cache.

**2. Boucle de remapping optimisée** :
```cpp
void dewarp_frame(const DewarpContext* ctx, const uint8_t* input_data, 
                  uint8_t* output_data, const int zone_id) {
    const auto* remap_ptr = get_zone_remap_data(ctx, zone_id);
    
    // Traitement ligne par ligne avec buffer local
    for (int j = 0; j < ctx->output_height; ++j) {
        uint8_t buffer[4096];  // Buffer stack, ultra-rapide
        
        for (int i = 0; i < ctx->output_width; ++i) {
            const int remap_offset = (j * ctx->output_width + i);
            const int16_t src_x = remap_ptr[remap_offset * 2];
            const int16_t src_y = remap_ptr[remap_offset * 2 + 1];
            
            const int src_offset = (src_y * ctx->width + src_x) * 3;
            
            // Copie RGB directe dans buffer local
            buffer[i*3]     = input_data[src_offset];
            buffer[i*3 + 1] = input_data[src_offset + 1];
            buffer[i*3 + 2] = input_data[src_offset + 2];
        }
        // Copie groupée du buffer vers output
        memcpy(output_data + j * ctx->output_width * 3, buffer, ctx->output_width * 3);
    }
}
```

**Pourquoi c'est rapide** :
- **Buffer local sur la stack** : Évite les allocations dynamiques répétées
- **Accès séquentiels** : Maximise l'utilisation du cache CPU
- **Pas de vérification de bornes** : Coordonnées pré-clippées dans le mapping
- **Pas de multithreading** : Zéro overhead de synchronisation ou contention

**3. Pas d'interpolation** : Plus proche voisin uniquement. Suffisant pour détection d'objets.

### Benchmark
```
Commande: python3 unwarper_ctypes.py ../images/fisheye.jpg --repeat-dewarp 1024

======================================================================
RESULTATS BENCHMARK
======================================================================
Wall time:              4.91s
CPU time (user+sys):     5.52s
  - User time:           5.48s
  - System time:         0.04s
CPU utilization:        112%
Cores utilises:         ~1.1
Memoire pic:            80.14 MB (82068 KB)
======================================================================

Speedup parallele:      1.12x
(CPU time / Wall time = 5.52s / 4.91s)
```

**Comparaisons** (toutes sur 1024 frames × 5 vues) :
- FFmpeg : 208.64s wall, 1411.84s CPU, 6.8 cores, 1784 MB
- Python pur : 1889.36s wall, 1889.34s CPU, 1.0 core, 647 MB
- NumPy : 110.11s wall, 113.83s CPU, 1.0 core, 256 MB
- OpenCV Python : 22.21s wall, 105.70s CPU, 4.8 cores, 288 MB
- OpenCV C++ : 10.09s wall, 48.90s CPU, 4.8 cores, 110 MB
- **Lib C++ custom : 4.91s wall, 5.52s CPU, 1.1 core, 80 MB**

### Analyse

**Gains spectaculaires sur toutes les métriques** :

**vs OpenCV C++ :**
- **2.1× plus rapide** en wall time (10.09s → 4.91s)
- **8.9× moins de CPU** (48.90s → 5.52s)
- **1.4× moins de mémoire** (110 MB → 80 MB)

**vs OpenCV Python :**
- **4.5× plus rapide** en wall time
- **19× moins de CPU**
- **3.6× moins de mémoire**

**vs FFmpeg :**
- **42× plus rapide** en wall time
- **256× moins de CPU**
- **22× moins de mémoire**

**D'où viennent ces gains massifs ?**

**1. Pas de multithreading = efficacité maximale**

Le paradoxe : on utilise **1.1 core** contre 4.8 pour OpenCV C++, mais on est **2.1× plus rapide** en wall time.

Explication : Le multithreading OpenCV a un **coût caché** :
- Synchronisation entre threads (mutex, barriers)
- Contention sur le cache (false sharing)
- Context switches fréquents (117k pour OpenCV vs 112 pour nous)
- Overhead de création/destruction de threads

Notre code mono-thread évite tout ça. Un seul thread qui tourne à fond, accès mémoire séquentiels, cache CPU optimal.

**Efficacité par core** :
- OpenCV C++ : 10.09s / 4.8 cores = **2.10s/core**
- Lib custom : 4.91s / 1.1 core = **4.46s/core**

Attends, 4.46 > 2.10 ? Non ! C'est un piège de mesure. Le vrai indicateur c'est le **CPU time total** :
- OpenCV C++ : **48.90s de CPU consommé**
- Lib custom : **5.52s de CPU consommé**

On consomme **8.9× moins de ressources CPU** pour le même travail.

**2. Code minimal sans overhead**

OpenCV `cv::remap()` est une fonction générique qui gère :
- Multiples types d'interpolation
- Multiples types de bordures
- Support GPU optionnel
- Vérifications de validité
- Abstraction cv::Mat avec compteurs de références

Notre code fait **exactement ce dont on a besoin, rien de plus** :
- Interpolation nearest neighbor uniquement
- Bordures pré-gérées (clipping dans le mapping)
- Pas d'abstraction, juste des pointeurs bruts
- Pas de vérifications en phase critique

**3. Gestion mémoire optimale**

- **80 MB** vs 110 MB (OpenCV C++) : Économie de 30 MB
- Table de mapping en `int16_t` : 2× plus compact que `float32`
- Pas de structures OpenCV (`cv::Mat` avec headers, refcounting)
- Buffer temporaire sur la stack (pas de malloc)

**4. Localité mémoire parfaite**

Le buffer local ligne par ligne maximise l'utilisation du cache L1/L2. Tous les accès sont dans ~4KB de données (une ligne), qui tient entièrement dans le cache L1 (32KB sur CPU modernes).

**✅ Points forts**

- **Performances absolues** : Le plus rapide de toutes les implémentations, sur toutes les métriques.
- **Efficacité CPU exceptionnelle** : 5.52s de CPU pour 5120 vues. Imbattable.
- **Empreinte mémoire minimale** : 80 MB seulement. Permet de traiter massivement en parallèle.
- **Simplicité d'appel Python** : Wrapper ctypes trivial, pas besoin de compiler des bindings complexes.
- **Pas de dépendances** : Juste stdlib C++17. Pas de OpenCV, pas de libs tierces.
- **Mono-thread optimal** : Pas de contention, pas de synchronisation. Idéal pour traiter N flux en parallèle.
- **Déploiement simple** : Un seul .so à compiler, pas de dépendances dynamiques.

**❌ Points faibles**

- **Code C++ à maintenir** : Toute modification nécessite recompilation.
- **Pas de multithreading** : Si on traite UN SEUL flux très lourd, on n'exploite pas le multi-core. Mais notre use case = N flux en parallèle.
- **Interpolation fixe** : Nearest neighbor uniquement. Pas de linear/cubic. Acceptable pour détection, pas pour qualité photographique.
- **Compilation nécessaire** : CMake, toolchain C++. Plus complexe que `pip install`.
- **Code spécialisé** : Optimisé pour notre cas d'usage précis (caméras fixes, vues statiques). Pas générique.

### Verdict

La bibliothèque C++ custom représente **l'optimisation ultime** : **256× moins de CPU que FFmpeg, 19× moins qu'OpenCV Python, 8.9× moins qu'OpenCV C++**.

**Cette implémentation est parfaite pour** :
- **Production haute performance** : Traiter des dizaines de flux simultanés avec efficacité maximale
- **Serveurs partagés** : Minimise la charge CPU totale (5.52s vs 48.90s pour OpenCV C++)
- **Environnements contraints en mémoire** : 80 MB seulement
- **Applications nécessitant performances prédictibles** : Mono-thread, pas de GC, pas de contention

**Trade-offs assumés** :
- Pas de multithreading (volontairement)
- Pas de flexibilité (interpolation fixe)
- Code spécialisé (pas de généricité OpenCV)

**Ces trade-offs sont acceptables** parce que notre use case le permet :
- On traite N flux en parallèle (pas besoin de multi-thread par flux)
- Nearest neighbor suffit pour la détection
- Caméras fixes = pas besoin de recalibration dynamique

**Code complet** : [github.com/pykoder/fisheye-dewarping/tree/main/06_ctypes_custom](lien-à-adapter)

---

## Conclusion : Choisir la bonne arme

Après avoir comparé 6 implémentations différentes du même algorithme, voici ce qu'on a appris :

### Récapitulatif des performances

**Pour 1024 frames × 5 vues (5120 images dewarpées) :**

| Implémentation | Wall Time | CPU Time | Cores | Mémoire | Speedup vs FFmpeg |
|----------------|-----------|----------|-------|---------|-------------------|
| **FFmpeg** | 208.64s | 1411.84s | 6.8 | 1784 MB | 1× (baseline) |
| **Python pur** | 1889.36s | 1889.34s | 1.0 | 647 MB | 0.11× |
| **NumPy vectorisé** | 110.11s | 113.83s | 1.0 | 256 MB | 1.9× |
| **OpenCV Python** | 22.21s | 105.70s | 4.8 | 288 MB | 9.4× |
| **OpenCV C++** | 10.09s | 48.90s | 4.8 | 110 MB | 20.7× |
| **Lib C++ custom** | **4.91s** | **5.52s** | 1.1 | **80 MB** | **42.5×** |

### Leçons apprises

**1. Le multithreading n'est pas toujours la réponse**

FFmpeg (6.8 cores) et OpenCV (4.8 cores) parallélisent agressivement... mais consomment **énormément de CPU total**. La lib custom mono-thread (1.1 core) est **256× plus efficiente en CPU**.

**Moralité** : Sur un serveur qui traite N flux en parallèle, mieux vaut N processus mono-thread efficients qu'un processus multi-thread qui monopolise tous les cores.

**2. L'overhead Python est réel mais pas dramatique**

OpenCV Python vs OpenCV C++ : facteur 2.4× en CPU, 2.6× en mémoire. Acceptable pour beaucoup d'use cases. Si Python suffit, pas besoin de passer au C++.

**3. La vectorisation NumPy a ses limites**

NumPy vectorisé = 17× plus rapide que Python pur, mais reste 4.5× plus lent qu'OpenCV Python. L'indexation avancée NumPy ne parallélise pas bien.

**4. Le code spécialisé écrase le code générique**

Lib custom vs OpenCV C++ : 8.9× moins de CPU. Pourquoi ? Parce qu'on fait **exactement ce dont on a besoin**, sans l'overhead d'une bibliothèque générique.

**5. La mémoire compte**

80 MB (custom) vs 1784 MB (FFmpeg) = **22× moins**. Sur un serveur traitant 20 flux simultanés :
- Custom : 20 × 80 MB = 1.6 GB
- FFmpeg : 20 × 1784 MB = 35 GB (impossible)

### Recommandations par use case

**Prototypage rapide / POC**
→ **FFmpeg CLI**
- Setup immédiat, aucun code
- Performances correctes
- Limitation : pas intégrable, grosse consommation RAM

**Pipeline Python existant, performances OK**
→ **OpenCV Python**
- Intégration triviale (`pip install`)
- Performances acceptables (9.4× vs FFmpeg en wall time)
- Limitation : consomme beaucoup de CPU (105.70s)

**Pipeline Python, besoin d'optimiser**
→ **Lib C++ custom via ctypes**
- Performances maximales tout en restant appelable depuis Python
- Efficacité CPU exceptionnelle
- Limitation : nécessite compilation, maintenance C++

**Application standalone, performance critique**
→ **OpenCV C++ ou lib custom**
- OpenCV C++ si besoin de flexibilité (interpolation, etc.)
- Lib custom si performance absolue requise
- Limitation : complexité de build/déploiement

**Apprentissage / compréhension**
→ **Python pur**
- Code pédagogique
- Chaque étape mathématique explicite
- Limitation : 20× trop lent pour production

### Le retour d'expérience

**Il y a trois ans**, face au memory leak de la lib propriétaire, on a codé une solution C++ en 15 jours. Ça a marché.

**Aujourd'hui**, avec le recul, qu'aurions-nous fait différemment ?

Probablement... **exactement pareil**. La lib C++ custom optimisée était le bon choix :
- Performances exceptionnelles (nécessaire pour traiter N flux)
- Consommation mémoire minimale (critique en prod)
- Pas de dépendances externes (pas de risque de nouveau memory leak)
- Code maîtrisé de bout en bout (pas de boîte noire)

FFmpeg aurait été une solution de secours acceptable en attendant mieux, mais on aurait vite été limités (RAM, intégration, contrôle).

### Et maintenant ?

Ce benchmark comparatif nous confirme qu'on a fait le bon choix technologique il y a trois ans. Mais il révèle aussi des pistes d'amélioration :

**Optimisations possibles sur la lib custom :**
- Instructions SIMD explicites (AVX2) pour le remapping
- Prefetching mémoire plus agressif
- Support GPU via CUDA (pour les très gros volumes)

**Mais** : Vu les performances actuelles (4.91s pour 5120 vues), est-ce vraiment nécessaire ? Le jus vaut-il la chandelle ?

**La vraie question** : À quel moment l'optimisation devient-elle de la sur-ingénierie ?

Pour notre use case (traitement temps réel de multiples flux), **la lib custom actuelle est largement suffisante**. Les 4.91s de wall time et 5.52s de CPU représentent moins de 1ms par vue - amplement suffisant pour du temps réel.

**Conclusion** : Avant d'optimiser, mesurez. Avant de mesurer, définissez vos contraintes. Et surtout : **la solution la plus simple qui marche est souvent la meilleure**.

---

## Remerciements

Merci à Damien pour les maths derrière l'algorithme (quaternions et projections sphériques), et à toute l'équipe de Veesion pour m'avoir supporté pendant cinq ans.

Le code complet des 6 implémentations est disponible sur GitHub : [github.com/pykoder/fisheye-dewarping](lien-à-adapter)

---

*Article écrit en décembre 2025. Les benchmarks ont été réalisés sur un Lenovo ThinkPad P14s - Ubuntu 25.04, Intel Core i7-1185G7 (4 cores physiques, 8 threads), 16GB RAM. Le CPU supportant AVX-512, les performances NumPy/OpenCV bénéficient des instructions SIMD avancées.*