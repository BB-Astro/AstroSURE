# Plan — Entraînement AstroSURE N2N

## Contexte rapide

- **986 paires N2N** prêtes dans `training_data/pairs_catalog.json`
- 260 fichiers `*_chip1.fits` / `*_chip2.fits`, 4096×2048 px float32, ~34 MB chacun
- 5 galaxies : NGC_628, NGC_1365, NGC_1559, NGC_1566, NGC_1672
- Alignement WCS à la volée (SIP, résidu 0 px vérifié par cross-corrélation)

---

## Étape 1 : Dataset PyTorch (`training/dataset_n2n.py`) — FAIT

### Logique d'un batch

```python
# Pour chaque paire (img_A, img_B) du même groupe :
# 1. Charger les deux chips (astropy.io.fits, extension SCI)
# 2. Construire le WCS (keywords D2IM/CPDIS nettoyés, SIP seul suffit)
# 3. Calculer le recouvrement en RA/DEC (footprint des 4 coins)
# 4. Tirer un (RA, DEC) aléatoire dans le recouvrement
# 5. wcs_A.all_world2pix(ra, dec) → (xA, yA)
#    wcs_B.all_world2pix(ra, dec) → (xB, yB)
# 6. Extraire patch 256×256 centré sur (xA,yA) dans A, (xB,yB) dans B
# 7. Normalisation jointe : percentile [1, 99] sur concat(A, B)
```

### Points d'attention
- Patches avec NaN → rejetés (jusqu'à 50 essais par sample)
- PATCH_SIZE = 256 (divisible par 32 → compatible U-Net 5 niveaux)
- Augmentations : flip H/V, rotation 90° (mêmes transforms sur A et B)
- **CRITIQUE** : normaliser A et B avec les MÊMES statistiques (percentile joint)
  - Run 1 normalisait A et B indépendamment → artefacts en anneaux, échec total
- Cache en RAM : ~9 GB pour les 260 fichiers (chargés au premier accès)

---

## Étape 2 : Modèle (`training/unet_model.py`) — FAIT

U-Net bilinéaire (pas de ConvTranspose → pas de checkerboard).
- **988 609 paramètres**
- 5 niveaux d'encodage (MaxPool2d) + 5 niveaux de décodage (Upsample bilinéaire)
- LeakyReLU(0.1), pas de BatchNorm
- Skip connections à chaque niveau
- Init Kaiming (He et al. 2015)

---

## Étape 3 : Entraînement (`training/train.py`) — EN COURS

```python
PATCH_SIZE    = 256
BATCH_SIZE    = 8
LEARNING_RATE = 1e-4
NUM_EPOCHS    = 50
LOSS          = MSELoss()   # N2N standard
DEVICE        = "mps"       # Apple M1 Max
```

### Split train/val
- 80% des paires → train (788 paires, 6304 samples)
- 20% → val (198 paires, 1584 samples)
- Sauvegarder checkpoints toutes les 5 époques + best.pth

### Scheduler
- ReduceLROnPlateau(factor=0.5, patience=5)

### Performance
- ~2 min/époque, ~1h40 pour 50 époques

### Historique des runs

#### Run 1 — ÉCHOUÉ (28 mars 2026)
- 50 époques complètes, PSNR final ~18.6 dB
- **Résultat catastrophique** : artefacts en anneaux autour des étoiles,
  image ressemble à un filtre passe-haut
- **Cause** : normalisation A et B indépendante. Le modèle apprenait à
  convertir entre deux espaces de normalisation au lieu de débruiter
- **Fix** : `_normalize_pair()` applique les mêmes percentiles aux deux patches

#### Run 2 — ÉCHOUÉ (4 août 2026)
- Relancé après correction des chemins absolus du catalogue (projet déménagé
  vers le SideCar le 28 mars : `pairs_catalog.json` pointait vers l'ancien
  emplacement, d'où le FileNotFoundError du run avorté de mars)
- 50 époques complètes, val MSE 0.0172 (= plancher du bruit N2N, entraînement sain)
- **Inférence catastrophique** : offset fond -107 e-, bruit ÷0.32 (amplifié !),
  photométrie étoiles 19 %
- **Trois causes racines identifiées** (batterie de tests patch/pleine image) :
  1. Paires N2N incohérentes : 54 % mélangent des EXPTIME différents (360 vs
     590 s), ciel variable jusqu'à ×4 dans un groupe (NGC 1559 F606W : 76 →
     314 e-). Le modèle apprenait une conversion photométrique, pas un débruitage.
  2. `_normalize_pair` écrêtait tout à [0,1] : cœurs d'étoiles et rayons
     cosmiques aplatis à 1.0 à l'entraînement, valeurs à 600 vues à l'inférence.
  3. Loss MSE : les rayons cosmiques de la cible (~2 % des pixels à 1200 s)
     biaisent le prédicteur moyen.

#### Run 3 — ÉCHEC INSTRUCTIF (5 août 2026, nuit)
- Correctifs : e-/s (÷EXPTIME) + soustraction du ciel par exposition au
  chargement, normalisation d'échelle seule sans écrêtage, symétrisation
  aléatoire des paires, loss L1, inférence en miroir du prétraitement
- Offset corrigé (-0.7 e-), champ lointain du résidu propre (-0.4σ),
  bruit fond ÷3.7... mais **étoiles effacées à 99.5 %**
- Cause : alignement au pixel entier (`int(round)`) avec des dithers HST
  volontairement demi-pixel + 96 paires avec rotation relative ~174°
  (NGC 628, proposals croisés). Une PSF sous-échantillonnée décalée de
  0.5 px entre entrée et cible est incohérente : le prédicteur médian
  la traite comme un rayon cosmique et l'efface.

#### Run 4 — ÉCHEC INSTRUCTIF (5 août 2026, nuit)
- Paires tournées (>0.1° de roulis relatif) exclues : 890 paires restantes
- Alignement subpixel : la cible est rééchantillonnée (bilinéaire) sur la
  grille exacte de l'entrée ; l'entrée reste brute (bruit intact).
  Validé par centroïdes d'étoiles : décalage médian 0.14 px, sans biais
- Résultat identique au run 3 (offset -0.5 e-, bruit ÷3.7, étoiles 0.4 %) :
  l'hypothèse du désalignement est réfutée
- **Sonde synthétique décisive** (patch de bruit + source injectée) :
  étoiles PSF faibles (< ~20σ) préservées à 95-97 %, rayons cosmiques
  effacés à tous les niveaux, sources étendues intactes à 100 %,
  MAIS étoiles PSF brillantes effacées (41 % à 45σ, 12 % à 135σ).
  Le modèle discrimine par la forme en régime faible ; en régime fort,
  les valeurs 30+ écrasent les features (LeakyReLU sans normalisation)
  et l'information de forme disparaît : tout pic devient un rayon cosmique.

#### Run 5 — PREMIER SUCCÈS PARTIEL (5 août 2026, ~7h15)
- Compression de dynamique asinh dans le prétraitement (ASINH_BETA = 1.0
  unité d'échelle ≈ 4.5σ du fond) : linéaire sous le genou, log au-dessus,
  inversée (sinh) à la dénormalisation de l'inférence
- Même recette sinon (alignement subpixel, paires filtrées, L1, e-/s,
  ciel soustrait)
- **Résultats (NGC 628 chip pleine image)** : fond ÷4.00, offset -0.75 e-,
  rayons cosmiques effacés à 100 % (n=4477), étoiles 10-30σ à **91.7 %**
  (n=1742, contre 0.4 % au run 3-4). Planche : `compare_NGC628/planche_run5.png`
- **Limite restante** : étoiles > ~30σ toujours effacées (0.3 %). La sonde
  montre un seuil d'amplitude appris : dans ces champs extragalactiques,
  les sources ponctuelles brillantes de l'entraînement sont presque toutes
  des rayons cosmiques, le prédicteur médian est statistiquement rationnel.
  Ce n'est plus un bug, c'est la donnée.

#### Run 6 — ÉCHEC INSTRUCTIF MAJEUR (5 août 2026, midi)
- Entraînement sur les 260 chips nettoyés par deepCR ACS-WFC (batch validé :
  vraies étoiles 100.0 % de survie par confirmation partenaire, CR à 1-4 %)
- Résultat : fond ÷2.8-3.0 (MOINS bon que run 5), étoiles 93-97 %
- **Découverte capitale en refaisant le QC avec une sélection d'étoiles
  PROPRE (chip nettoyé, pureté 99.9 %)** : le run 5 préservait déjà les
  vraies étoiles à 95-99 % À TOUS LES ÉTAGES. Le "massacre des étoiles
  brillantes" (0.3-0.4 %) qui a motivé les runs 4 et 6 était un artefact de
  mesure : sur un FLC brut, 97 % des pics brillants "PSF-like" sont des
  rayons cosmiques, et la métrique mesurait leur réjection (souhaitable),
  pas une destruction d'étoiles. La sonde synthétique (gaussienne sans
  ailes de PSF) a renforcé l'illusion.
- Matrice complète (NGC 628, sélection propre) :
  run 5 sur brut ÷3.79 / 95-99 % ; run 5 sur nettoyé ÷3.75 / 95-97 % ;
  run 6 sur brut ÷2.82 / 94-97 % ; run 6 sur nettoyé ÷2.99 / 93-96 %
- Verdict : **le nettoyage des données d'entraînement n'apporte rien et
  coûte du pouvoir débruitant** (cibles inpaintées localement sans bruit =
  légère invitation au flou). Le run 5 reste le modèle de référence, et il
  élimine les rayons cosmiques gratuitement.

#### Run 7 — TERMINÉ, N'AMÉLIORE PAS (5 août 2026, après-midi)
- Recette run 5 inchangée, sur le dataset BRUT élargi : 69 groupes
  (58 anciens + 12 nouveaux - NGC 4258 chip2 exclu pour gradient WCS),
  366 chips, ~1350 paires, corrections WCS par exposition actives
  (training_data/wcs_offsets.json, résidu M33/IC10 : 0.068 px médian)
- Résultats (sélection propre) : NGC 628 fond ÷3.59 et étoiles 95-96 %
  (run 5 : ÷3.79 et 95-99 %) ; M33 dense : offset +1.4 e- (mieux que les
  +5.7 e- du run 5) mais étoiles 96-97 % (run 5 : 97-101 %)
- Lecture : le rééquilibrage du dataset vers les champs denses corrige
  l'offset en champ dense mais dilue les patchs de type "galaxie sur fond
  vide" et coûte du pouvoir débruitant là où il compte. Sur champ dense,
  les deux modèles préservent (à raison) le tapis stellaire non résolu au
  lieu de le lisser (÷1.1-1.3).

### VERDICT DU 5 AOÛT : le modèle de référence est le RUN 5
(`training/checkpoints_run5/best.pth`). Le run 7 (`checkpoints_run7/`) est
l'alternative "champ dense" si l'offset y devient gênant. Le dataset élargi
et validé reste un acquis pour la suite (échantillonnage équilibré par type
de champ, entraînement plus long, M87 en réserve).

#### Validation bout-en-bout sur Arp 70 (5 août 2026, soir)
- FLC du programme 15446 (F606W, 2×390 s) dans `~/DocM2max/Astro/Arp070_MAST/`
- `process_flc_asure.py` : AstroSURE run 5 par chip + StripeField (`--angles 0`),
  réassemblage FLC complet → `*_flc_asure.fits` (ERR/DQ/WCS intacts,
  ERR non remis à l'échelle)
- **Vérité terrain Gaia DR3 (20 étoiles G=14.5-21.2) : survie 100.2 %** ;
  rayons cosmiques 99.9 % éliminés (14 000/chip) ; striping 0.64→0.11 e-
  par AstroSURE seul (0.08 après StripeField) ; fond ÷24 ; galaxies intactes
- ⚠️ Double piège de mesure sur champ à >10⁴ CR/chip : la confirmation par
  partenaire (même avec fenêtre de flux 0.5-2) est dominée par les
  coïncidences CR-CR ("0.4 % de survie" mesuré, 100 % réel). DeepCR aussi
  s'y est fait piéger (5.2 % apparent). Gaia obligatoire dans ce régime.
- Limite VISUELLE constatée par BB : lissage des détails fins de la galaxie.
  Cause dominante : plafond d'information d'une pose unique de 390 s
  (l'estimateur médian lisse ce qui est près du bruit) + léger flou de
  recette (cible bilinéaire). Planches : `drizzle_vs_astrosure_tel.png`
  (vs chaîne MAST classique), combine dans `Arp070_asure_combine_chip2.fits`
- **Drizzle final réel** (moteur STScI drizzle 2.2.0, pixfrac 0.8, grille
  DRC MAST, distorsion complète) : `Arp070_asure_drizzled.fits`. À stretch
  absolu identique, la chaîne AstroSURE→drizzle bat la chaîne classique :
  fond haute fréquence ÷15.8, basse fréquence ÷2.6, étoiles Gaia intactes.
  Le "bruit basse fréquence" vu par BB sur les planches précédentes était
  un artefact d'affichage (autostretch par panneau, ~24× plus agressif sur
  le panneau au sigma effondré). Leçon planches : toujours comparer à
  fenêtre de stretch ABSOLUE identique. Reste ouvert : détails fins → run 8.

## ORDRE CANONIQUE DU PIPELINE DE PRODUCTION
## (v2 validée par BB le 10 août, « cette séquence est très bien »)

1. FLC bruts MAST (grille détecteur, électrons, CR et striping présents)
2. **deepCR** par chip (`pipeline/deepcr_flc.py`, veto flux partenaire
   par clump) → `*_flc_dcr.fits`
3. **StripeField** par chip, angle 0 en repère détecteur
   (`pipeline/destripe_flc.py`, moteur destripe_astro de BB)
   → `*_flc_dcr_dsf.fits`
4. **AstroSURE** par chip et par pose (`pipeline/asure2_flc.py`,
   run 9 e20, médiane masquée DQ) → `*_flc_dcr_dsf_asure9e20b.fits`
5. **Drizzle** (moteur STScI, grille DRC, masque DQ serré) du débruité
   ET du non-débruité (donneur de bruit) → `*_dcr_dsf_asure9e20b_drizzled`
   + `*_dcr_dsf_drizzled`
6. **Blend α=0.75** (choix BB) : `0.75*débruité + 0.25*donneur`
   → `*_dsf_blend075_drizzled.fits` (bruit ÷3.4)
7. Post-production PixInsight (DBE…) ; les passes DeepCR et StripeField
   aval de BB deviennent inutiles

Chaque correction instrumentale se fait dans SON domaine : CR et stripes
en repère détecteur sur FLC brut (statistiques intactes), débruitage
avant drizzle (le rééchantillonnage corrèle le bruit et sort le modèle
de son domaine, cause du "moucheté" du 5 août). L'ancien ordre v1
(AstroSURE directement sur FLC, StripeField en finition aval) est
remplacé : StripeField aval ne voit plus les zones claires (masque MAD
+ Wiener → zéro hors mesure) une fois le bruit effondré.

#### Validation n°2 : Arp 130 (6 août 2026, pipeline générique)
- Outils réutilisables créés : `pipeline/asure_flc.py` (AstroSURE sur FLC
  complets, multi-chips) et `pipeline/drizzle_asure.py` (drizzle STScI sur
  grille DRC de référence). Bout en bout en ~4 min sur M2 Max.
- Arp 130 (même programme 15446, 2×390 s F606W, `~/DocM2max/Astro/Arp130_MAST/`) :
  fond haute fréquence **÷11.6**, étoiles Gaia **99.9 %** (n=16) ; basse
  fréquence ÷1.0 = structure réelle du champ, identique dans les deux
  chaînes. Généralisation confirmée. Arp 141 et Arp 204 (même programme)
  à une commande de distance.

## CHANTIER RUN 8 (spécifié le 5 août, lancé le 5 août : implémentation ci-dessous)

Objectif : récupérer du détail fin sans sacrifier la photométrie. Deux
changements, aucune nouvelle donnée requise :

1. **Cibles en pile** : pour une entrée A d'un groupe à N expositions,
   cible = moyenne des N-1 autres, chacune alignée sur la grille de A par
   la mécanique subpixel existante (`_extract_pair` + `wcs_offsets.json`).
   Réduit le bruit de cible d'un facteur √(N-1), le lissage bilinéaire se
   moyenne, et le plancher de loss chute. Attention : rayons cosmiques des
   cibles dilués à 1/(N-1) mais plus nombreux ; garder L1, ou masquer par
   sigma-clip contre la pile.
2. **Entrée à 2 canaux** : `UNet(in_channels=2)`, entrée = paire
   d'expositions alignées (la 2e rééchantillonnée sur la grille de la 1re),
   cible = pile des autres. Le modèle voit 2× plus de photons : le
   compromis bruit/détail recule mécaniquement. Inférence : nouveau
   wrapper prenant deux FLC (cas d'usage réel de BB, toujours ≥2 poses).
   Généralisation : canal 2 = moyenne des poses restantes si N>2.
3. **Dosage aval avec masque CR** (idée BB, "à essayer") : sur FLC,
   `final = débruité + k×(source − débruité)×(1 − masque_CR)` avec
   masque_CR = |source − débruité| > 5σ dilaté 1 px. Rend la texture sans
   réinjecter les impacts (un dosage nu à 30 % réinjecterait 30 % de
   chaque impact, objection de BB fondée).

### RUN 8 : implémentation (lancé le 5 août, points 1+2 combinés)

- `training/dataset_n2n_stack.py` : `N2NStackDataset` (hérite de
  `N2NDataset`, réutilise cache, offsets WCS, extraction subpixel).
  Unité = fichiers d'un groupe au même angle de roulis, **N ≥ 3 exigé**
  (2 entrées + ≥ 1 cible) : 43 unités (3 à 20 poses), 34 amas < 3 exclus.
  Cible = moyenne des partenaires (plafonnée à 12) avec **réjection CR
  contre la pile** avant moyenne (un CR dilué dans une moyenne devient un
  petit biais que la L1 suit, contrairement à un gros outlier isolé) :
  M ≥ 3 → clip 5σ contre la médiane (outliers positifs seulement) ;
  M = 2 → min des deux là où ils divergent > 5σ. Normalisation jointe
  (entrées + cible), asinh β=1.0 inchangé.
- Catalogue : `pairs_catalog_full_raw.json` (le run 5 n'a que 30 groupes
  N ≥ 3 ; l'élargi en a 41) avec **échantillonnage équilibré par unité**
  (même nombre de tirages qu'une unité ait 3 ou 20 poses), la piste notée
  à la clôture du run 7 contre la domination des champs denses.
- `training/train.py` : `--mode stack2`, `--ckpt-dir` (garde-fou : refuse
  un répertoire symlink, `checkpoints` pointe sur le champion run 5),
  `--samples-per-unit 128`. Split train/val au niveau des unités (43 → 34+9).
- Hygiène QC : le chip de référence NGC 628 (visite j96r21, groupes de 2)
  est **hors entraînement run 8** (exclu par N ≥ 3), alors que le run 5
  l'a vu à l'entraînement : comparaison out-of-sample favorable à la rigueur.
- `training/infer2.py` : inférence 2 canaux (réf brute + partenaire aligné),
  échelle jointe sur les deux canaux, même split vertical.
- `pipeline/asure2_flc.py` : production sur FLC complets ≥ 2 poses ; canal 2
  = moyenne des autres poses rééchantillonnées par WCS à distorsion complète,
  avec **contrôle du résidu par corrélation de phase** (suréchantillonnage
  DFT local, basses fréquences seules ; testé à 0.04 px près ; correction
  appliquée si 0.05 < |s| < 2 px). Sorties `*_asure8.fits`.
- `training/qc_run8_ngc628.py` : QC protocolaire (étoiles sur chip nettoyé,
  étages 10-30/30-100/100-500σ) + **score de détail fin** = corrélation
  passe-haut avec l'exposition partenaire dans le masque galaxie (le bruit
  des deux images étant indépendant, seule la structure réelle corrèle).
- Entraînement : 50 époques, ~170 s/époque M2 Max, log `train_run8.log`,
  checkpoints `training/checkpoints_run8/`.

### RUN 8 : RÉSULTATS (5 août soir) — VALIDÉ, production FLC par paires

QC NGC 628 F814W j96r21je chip1 (hors entraînement pour le run 8, vu à
l'entraînement par le run 5 ; sélection d'étoiles sur chip nettoyé) :

| Métrique | run 5 | run 8 |
|---|---|---|
| Fond | ÷3.79 | ÷3.16 |
| Étoiles 10-30σ | 98.4 % | 100.9 % |
| Étoiles 30-100σ | 99.6 % | 99.6 % |
| Étoiles 100-500σ | 99.7 % | 98.1 % |
| Détail fin (corr. HF avec partenaire) | 0.775 | **0.815** |

Nota métrologie : le score de détail chip-level corrèle avec l'exposition
partenaire, qui est AUSSI le canal 2 du run 8 (juge non indépendant,
biais possible en faveur du run 8) ; la validation drizzlée ci-dessous,
jugée contre la chaîne MAST, confirme l'avantage sans ce biais.

Validation bout-en-bout Arp 70 (asure2_flc → drizzle_asure, grille DRC,
mêmes masques, stretch absolu) :

| Métrique (vs chaîne MAST classique) | run 5 | run 8 |
|---|---|---|
| Fond HF | ÷39.9 | ÷35.9 |
| Fond BF | ÷1.77 | ÷1.49 |
| Gaia 21 étoiles, médiane | 103.6 % | 98.2 % |
| Gaia p10 / p90 | 92.0 / 114.2 % | **96.9 / 100.1 %** |
| Détail galaxie (corr. HF avec source) | 0.793 | **0.856** |
| Amplitude HF conservée dans la galaxie | 82 % | 84 % |

Visuel (arp70_run8_vs_run5_tel.png) : bras spiraux et nœuds nettement
plus texturés, fini le rendu "cire" du run 5 ; fond quasi identique.
VERDICT : run 8 (`training/checkpoints_run8/best.pth`, 2 canaux) devient
la voie production pour les FLC par paires (cas standard, toujours ≥ 2
poses) via `pipeline/asure2_flc.py`. Le run 5 reste champion 1 canal
(`training/checkpoints` inchangé) pour chip isolé et images drizzlées
(`infer_simple.py`). Reste "à essayer" du chantier : le dosage aval à
masque CR (point 3), non nécessaire à ce stade.

### RUN 9 (6 août) : dataset élargi + sélection par QC — **CHAMPION (époque 20)**

Dataset étendu par agent (110 poses, 18.6 Go, 9 champs galaxie : M51,
NGC 4921/4911, NGC 4639, NGC 3314, NGC 1316, NGC 1448, M87, NGC 3370 ;
43 → 73 unités, offsets WCS 116 → 318 entrées, résidu médian 0.010 px,
catalogue `pairs_catalog_run9.json`). Recette run 8 inchangée, 100
époques (~7h45 M2 Max, `checkpoints_run9/`).

Épisode en deux temps, riche d'enseignement :
1. **Sélectionné par val loss (best.pth = époque ~100), le run 9 perdait**
   contre le run 8 : détail 0.819 vs 0.856 sur Arp 70, Gaia 97.4 vs
   98.2 %, malgré un fond ÷53.5. Avec des cibles en pile très calmes, la
   val loss vote mécaniquement pour l'époque la plus lisse.
2. **Balayage QC des 20 checkpoints** (sonde NGC 628 sur chaque
   epoch_NNN.pth) : le compromis détail/lissage culmine à l'ÉPOQUE 20
   puis se dégrade. Validé en production sur Arp 70 :

| Arp 70 (vs MAST) | run 5 | run 8 | run 9 @100 | **run 9 @20** |
|---|---|---|---|---|
| Détail galaxie (corr. HF) | 0.793 | 0.856 | 0.819 | **0.890** |
| Amplitude HF conservée | 82 % | 84 % | 80 % | **86 %** |
| Gaia médiane | 103.6 % | 98.2 % | 97.4 % | **100.4 %** |
| Gaia p10/p90 | 92/114 | 96.9/100.1 | 95.6/100.2 | 94.1/103.0 |
| Fond HF | ÷39.9 | ÷35.9 | ÷53.5 | ÷32.0 |

VERDICT : **champion = run 9 époque 20**, promu
`checkpoints_run9/best_qc.pth` (modèle par défaut de
`pipeline/asure2_flc.py` ; `best.pth` = choix val-loss, conservé).
Les données du run 9 étaient bonnes, c'est le critère de sélection qui
gâchait leur apport. "La val loss ne dit rien de la qualité", épisode
2 : desormais, **sélection de checkpoint par QC obligatoire** (balayage
`sweep_run9.log`, script du sweep reproductible en une boucle sur
`qc_run8_ngc628.py <ckpt>`). Acquis annexes :
`download_mast/download_robust.py` (téléchargeur MAST à reprise Range +
détecteur de décrochage).

#### Épisode PixInsight (6 août) : trois défauts de la chaîne drizzle corrigés

BB inspecte les produits dans PixInsight : "vraiment pas beau". L'autostretch
souffrait, mais parce que les images contenaient de VRAIS défauts, invisibles
sur les zooms galaxie et flagrants en plein champ à stretch dur :

1. **Sous-débruitage des chips à galaxie brillante** : l'échelle
   d'inférence p99-p1 plein chip est gonflée par la galaxie (2.2× sur
   Arp 141 SCI2) → le fond passe sous le point de travail du réseau
   (σ ÷3 au lieu de ÷12), les CR compressés ressemblent à des étoiles
   faibles et sont gardés (moucheté d'un demi-champ). Correctif
   `infer2.py` : échelle = médiane des p99-p1 par tuiles 256×256, la
   statistique de l'entraînement.
2. **Couture entre chips** : ciel soustrait = médiane brute par chip,
   biaisée par la galaxie → marche visible entre les empreintes.
   Correctif `drizzle_asure.py` : ciel sigma-clippé (équivalent
   skymatch). Marche résiduelle ramenée au niveau du DRC MAST
   (halo réel de la galaxie).
3. **Pas de masquage DQ** : colonnes mortes, pixels chauds, amas de
   pixels NÉGATIFS (mesuré -79 e- sur ciel à 44, DQ=4096+16) passaient
   dans le produit ; pire, le U-Net étale un tel amas en traînée sombre
   PLUS LARGE que l'empreinte DQ (226 px sombres → le masquage drizzle
   seul ne suffit pas). Correctifs : poids drizzle = 0 hors
   DQ_GOOD = 16|64|256|2048 (les cœurs saturés restent), ET
   pré-nettoyage des pixels DQ-mauvais AVANT inférence (médiane locale
   5×5) dans `asure2_flc.py`. Traînées sombres : 0 après correctif.

Deuxième salve (capture PixInsight de BB sur Arp 130, "ligne de bruit
centrale") :

4. **Bande du gap inter-chips** : les 2 CCD ACS sont séparés de ~50 px ;
   avec le dither 2 points de ~3", la bande n'est couverte que par UNE
   pose. Le repli "canal 2 = canal 1" y faisait garder les CR par le
   modèle 2 canaux (toute source y est "confirmée"), et les flags DQ n'y
   existent pas (posés par comparaison de poses). Correctif : la bande
   est comblée par le **run 5 mono-canal** (élimine les CR sans
   partenaire, 99.9 % validé), fondu gaussien sur ~16 px
   (`asure2_flc.py` + `infer_simple.py --tile-scale`).
5. **Ciel par exposition au drizzle** (pool des 2 chips, un seul
   constant) au lieu d'un ciel par chip : une marche entre chips devient
   impossible par construction.
6. **Ré-ancrage du fond débruité** : le modèle décale le fond d'un
   offset dépendant du contenu (+1.4 e- sur le chip à galaxie d'Arp 130
   contre +0.15 e- sur le chip vide = la marche visible). Mesure du
   offset sur l'IMAGE DIFFÉRENCE brut−débruité (le halo s'y annule pixel
   à pixel, les CR sont clippés). Piège d'estimateur documenté :
   clipped_sky(brut) − clipped_sky(débruité) ne marche PAS, le clip à 3σ
   garde le halo sur le brut bruité (σ=14 e-) et le rejette sur le
   débruité lisse (σ=0.5 e-), la différence contient alors le halo
   (sur-correction de +1.4 e- observée).

Épilogue : après ces correctifs, à STRETCH ABSOLU IDENTIQUE (fenêtre du
DRC), notre produit et le DRC MAST ont les mêmes niveaux gauche/droite
(marche +0.15σ/−0.05σ identique) ; la "zone très différente" restante
dans PixInsight est l'autostretch, ~20× plus agressif sur une image au σ
effondré (même artefact d'affichage que la planche alignée du 5 août).

Leçons : (a) valider un produit d'imagerie PLEIN CHAMP à stretch dur ET
à stretch absolu identique au produit de référence ; les défauts étaient
hors des masques du QC quantitatif. (b) L'œil de BB dans PixInsight est
un instrument de QC : six défauts réels trouvés en une session.

#### Deux pièges de production trouvés et corrigés en validation
1. **Bande de dither sans partenaire** : avec un dither de ~60 px, ~3 %
   du chip n'est couvert que par une pose ; le canal 2 y contenait du
   remplissage de bord → "présent canal 1, absent canal 2" = signature
   CR → 3 étoiles Gaia effacées à 0-4 %. Correctif : hors couverture
   partenaire, canal 2 = canal 1 (source "confirmée", conservée ; cette
   bande garde ses CR, comme la chaîne MAST, physique honnête).
2. **Corrélation de phase verrouillée sur le détecteur** : striping,
   pixels chauds et colonnes communs aux deux poses en coordonnées
   détecteur créent un faux pic à exactement −dither (|s| = 58-62 px
   mesuré sur un alignement pourtant Gaia-parfait). Correctifs : recherche
   locale ±3 px seulement, médiane par ligne soustraite ; et sur champ
   pauvre à fort taux de CR la mesure reste fragile → le raffinement est
   désormais **opt-in** (`--refine`), le WCS fait foi par défaut.

#### Faux départ instructif (5 août, premier lancement, archivé `_clipbias`)
La première version de la réjection CR clippait les outliers positifs de
la pile à 5σ du FOND. Au cœur d'une étoile brillante, Poisson + résidus
subpixel dépassent largement 5σ_fond : le clip coupait les fluctuations
positives et gardait les négatives → biais de flux systématique de la
CIBLE (mesuré −4.2 % médian, −9 % p10 sur les pixels > 50σ), que le
modèle a fidèlement appris (sonde époque 30 : étoiles 100-500σ à 91.7 %,
fond retombé à ÷3.06). Correctif validé : seuil = 5σ_fond + 0.5×signal
(brillants 100.1 %, faibles 99.8 %, réjection CR intacte à −0.15σ).
Run relancé de zéro. Leçon : tout seuil de réjection sur une pile
d'images doit avoir un terme proportionnel au signal, sinon il devient
un compresseur de sources brillantes. Sondes QC intermédiaires (époques
10/30) rentabilisées : bug attrapé à mi-course, pas après 50 époques.

#### Chaîne de production v2 (10 août 2026) : traînée sombre élucidée, deepCR en amont

BB signale sur Arp 141 (PixInsight) une traînée sombre absente des SCI
MAST, et des résidus CR que deepCR ne rattrape plus APRÈS AstroSURE.
Diagnostic et correctifs, validés sur Arp 141 + Arp 70 :

1. **Traînée sombre = double colonne morte** du détecteur (SCI2
   x=1456-1457, DQ bit 4, −45/−50 e-, présente dans les DEUX poses ;
   le drizzle tourne la colonne en ligne horizontale, et le dither de
   63 px lui est parallèle → le partenaire ne peut pas combler).
   Cascade : la médiane 5×5 du pré-nettoyage est contaminée (10/25
   pixels morts dans la fenêtre) → réparation biaisée de −2 e- (p37 des
   bons pixels au lieu de p50) → le U-Net étale le sillon en vallée de
   ~20 px → le masque DQ du drizzle (2 px) laisse passer. Correctif :
   **médiane masquée** (`masked_median_fill` dans `asure2_flc.py`,
   remplissage itératif pour les amas plus grands que la fenêtre).
   Traînée : −0.0049 → −0.0014 e-/s (plat, niveau DRC). Photométrie
   Gaia inchangée (23 étoiles, écart médian −0.03 pt).
2. **Dilatation du masque CR au drizzle : REJETÉE après mesure.** Le
   SVM flagge des cœurs de PSF sous-échantillonnés comme CR dans les
   DEUX poses ; un masque dilaté évide alors l'étoile (mesuré : une
   étoile Gaia G=19.74 à 65 % au lieu de 90 %). Masque serré conservé.
3. **deepCR AVANT AstroSURE** (`pipeline/deepcr_flc.py`, nouveau) :
   masque ACS-WFC (10 s/chip CPU), seuil 0.5, réparation par médiane
   masquée, PUIS asure2_flc sur les FLC nettoyés. Deux pièges mesurés
   en route : (a) deepCR flagge les cœurs brillants (prob 0.84-0.90 sur
   une G=17.3, −12.6 pt à travers la chaîne) ; (b) le veto par
   probabilité partenaire rate les cœurs flaggés dans UNE seule pose
   (centrage subpixel, −14.7 pt sur une G=18.69). Solution : **veto par
   FLUX partenaire au niveau du clump** (composante connexe du masque :
   somme des flux partenaire > 0.3× somme locale ET > 3σ√n). Par pixel
   ça ne marche PAS (le bruit positif du ciel "confirme" les franges
   des CR : 80-90 % de faux vetos). Les CR sur halo brillant vetoés
   restent à la charge du modèle 2 canaux (non confirmés en canal 2),
   division du travail propre : deepCR ne traite que ce que la logique
   de paire ne peut pas voir (CR mono-pose, gap inter-chips, marges de
   dither).

QC (zone fond 2000×800 d'Arp 141, seuils ABSOLUS, leçon de
l'autostretch) :

| Pics | DRC MAST | asure9e20b | deepCR+asure9e20b |
|---|---|---|---|
| >+0.07 e-/s | 687 | 57 | 50 |
| >+0.03 e-/s | 16025 | 219 | 151 |
| >+0.015 e-/s | 96937 | 869 | 482 |

Gaia Arp 70 (23 étoiles, ratio ouverture r=4 vs drizzle brut, protocole
différent du QC historique donc comparer entre colonnes seulement) :
médiane 97.4 % identique avec et sans deepCR, pire écart −0.68 pt.
Planche : `Arp141_MAST/arp141_chainfix_tel.png`. Produits de
validation : `Arp{070,141}_asure9e20b_drizzled.fits` (chaîne corrigée
seule) et `Arp{070,141}_dcr_asure9e20b_drizzled.fits` (avec deepCR).
deepCR 0.3.1 + astroscrappy 1.3.0 désormais installés dans l'env dip.

**Batch v2 relancé le 10 août (mandat BB)** : `batch_arp.py` passe par
deepcr_flc → asure2_flc → drizzle, sortie
`ArpNNN_dcr_asure9e20b_drizzled.fits`, 12/12 champs OK
(`pipeline/batch_arp_v2.log`), copiés dans `ArpBatchAout2026/` à côté
des `_asure9e20` (conservés) et des `_raw_drizzled`. Validation sur les
12 : profil PERPENDICULAIRE à la ligne de la colonne morte (piège : la
ligne mappée n'est pas horizontale sur tous les champs, l'angle varie
de −170° à +175° selon l'orientation du drizzle ; un profil en dy
vertical mesure le long de la ligne sur Arp221 et donne n'importe
quoi). Résultat : résidu nouveau ≤ ancien sur 12/12, et ≤ résidu DRC
MAST au même endroit partout sauf Arp176 (−0.0037 vs −0.0026, dans le
bruit de structure de ce champ brillant). Pire ancienne traînée du
lot : Arp221 à −0.0071, ramenée à −0.0011. Pics résiduels (crop
central 2400², seuil absolu +0.03) en baisse sur 12/12.

Leçon : tout remplissage par médiane DOIT exclure les pixels qu'il
répare de sa propre fenêtre, sinon il devient un propagateur de défaut
(même famille que le clip 5σ_fond du faux départ : un estimateur
contaminé par ce qu'il corrige). Et toute règle de veto étoile/CR doit
être testée au niveau de l'OBJET (clump), pas du pixel.

**StripeField en amont (10 août, retour BB « stripefield marche moins
bien après AstroSURE, surtout zones claires »)** : cause comprise en
lisant `destripe_astro.py` (moteur StripeField de BB) : le profil de
stripes est estimé sur les pixels de FOND (masque MAD 3σ) et le
shrinkage de Wiener ramène à zéro tout profil non mesuré → sur une
image débruitée, les stripes résiduels sont modulés par le contenu
(effacement AstroSURE non uniforme) et les zones claires sont exclues
de la mesure = pas de correction là où le résidu se voit. Correctif :
`pipeline/destripe_flc.py` appelle le moteur de BB (import depuis
ModulePixinsightByBB, source unique) sur chaque chip FLC AVANT
AstroSURE, angle forcé à 0 (repère détecteur), après deepCR (lignes
sans CR). RMS du champ de stripes : 0.44-0.47 e- par chip (Arp 164).
Validation avec l'angle_scan de BB sur le blend065 final : pic à
+68.5° (direction lignes détecteur mappée) à 6.6σ sans destripe amont,
**2.2σ avec** (sous le seuil d'arrêt de 5σ du moteur), score ÷4.6.
Planche `arp164_destripe_tel.png`. Le donneur de bruit du blend doit
être destripé aussi (`Arp164_dcr_dsf_drizzled.fits`), sinon (1−α)
réinjecte les stripes. Ordre canonique de la chaîne v2 complète :
**deepCR → StripeField → AstroSURE → drizzle → blend**.

**Investigation striping basse fréquence (10 août, retour BB)** :
BB voit des bandes LF résiduelles dans les produits finaux. Mesures sur
Arp 141 (bande passante 25-400 px, sources masquées, zone de fond) :

| Produit | RMS LF (e-/s) | plancher bruit blanc |
|---|---|---|
| donneur dcr_dsf (sans asure) | 0.00092 | 0.00028 |
| asure pur | 0.00067 | 0.00002 |
| blend075 | 0.00065 | 0.00008 |
| **DRC MAST** | **0.00096** | 0.00026 |

Conclusions : (1) la structure LF est DANS les données, le DRC MAST en
a autant ; AstroSURE la réduit légèrement et ne la crée pas ; elle
devient visible parce que le bruit pixel chute de 0.012 à 0.0035.
(2) L'amplitude ≈ 1 % du fond de ciel = ordre du résidu de L-flat ACS.
(3) Il existe en plus une composante directionnelle alignée lignes
détecteur (excès à 11σ au scan d'angles LF) ; contributeur plausible :
la bande du gap inter-chips et les coutures de couverture (alignées
lignes), à confirmer. (4) Tentatives de correction LF au niveau FLC
ABANDONNÉES et retirées de `destripe_flc.py` : tout estimateur de
profil ligne testé (médiane par ligne, plancher par blocs de colonnes)
mesure surtout le CIEL (profils corrélés à 91-96 % entre les deux
poses au lag du dither sur le chip galaxie) ; avec deux poses
seulement, la composante symétrique d'un banding LF est
mathématiquement indiscernable du ciel le long de l'axe transverse.
(5) Test « motif statique détecteur » (corrélations inter-champs)
non concluant, estimateurs 2D trop instables aux grandes échelles.

Pistes restantes, par ordre de promesse : (a) super-sky-flat du
programme : médiane en repère détecteur des 24 poses normalisées et
masquées (les 12 ciels différents décorrèlent le ciel, PAS les galaxies
SNAP toujours centrées pareil → trou central sans donnée, acceptable) ;
si le motif est statique, correction propre des 12 champs d'un coup.
(b) Traitement aval type DBE (mais les échelles 25-100 px = 1-4″ sont
sous l'échantillonnage DBE typique). (c) Accepter : 0.00065 e-/s
≈ 0.2σ du bruit du blend075, visible seulement à stretch extrême.
Décision BB (10 août) : piste (c) pour l'instant, il traitera le LF
lui-même par une passe aval dédiée.

**Batch v3 final (10 août soir, mandat BB)** : les 12 champs
régénérés avec la séquence canonique complète (deepCR → StripeField HF
→ AstroSURE run 9 e20 → drizzle ×2 → blend 0.75), 12/12 OK
(`pipeline/batch_arp_v3.log`). QC : bruit ÷3.3-3.7 partout, NaN
inchangés, colonne morte ≤ niveau DRC sur 12/12. Copiés dans
`ArpBatchAout2026/` : `ArpNNN_dsf_blend075_drizzled.fits` (final),
`ArpNNN_dcr_dsf_asure9e20b_drizzled.fits` (asure pur) et
`ArpNNN_dcr_dsf_drizzled.fits` (donneur de bruit) pour réglage α en
PixelMath par champ.

**Revue Codex traitée (11 août, `codexreview.md`)** : triage vérifié
sur pièces. Le P0 (« crash 139 au chargement des checkpoints MPS ») ne
se reproduit PAS sur la machine de production (chargement direct OK,
batch v3 complet la veille) : artefact de la sandbox d'audit sans accès
Metal. MAIS le correctif est retenu comme bug de PORTABILITÉ réel
(checkpoints taggés MPS illisibles sur Linux/CUDA et RunPod) :
`unet_model.py` charge désormais en `map_location="cpu"` puis
l'appelant déplace (state dicts vérifiés identiques bit à bit, forward
CPU/MPS à 7e-7 près). Corrigés aussi, tous vérifiés par test : garde
2-poses-minimum + garde couverture partenaire dans `deepcr_flc.py`
(avec 1 pose, sig=NaN et le veto mourait en silence, reproduit),
exit code non nul du batch incomplet (reproduit), `clipped_sky` sur
entrée constante/vide (NaN reproduit, contaminait le ré-ancrage),
mot-clé `ERRSTALE` sur les extensions ERR des trois étages FLC,
harmonisation run 8 → run 9 (docstring, CLI, suffixe par défaut
`_asure9e20b`, work dir `asure_work`, README), `requirements.txt` aux
versions exactes de l'env dip validé. AUCUN de ces correctifs ne
change le chemin numérique de la chaîne sur données réelles. Restent
ouverts (chantiers séparés) : inférence tuilée avec halo pour h>2048
(couture mesurée par la revue, nos chips 2048 non concernés),
reproductibilité train/val + suppression des échantillons zéro de
secours (avant tout run 10), suite pytest minimale.

**Bouton de force (10 août, retour BB « le bruit est trop effacé »)** :
le réseau n'a pas de réglage interne, le knob est un mélange convexe
`sortie = α·débruité + (1−α)·entrée`, photométriquement sûr (les deux
termes estiment la même scène ; le flux stellaire remonte VERS 100 %).
Deux implémentations : `--blend` dans `asure2_flc.py` (mélange au FLC,
entrée = chip pré-nettoyé DQ+deepCR, donc le bruit revient SANS les
CR), et `ArpNNN_dcr_drizzled.fits` (12 champs, drizzle des FLC deepCR
non débruités, même grille/noyau/ciel que les produits asure) pour
régler α en PixelMath dans PixInsight sans rien relancer. Bruit pixel
mesuré sur Arp 141 (fond, passe-haut) : brut 0.0119 e-/s, α=1.0 ÷14.8,
α=0.9 ÷6.6, α=0.8 ÷4.1, α=0.65 ÷2.6, α=0.5 ÷1.9. Planche :
`arp141_blend_tel.png`. Nuance honnête : au niveau drizzlé, la
composante (1−α) réinjecte aussi (1−α)× les rares CR que deepCR+veto a
laissés (CR vetoés sur halo brillant), que le modèle 2 canaux avait
effacés ; au niveau FLC (`--blend`), idem. Négligeable aux α ≥ 0.65.

### Prochaines étapes possibles
1. **Pré-nettoyer les rayons cosmiques des données d'entraînement** en
   utilisant les groupes eux-mêmes (10-20 expositions alignées par pointage :
   masque CR par sigma-clipping sur la pile). Supprime le dilemme étoile/CR
   à la racine ; le modèle apprendrait "toute source ponctuelle persiste".
   C'est ce que font les travaux N2N publiés (entraînés sur CRJ/DRZ).
2. **Réinjection des sources brillantes à l'inférence** (philosophie du
   projet DIP) : détecter les pics PSF-like > 30σ de l'entrée et les
   recopier. Heuristique mais immédiat.
3. Attention domaine : le modèle attend du FLC brut (e-/s, CRs présents).
   Sur une image drizzlée déjà nettoyée des CRs, le réflexe "effacer les
   pics brillants" est un handicap, pas une aide. Pour le cas d'usage
   "débruiter l'image finale de BB", la voie 1 est la bonne.

### Métriques à suivre
- Loss train / val
- PSNR sur patches de validation
- **Vérification visuelle obligatoire** : comparer original vs débruité dans PixInsight
  - Les étoiles doivent rester ponctuelles (pas d'anneaux)
  - Le fond de ciel doit être lissé sans structures parasites
  - Les détails de galaxie doivent être préservés

---

## Étape 4 : Inférence (`training/infer.py`) — FAIT (code prêt)

```python
# Full-image, pas de découpage en patches
# 1. Charger l'image (4096×2048)
# 2. Normalisation percentile [1, 99]
# 3. Padding réfléchissant pour rendre dims divisibles par 32
# 4. Passer dans le modèle (split vertical si > 2048 lignes)
# 5. Enlever le padding + dénormaliser
# 6. Sauvegarder en FITS (PRIMARY + SCI, header WCS conservé)
```

---

## Commandes

```bash
source venv/bin/activate

# Vérifier que le dataset charge bien quelques paires
python training/dataset_n2n.py --test

# Lancer l'entraînement
python -u training/train.py

# Reprendre depuis un checkpoint
python -u training/train.py --resume training/checkpoints/best.pth

# Débruiter une image
python training/infer.py \
    training_data/NGC_628/F814W/hst_..._chip1.fits \
    training/checkpoints/best.pth \
    output_denoised.fits
```

---

## Dépendances

```
astropy 7.2, numpy 2.4, torch 2.10 (MPS), astroquery, scipy
```
