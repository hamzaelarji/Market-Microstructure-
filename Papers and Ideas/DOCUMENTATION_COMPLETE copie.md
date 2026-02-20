# Documentation complète du projet — Réplication de Guéant (2017)

## Table des matières

1. [Vue d'ensemble et architecture](#1-vue-densemble-et-architecture)
2. [Conventions globales](#2-conventions-globales)
3. [params.py — Paramètres du papier](#3-paramspy)
4. [intensity.py — Intensité et Hamiltonien](#4-intensitypy)
5. [test_intensity.py — Tests unitaires](#5-test_intensitypy)
6. [ode_solver_1d.py — Solveur single-asset](#6-ode_solver_1dpy)
7. [closed_form.py — Approximations fermées](#7-closed_formpy)
8. [ode_solver_2d.py — Solveur multi-asset](#8-ode_solver_2dpy)
9. [simulator.py — Monte Carlo](#9-simulatorpy)
10. [01_single_asset.py — Figures 1–5, 10–14](#10-01_single_assetpy)
11. [02_closed_form.py — Figures 2–7, 11–14](#11-02_closed_formpy)
12. [03_model_a_vs_b.py — Figures 8–9, 15–16](#12-03_model_a_vs_bpy)
13. [04_multi_asset.py — Figures 17–19](#13-04_multi_assetpy)
14. [05_monte_carlo.py — Extension Monte Carlo](#14-05_monte_carlopy)
15. [Mapping complet code ↔ papier](#15-mapping-complet-code--papier)
16. [Hypothèses et écarts par rapport au papier](#16-hypothèses-et-écarts-par-rapport-au-papier)
17. [Guide de debugging](#17-guide-de-debugging)

---

## 1. Vue d'ensemble et architecture

### Le problème du papier

Un market maker cote en continu un bid (prix d'achat) et un ask (prix de vente) sur un ou plusieurs actifs. Il gagne le spread à chaque transaction mais s'expose au **risque d'inventaire** (le prix bouge pendant qu'il détient une position). Le papier de Guéant (2017) résout le problème de contrôle stochastique optimal qui détermine les quotes optimaux en fonction du temps et de l'inventaire.

### Architecture du code

```
Couche 0 — PARAMÈTRES
  params.py           Données brutes de la Section 6 du papier

Couche 1 — BRIQUES MATHÉMATIQUES
  intensity.py        Λ(δ), C(ξΔ), H_ξ(p), H'_ξ(p), H''_ξ(p), δ*(p)
  test_intensity.py   21 tests de validation

Couche 2 — SOLVEURS D'EDP/EDO
  ode_solver_1d.py    Résout l'EDO (3.9)/(5.13 pour d=1) : θ(t,n) pour 1 actif
  ode_solver_2d.py    Résout l'EDO (5.13) pour d=2 actifs : θ(t,n₁,n₂)
  closed_form.py      Approximations (4.6)–(4.9) : formules analytiques directes

Couche 3 — SIMULATION
  simulator.py        Monte Carlo : trajectoires de prix + fills Poisson

Couche 4 — NOTEBOOKS (figures)
  01_single_asset.py  Figures 1–5 (IG) et 10–14 (HY)
  02_closed_form.py   Figures 2–7 (IG) et 11–14 (HY) : approx vs exact
  03_model_a_vs_b.py  Figures 8–9 (IG) et 15–16 (HY)
  04_multi_asset.py   Figures 17–19 (2 actifs corrélés)
  05_monte_carlo.py   Figs A–C (extension, pas dans le papier)
```

### Flux de données

```
params.py
    ↓
intensity.py ←── test_intensity.py (validation)
    ↓
ode_solver_1d.py ──→ 01_single_asset.py  (Figs 1-5, 10-14)
    │               ──→ 02_closed_form.py   (Figs 2-7, 11-14)
    │               ──→ 03_model_a_vs_b.py  (Figs 8-9, 15-16)
    │
closed_form.py ────→ 02_closed_form.py
    │
ode_solver_2d.py ──→ 04_multi_asset.py    (Figs 17-19)
    │
simulator.py ──────→ 05_monte_carlo.py    (Figs A-C)
```

---

## 2. Conventions globales

Ces conventions sont **critiques** et s'appliquent à TOUS les fichiers. Une erreur ici casse tout.

### Inventaire = nombre de LOTS, pas de dollars

```
n ∈ {-Q, -Q+1, ..., Q-1, Q}    ← entier, nombre de lots
q = n · Δ                        ← notionnel en dollars

Pour IG : Δ = 50M$, Q = 4, donc q ∈ {-200M$, ..., +200M$}
Pour HY : Δ = 10M$, Q = 4, donc q ∈ {-40M$, ..., +40M$}
```

**D'où vient Q = 4 ?** Le papier dit "Q·Δ = 4" dans la Section 6. Attention : le "4" est un nombre de lots (pas de dollars). La grille a donc 2Q+1 = 9 points : {-4, -3, -2, -1, 0, 1, 2, 3, 4}.

### δ = distance ABSOLUE au mid, en dollars

```
S^bid = S - δ^b     (prix auquel on achète)
S^ask = S + δ^a     (prix auquel on vend)

δ est en $/upfront rate. Ce n'est PAS un ratio, PAS un pourcentage.
```

**Conséquence pour le simulateur :** quand on achète (bid fill), le cash diminue de (S - δ^b) × Δ. Et **pas** de (S - δ^b × S) × Δ — c'est une erreur fréquente.

### Le paramètre ξ et le produit ξΔ

Le papier introduit une famille d'EDO paramétrée par ξ :
- **Model A** (utilité CARA) : ξ = γ. Le market maker a de l'aversion au risque sur l'exécution ET le prix.
- **Model B** (mean-variance) : ξ = 0. Seul le risque de prix est pénalisé.

**Le produit ξΔ** (et pas ξ seul) est ce qui entre dans les formules H_ξ et δ*. C'est une erreur très courante de l'oublier.

```
Pour IG Model A : ξΔ = γ × Δ_IG = 6e-5 × 50e6 = 3000
Pour HY Model A : ξΔ = γ × Δ_HY = 6e-5 × 10e6 = 600
Pour Model B :    ξΔ = 0
```

---

## 3. `params.py`

### Ce qu'il fait

Stocke les paramètres numériques de la Section 6 du papier (Table p. 31) dans des dictionnaires Python. Aucun calcul, juste des constantes.

### Partie du papier

**Section 6, paragraphe "Market data and parameters"** (p. 31). Le papier étudie deux indices de crédit CDX :
- **IG** (Investment Grade) : faible volatilité σ = 5.83×10⁻⁶, gros lots Δ = 50M$
- **HY** (High Yield) : haute volatilité σ = 2.15×10⁻⁵, petits lots Δ = 10M$

### Contenu détaillé

| Variable | IG | HY | Unité | Signification physique |
|---|---|---|---|---|
| `sigma` | 5.83e-6 | 2.15e-5 | $/√s | Volatilité de l'upfront rate par racine de seconde |
| `A` | 9.10e-4 | 1.06e-3 | 1/s | Taux d'arrivée des ordres quand δ=0 (≈3.3/heure pour IG) |
| `k` | 1.79e4 | 5.47e3 | 1/$ | Sensibilité de l'intensité à la distance au mid |
| `Delta` | 50e6 | 10e6 | $ | Notionnel d'un lot (taille de chaque trade) |
| `Q` | 4 | 4 | lots | Inventaire max en nombre de lots |
| `GAMMA` | 6e-5 | 6e-5 | 1/$ | Aversion au risque (commune aux deux actifs) |
| `RHO` | 0.9 | 0.9 | — | Corrélation IG–HY |
| `T` | 7200 | 7200 | s | Horizon de trading (2 heures) |

### Fonctions

- **`xi_model_a(gamma, Delta)`** : retourne γ (c'est trivial mais documente la convention)
- **`covariance_matrix(σ₁, σ₂, ρ)`** : construit Σ = [[σ₁², ρσ₁σ₂], [ρσ₁σ₂, σ₂²]]
- **`print_summary()`** : affichage lisible pour vérification rapide

### Fidélité au papier

**100% fidèle.** Tous les chiffres viennent directement de la Section 6.

---

## 4. `intensity.py`

### Ce qu'il fait

Implémente toutes les fonctions mathématiques liées à l'intensité d'exécution et au Hamiltonien. C'est le **cœur mathématique** du projet — tous les solveurs en dépendent.

### Partie du papier

- **Λ(δ)** : Eq. implicite, Section 2 — hypothèse d'intensité exponentielle
- **C(ξΔ)** : dérivé de l'Eq. (3.13) et Section 5
- **H_ξ(p)** : Eq. (3.13) (single-asset) et Eq. (5.8) (multi-asset)
- **δ*(p)** : Eqs. (4.6)–(4.7) et (5.16) — la quote optimale

### Fonctions une par une

#### `Lambda(delta, A, k)` → Λ(δ) = A·exp(-k·δ)

**Ce que c'est :** La fonction d'intensité. Elle donne le taux d'arrivée des ordres (en 1/s) en fonction de la distance δ au mid. Plus δ est grand (quote loin du mid), moins il y a de fills.

**Hypothèse :** Intensité exponentielle. Le papier traite le cas général mais toutes les formules fermées supposent cette forme. C'est le seul cas où H_ξ a une forme analytique.

**Propriétés :** Λ(0) = A, Λ > 0, Λ strictement décroissante.

#### `C_coeff(xi_Delta, k)` → C(ξΔ)

**Ce que c'est :** Un coefficient qui apparaît dans le Hamiltonien, issu du calcul du supremum dans H_ξ.

**Formule :**
```
Si ξΔ > 0 :  C = (1 + ξΔ/k)^{-(k/(ξΔ) + 1)}
Si ξΔ = 0 :  C = e^{-1} ≈ 0.3679
```

**D'où ça vient :** Dans l'Eq. (3.13) du papier, quand on spécialise au cas exponentiel Λ(δ) = Ae^{-kδ}, le supremum dans la définition de H_ξ peut se calculer analytiquement. Le résultat fait apparaître ce coefficient C.

**Limite :** C(ξΔ) → e⁻¹ quand ξΔ → 0 (continuité entre Model A et Model B).

**Attention :** L'argument est ξΔ (pas ξ seul). Si on passe ξ au lieu de ξΔ, les quotes seront fausses d'un facteur énorme.

#### `H_val(p, xi, A, k, Delta)` → H_ξ(p)

**Ce que c'est :** Le Hamiltonien. C'est la valeur du supremum dans le problème de contrôle, évalué à la "pente" p de θ.

**Formule :**
```
H_ξ(p) = (A·Δ/k) · C(ξΔ) · exp(-k·p)
```

**D'où ça vient :** Eq. (3.13) spécialisée au cas exponentiel. Dans la convention multi-asset (Section 5), le facteur Δ est inclus dans H (contrairement à la convention single-asset de la Section 3 où il n'y est pas).

**Rôle dans le solveur :** H_ξ apparaît dans l'EDO (5.13). À chaque pas de temps, pour chaque point de grille n, on évalue H_ξ en `p_bid = (θ(n) - θ(n+1))/Δ` et `p_ask = (θ(n) - θ(n-1))/Δ`.

**Propriétés :** H_ξ(p) > 0 pour tout p, H est strictement décroissante, H'(p) = -k·H(p).

#### `H_prime(p, xi, A, k, Delta)` → H'_ξ(p) = -k·H_ξ(p)

**Rôle :** Utilisé dans le Jacobien du Newton (solveurs 1D et 2D). La propriété H' = -k·H vient du fait que H est proportionnel à exp(-kp).

#### `H_second(p, xi, A, k, Delta)` → H''_ξ(p) = k²·H_ξ(p)

**Rôle :** Utilisé dans les approximations fermées (closed_form.py). La formule du slope ω dépend de H''_ξ(0).

#### `delta_star(p, xi, k, Delta)` → δ*(p)

**Ce que c'est :** La distance optimale au mid (la quote) qui réalise le supremum dans la définition de H_ξ(p). C'est LA formule qui donne les quotes finales.

**Formule :**
```
Si ξΔ > 0 :  δ*(p) = p + (1/(ξΔ)) · ln(1 + ξΔ/k)
Si ξΔ = 0 :  δ*(p) = p + 1/k
```

**D'où ça vient :** Eqs. (4.6)–(4.7) du papier. Le terme p est la "pente" de θ (différence finie entre voisins divisée par Δ), et le terme `(1/(ξΔ))·ln(1+ξΔ/k)` est un "markup" constant dû à l'incertitude d'exécution.

**Comment on l'utilise :**
```
p_bid(t, n) = (θ(t,n) - θ(t,n+1)) / Δ     ← pente vers le voisin bid
δ^b(t, n)   = δ*(p_bid)                      ← quote bid

p_ask(t, n) = (θ(t,n) - θ(t,n-1)) / Δ     ← pente vers le voisin ask
δ^a(t, n)   = δ*(p_ask)                      ← quote ask
```

### Fidélité au papier

**100% fidèle** au cas exponentiel. On utilise la convention multi-asset (facteur Δ dans H) même en 1D, ce qui est cohérent avec l'Eq. (5.13).

---

## 5. `test_intensity.py`

### Ce qu'il fait

21 tests unitaires qui valident `intensity.py` contre les propriétés analytiques du papier.

### Liste des tests et ce qu'ils vérifient

| Test | Propriété vérifiée | Équation du papier |
|---|---|---|
| `test_lambda_at_zero` | Λ(0) = A | Définition |
| `test_lambda_positive` | Λ(δ) > 0 ∀δ | Propriété exponentielle |
| `test_lambda_decreasing` | Λ strictement décroissante | Propriété exponentielle |
| `test_C_at_xi_zero` | C(0) = e⁻¹ | Eq. sous (3.13), cas ξ=0 |
| `test_C_positive` | C(ξΔ) > 0 | Nécessaire pour H > 0 |
| `test_C_known_values` | Valeur numérique pour IG | Calcul direct |
| `test_C_continuity_at_zero` | C(ξΔ) → e⁻¹ quand ξΔ → 0 | Continuité Model A → B |
| `test_H_at_p_zero` | H(0) = (AΔ/k)·C | Eq. sous (3.13) |
| `test_H_positive` | H(p) > 0 | Lemma 3.1 |
| `test_H_decreasing` | H strictement décroissante | Exponentielle décroissante |
| `test_H_prime_is_minus_k_H` | H'(p) = -k·H(p) | Dérivée de exp(-kp) |
| `test_H_second_is_k2_H` | H''(p) = k²·H(p) | Dérivée seconde |
| `test_H_prime_numerical` | H' vs différences finies | Validation numérique |
| `test_delta_star_model_b` | δ*(p) = p + 1/k pour ξ=0 | Eq. (4.7) |
| `test_delta_star_model_a` | δ*(p) = p + (1/(ξΔ))ln(...) pour ξ>0 | Eq. (4.6) |
| `test_delta_star_is_argmax_xi0` | δ* maximise Λ(δ)·Δ·(δ-p) | Définition de H₀ |
| `test_delta_star_is_argmax_xi_pos` | δ* maximise Λ(δ)/ξ·(1-exp(-ξΔ(δ-p))) | Définition de H_ξ, Eq. (5.8) |
| `test_H_equals_sup_xi0` | H₀(p) = Δ·sup_δ Λ(δ)(δ-p) | Eq. implicite Section 5 |
| `test_H_equals_sup_xi_pos` | H_ξ(p) = sup_δ Λ(δ)/ξ·(1-exp(-ξΔ(δ-p))) | Eq. (5.8) |
| `test_delta_star_order_of_magnitude` | δ* ∈ [10⁻⁷, 10⁻²] | Sanity check dimensionnel |
| `test_H_at_zero_order_of_magnitude` | H(0) ∈ [10⁻¹², 1] | Sanity check dimensionnel |

---

## 6. `ode_solver_1d.py`

### Ce qu'il fait

Résout l'EDO backward (5.13) spécialisée à d=1 pour obtenir θ(t,n), puis en extrait les quotes optimales δ^b(t,n) et δ^a(t,n). Contient 3 solveurs + 2 helpers internes.

### Partie du papier

**Eq. (3.9) / (5.13) pour d=1** — le système d'EDO :

```
∂_t θ_n + ½ γ σ² (nΔ)²
  - 𝟙_{n<Q}  H_ξ( (θ_n - θ_{n+1}) / Δ )
  - 𝟙_{n>-Q} H_ξ( (θ_n - θ_{n-1}) / Δ )  =  0

Condition terminale : θ_n(T) = -ℓ(|nΔ|)
```

**Signification physique :** θ(t,n) est la "correction" à la valeur du portefeuille due au risque d'inventaire et d'exécution. Les termes :
- `½ γ σ² (nΔ)²` : pénalité d'inventaire — coûte d'avoir n lots
- `H_ξ(p_bid)` : gain espéré de la prochaine transaction au bid
- `H_ξ(p_ask)` : gain espéré de la prochaine transaction à l'ask

### Fonctions une par une

#### `solve_model_a(params, gamma, T, N_t, ell_func)` — Solveur linéaire par v-transform

**Méthode :** Changement de variable v_n = exp(+k·θ_n / Δ) qui LINÉARISE l'EDO. Eq. (3.13) du papier :

```
∂_τ v_n = -½ k γ σ² n² Δ · v_n  +  A·C_{ξΔ} · (v_{n+1} + v_{n-1})
```

Ceci est un système linéaire tridiagonal à coefficients constants, résolu par Euler implicite backward. La matrice (I + dt·M) est factorisée une seule fois, puis appliquée 7200 fois via `scipy.linalg.solve_banded`.

**Avantage :** Plus rapide et numériquement plus stable que Newton (pas d'itération).
**Limitation :** Ne marche QUE pour Model A (ξ = γ) avec intensités exponentielles.

**Étapes :**
1. Calcule les coefficients α = ½kγσ²Δ (diagonale) et β = A·C (hors-diag)
2. Construit la matrice bande constante ab[3,N]
3. Boucle backward τ = 0 → T : résout (I + dt·M)·v^{new} = v^{old}
4. Convertit v → θ = +(Δ/k)·ln(v) et renverse le temps
5. Extrait les quotes via `_extract_quotes`

**Bug historique corrigé :** La ligne 88 avait originalement `-(Δ/k)·ln(v)` au lieu de `+(Δ/k)·ln(v)`. Ce signe inversé causait des quotes de signe opposé.

#### `solve_model_b(params, gamma, T, N_t, ell_func)` — Newton pour ξ = 0

**Méthode :** Euler implicite backward + Newton à chaque pas de temps. Identique à `solve_general` avec ξ = 0 fixé.

#### `solve_general(params, gamma, T, xi, N_t, ell_func)` — Newton pour ξ quelconque

**Méthode :** Résolution directe de l'EDO nonlinéaire par Newton.

**Schéma implicite :** À chaque pas de temps, on résout :

```
G(θ^{new}) = 0

G_n = θ^{new}_n - θ^{old}_n + dt · F_n(θ^{new})

F_n = ½ γ σ² (nΔ)²  -  𝟙_{n<Q} H_ξ(p_bid)  -  𝟙_{n>-Q} H_ξ(p_ask)
```

**ATTENTION :** F est évalué avec θ^{new} (implicite), PAS avec θ^{old} (explicite). C'est pour ça qu'il faut Newton.

**Newton :** On itère θ^{new} ← θ^{new} - J⁻¹·G, avec J = ∂G/∂θ^{new} tridiagonal.

- Initialisation : θ^{new} = θ^{old}
- 12 itérations max, stop si |correction| < 10⁻¹⁵
- Résolution du système tridiagonal via `solve_banded`

#### `_newton_residual(...)` — Résidu et Jacobien tridiagonal

Calcule G et les 3 diagonales de J pour un pas de Newton.

**Pour chaque n ∈ {-Q, ..., +Q} :**

```
# Pentes
p_bid = (θ_n - θ_{n+1}) / Δ    si n < Q
p_ask = (θ_n - θ_{n-1}) / Δ    si n > -Q

# Hamiltoniens
H_bid = H_ξ(p_bid)
H_ask = H_ξ(p_ask)

# Résidu
G_n = θ^{new}_n - θ^{old}_n + dt · (½γσ²(nΔ)² - H_bid - H_ask)

# Jacobien (utilise H' = -k·H)
J[n,n]   = 1 + dt · (k·H_bid + k·H_ask) / Δ²     ← diagonale
J[n,n+1] = -dt · k·H_bid / Δ²                       ← upper
J[n,n-1] = -dt · k·H_ask / Δ²                       ← lower
```

Le Jacobien est tridiagonal car θ_n ne dépend que de θ_{n-1} et θ_{n+1}.

#### `_extract_quotes(theta, lots, xi, k, Delta, Q)` — Extraction des quotes

**Étape :** Une fois θ(t,n) calculé pour tout (t,n), on récupère les quotes :

```
Pour chaque (t, n) :
  p_bid = (θ(t,n) - θ(t,n+1)) / Δ    → δ^b = δ*(p_bid)
  p_ask = (θ(t,n) - θ(t,n-1)) / Δ    → δ^a = δ*(p_ask)
```

**Bords :** δ^b n'existe pas pour n = +Q (pas de bid quand inventaire max), et δ^a n'existe pas pour n = -Q. On met NaN.

### Validation croisée

Les 3 solveurs doivent donner les mêmes résultats :
- `solve_model_a` ↔ `solve_general(xi=gamma)` : écart < 2×10⁻¹¹
- `solve_model_b` ↔ `solve_general(xi=0)` : identique par construction

### Fidélité au papier

**Très fidèle.** Le schéma numérique (Euler implicite backward + Newton) n'est pas explicitement décrit dans le papier, mais c'est la méthode standard pour les EDO raides de ce type. Le papier mentionne la résolution numérique de (3.9) sans détailler l'algorithme.

---

## 7. `closed_form.py`

### Ce qu'il fait

Implémente les approximations analytiques des Eqs. (4.6)–(4.9) qui donnent les quotes sans résoudre d'EDO.

### Partie du papier

**Section 4.2** — "Generalization of the Guéant-Lehalle-Fernandez-Tapia's formulas". Ces formules sont valides **loin de T** (régime asymptotique) et supposent des intensités symétriques Λ^b = Λ^a = Λ.

### Dérivation mathématique (résumée)

1. On remplace l'EDO discrète (3.9) par une EDP continue (4.1) via un développement de Taylor en Δ
2. On applique la transformation de Hopf-Cole v̄ = exp(-H''(0)/H'(0) · θ̃) pour linéariser
3. On étudie le comportement asymptotique T → ∞ par théorie spectrale
4. On en déduit que θ̃ ~ -½√(γσ²/(2H''(0))) · q² + const, d'où les quotes

### Fonctions

#### `approx_quotes(n, params, gamma, xi)` → (δ^b_approx, δ^a_approx)

**Formule implémentée (Eqs. 4.6–4.7) :**

```
δ_static = (1/(ξΔ)) · ln(1 + ξΔ/k)     si ξ > 0     (Eq. 4.6, partie constante)
         = 1/k                            si ξ = 0     (Eq. 4.7, partie constante)

ω = √(γσ² / (2·A·Δ·k·C(ξΔ)))           (slope, dérivé de H''(0) via Eq. 4.2)

δ^b_approx(n) = δ_static + ω · (2n + 1) · Δ / 2
δ^a_approx(n) = δ_static - ω · (2n - 1) · Δ / 2
```

**Décomposition :**
- `δ_static` : la moitié du spread "incompressible", lié au risque d'exécution seul
- `ω · (2n ± 1) · Δ / 2` : l'ajustement d'inventaire (le skew)

#### `approx_spread(n, ...)` → δ^b + δ^a

**Eq. (4.8) :** Spread ≈ 2·δ_static + ω·Δ. **Le spread approché est constant** (indépendant de n). C'est une approximation — le vrai spread dépend de n.

#### `approx_skew(n, ...)` → δ^b - δ^a

**Eq. (4.9) :** Skew ≈ 2n·ω·Δ. **Le skew approché est linéaire** en n. Le vrai skew est non-linéaire.

### Fidélité au papier

**Fidèle aux Eqs. (4.6)–(4.9).** Les approximations sont par nature imparfaites — c'est le point des Figures 2–5 du papier qui montrent l'écart. L'approximation est meilleure pour :
- Petit |n| (inventaire faible)
- Petite σ (faible volatilité) — cf. Figures 6–7

---

## 8. `ode_solver_2d.py`

### Ce qu'il fait

Résout l'EDO (5.13) pour d = 2 actifs (IG + HY) couplés par la corrélation ρ, via Newton sur Euler implicite avec Jacobien creux.

### Partie du papier

**Section 5, Eq. (5.13)** — l'EDO multi-asset :

```
∂_t θ(n₁,n₂) + ½ γ q^T Σ q
  - Σᵢ 𝟙_{nᵢ<Qᵢ}  Hᵢ_ξ( (θ(n) - θ(n+eᵢ)) / Δᵢ )
  - Σᵢ 𝟙_{nᵢ>-Qᵢ} Hᵢ_ξ( (θ(n) - θ(n-eᵢ)) / Δᵢ )  =  0
```

**Différences clés avec le 1D :**
- Le terme de risque est `½ γ q^T Σ q` au lieu de `½ γ σ² q²`. La matrice Σ couple les deux actifs via ρ.
- Chaque actif a ses propres paramètres (A, k, Δ) donc ses propres H^i_ξ.
- La grille est 2D : 9 × 9 = 81 points au lieu de 9.

### Fonctions

#### `_build_grid(Q1, Q2)` → grille 2D

Construit un dictionnaire (n₁, n₂) → indice plat j ∈ {0, ..., 80}. La grille est {-4,...,4} × {-4,...,4}.

#### `solve_2d(params1, params2, gamma, rho, T, xi, N_t)` — Solveur principal

**Même logique que le 1D** mais sur la grille 2D :
1. Terminal : θ = 0 (ou -ℓ)
2. Boucle backward : Newton sur G(θ^{new}) = 0 à chaque pas
3. Résolution du système creux 81×81 via `scipy.sparse.linalg.spsolve`
4. Extraction des 4 quotes (bid/ask × 2 actifs)

#### `_residual_and_jacobian(...)` — Résidu et Jacobien creux

Pour chaque point j de la grille (n₁, n₂) :
- Calcule le risque d'inventaire : `½ γ [n₁Δ₁, n₂Δ₂]^T · Σ · [n₁Δ₁, n₂Δ₂]`
- Pour chaque actif i, pour chaque côté (bid/ask), calcule H et H'
- Assemble le résidu G_j et les éléments non-nuls de J

Le Jacobien est **creux** (pas tridiagonal) : chaque θ_j dépend de 4 voisins (n₁±1, n₂) et (n₁, n₂±1). On utilise `lil_matrix` pour l'assemblage puis conversion en `csc_matrix` pour la résolution.

#### `_extract_quotes_2d(...)` — 4 quotes par point de grille

Même logique que le 1D : pour chaque (n₁, n₂), calcule p = (θ(n) - θ(voisin))/Δᵢ puis δ* = delta_star(p).

### Validation

**Test ρ = 0 :** Quand les deux actifs sont indépendants, les quotes de l'actif 1 à n₂ = 0 doivent coïncider exactement avec le solveur 1D. Écart mesuré : < 6×10⁻²⁰.

### Fidélité au papier

**Fidèle.** Implémente directement l'Eq. (5.13). La méthode numérique (Newton + Euler implicite + Jacobien creux) est un choix d'implémentation standard.

---

## 9. `simulator.py`

### Ce qu'il fait

Simule des trajectoires Monte Carlo d'un market maker qui utilise les quotes pré-calculées par les solveurs ODE. Enregistre le P&L, l'inventaire, le cash et le MtM à chaque instant.

### Partie du papier

**Pas directement dans le papier.** C'est une extension pour valider numériquement que la stratégie optimale fait bien mieux qu'une stratégie naïve.

### Fonctions

#### `simulate_1d(sol, params, gamma, T, N_sim, seed)` — Simulation optimale

**Algorithme (pour chaque trajectoire m) :**

```
Initialiser S = 0, X = 0, n = 0

Pour t = 0, dt, 2dt, ..., T-dt :
    1. LOOKUP des quotes : δ^b = delta_bid[t, n+Q],  δ^a = delta_ask[t, n+Q]
    
    2. PRIX : S ← S + σ√dt · Z    (Z ~ N(0,1))
    
    3. FILLS (Poisson) :
       prob_bid = A·exp(-k·δ^b)·dt
       Si U_bid < prob_bid  ET  n < Q :
           n ← n + 1                     (inventaire augmente)
           X ← X - (S - δ^b) · Δ         (on PAYE S - δ^b par lot, cash diminue)
       
       prob_ask = A·exp(-k·δ^a)·dt
       Si U_ask < prob_ask  ET  n > -Q :
           n ← n - 1                     (inventaire diminue)
           X ← X + (S + δ^a) · Δ         (on REÇOIT S + δ^a par lot, cash augmente)
    
    4. MtM = X + n · Δ · S

P&L_final = X_T + n_T · Δ · S_T
```

**Points critiques :**
- δ est **ABSOLU** : `S^bid = S - δ^b`, pas `S·(1 - δ^b)`. Erreur documentée dans le correctif.
- Les fills bid et ask sont **indépendants** (deux tirages uniformes séparés).
- On fait le lookup dans la table précalculée `delta_bid[t_idx, lot_idx]` — pas de recalcul en ligne.

#### `simulate_naive(params, gamma, T, half_spread, N_t, N_sim, seed)` — Stratégie naïve

Même simulation mais avec un spread constant δ^b = δ^a = `half_spread`, indépendant de l'inventaire et du temps.

### Fidélité au papier

**Le simulateur est standard** mais n'apparaît pas dans le papier. La dynamique de prix (Brownien pur, pas de drift) et les fills Poisson sont cohérents avec les hypothèses du modèle.

---

## 10. `01_single_asset.py`

### Ce qu'il fait

Résout l'EDO pour IG et HY séparément (Model A, ξ = γ) et produit les figures single-asset du papier.

### Figures produites

| Figure papier | Ce qui est tracé | Axes |
|---|---|---|
| **Fig 1** | t → δ^{IG,bid}(t, n) pour n=-3,...,3 | x: temps, y: δ^bid |
| **Fig 2** | n → δ^{IG,bid}(0, n) | x: inventaire, y: δ^bid |
| **Fig 3** | n → δ^{IG,ask}(0, n) | x: inventaire, y: δ^ask |
| **Fig 4** | n → spread_IG(0, n) = δ^b + δ^a | x: inventaire, y: spread |
| **Fig 5** | n → skew_IG(0, n) = δ^b - δ^a | x: inventaire, y: skew |
| **Fig 10** | idem Fig 1 pour HY | |
| **Figs 11–14** | idem Figs 2–5 pour HY | |

### Ce que tu dois vérifier

- **Fig 1/10 :** Les courbes doivent converger vers un plateau loin de T. C'est le "régime asymptotique" mentionné Section 4.
- **Fig 2/11 :** δ^bid doit être **croissant** en n (plus d'inventaire → bid plus loin du mid = plus conservateur).
- **Fig 3/12 :** δ^ask doit être **décroissant** en n.
- **Fig 4/13 :** Le spread n'est **PAS constant** (contrairement à l'approximation fermée). Il est minimum à n=0 et augmente pour |n| grand.
- **Fig 5/14 :** Le skew n'est **PAS linéaire**. Antisymétrique en n (symétrie bid/ask quand n → -n).

---

## 11. `02_closed_form.py`

### Ce qu'il fait

Superpose les approximations fermées (4.6)–(4.9) sur les solutions exactes de l'EDO.

### Figures produites

| Figure papier | Contenu |
|---|---|
| **Figs 2–5** | IG, σ normal : croix = exact (ODE), ligne = approx fermée |
| **Figs 6–7** | IG, σ/2 : idem mais l'approx est MEILLEURE (le papier le montre) |
| **Figs 11–14** | HY, σ normal : idem IG |

### Ce que tu dois vérifier

- **Pour |n| petit :** approx et exact quasi-confondus.
- **Pour |n| = 3 :** écart visible. L'approximation linéarise la dépendance en n, mais le vrai comportement est non-linéaire.
- **Pour σ/2 :** l'écart se réduit. C'est parce que l'approximation vient d'un développement qui suppose σ petit.
- **Spread approx = CONSTANT (ligne horizontale).** Le spread exact est une courbe en U.
- **Skew approx = DROIT.** Le skew exact est une courbe en S.

---

## 12. `03_model_a_vs_b.py`

### Ce qu'il fait

Compare les quotes entre Model A (ξ = γ, utilité CARA) et Model B (ξ = 0, mean-variance).

### Figures produites

| Figure papier | Contenu |
|---|---|
| **Fig 8** | n → δ^{IG,bid}(0,n) : Model A (croix) vs Model B (cercles) |
| **Fig 9** | n → δ^{IG,ask}(0,n) : Model A vs B |
| **Figs 15–16** | Idem pour HY |

### Ce que tu dois vérifier

- **Les deux modèles sont très proches** (diff < 6%). C'est le message principal : Model B est une simplification valide de Model A.
- Le script affiche aussi la différence max en absolu et en relatif.
- Model A a un spread légèrement plus petit que Model B (le market maker CARA réduit son spread pour maximiser l'exécution).

---

## 13. `04_multi_asset.py`

### Ce qu'il fait

Résout le problème à 2 actifs (IG + HY corrélés) et trace les surfaces 3D et les coupes.

### Figures produites

| Figure papier | Contenu |
|---|---|
| **Fig 17** | Surface 3D (n_IG, n_HY) → δ^{IG,bid}(0) pour ρ=0.9 |
| **Fig 18** | Surface 3D (n_IG, n_HY) → δ^{HY,bid}(0) pour ρ=0.9 |
| **Fig 19** | Coupe n_IG → δ^{HY,bid}(0, n_IG, n_HY=0) pour ρ ∈ {0, 0.3, 0.6, 0.9} |

### Ce que tu dois vérifier

- **Fig 17–18 :** Les surfaces doivent être "tiltées" — l'inventaire d'un actif affecte les quotes de l'autre via la corrélation.
- **Fig 19 (CRITIQUE) :**
  - **ρ = 0 :** courbe PLATE (l'inventaire IG n'affecte pas le bid HY)
  - **ρ = 0.9 :** forte pente positive (quand n_IG augmente, le market maker remonte son bid HY car les deux actifs bougent ensemble)
  - La pente augmente avec ρ

---

## 14. `05_monte_carlo.py`

### Ce qu'il fait

Simule 2000 trajectoires avec la stratégie optimale et 2000 avec une stratégie naïve (spread fixe), puis compare.

### Figures produites (extension, PAS dans le papier)

| Figure | Contenu |
|---|---|
| **Fig A** | Histogramme P&L : optimal vs naïf |
| **Fig B** | Trajectoire type : prix, inventaire, MtM |
| **Fig C** | E[|inventaire|] et std(inventaire) au cours du temps |

### Ce que tu dois vérifier

- **Fig A :** La distribution optimale est plus concentrée (std plus faible) et le Sharpe est meilleur. La stratégie naïve a des queues plus épaisses.
- **Fig B :** L'inventaire doit osciller autour de 0 (la stratégie ramène l'inventaire vers 0 via le skew).
- **Fig C :** E[|inv|] optimale < E[|inv|] naïve à tout instant. La stratégie optimale contrôle mieux l'inventaire.
- **Sharpe ratio :** optimal ≈ 0.75, naïf ≈ 0.40 (valeurs typiques obtenues).

---

## 15. Mapping complet code ↔ papier

| Équation papier | Fichier | Fonction/Ligne | Description |
|---|---|---|---|
| Λ(δ) = Ae^{-kδ} | `intensity.py` | `Lambda()` | Intensité exponentielle |
| C_ξ (sous Eq. 3.13) | `intensity.py` | `C_coeff()` | Coefficient du Hamiltonien |
| H_ξ(p) (Eq. 3.13) | `intensity.py` | `H_val()` | Hamiltonien cas exponentiel |
| δ*(p) (Eqs. 4.6-4.7) | `intensity.py` | `delta_star()` | Quote optimale |
| Eq. 3.9 / 5.13 (d=1) | `ode_solver_1d.py` | `solve_general()` | EDO single-asset |
| Eq. 3.13 (linéaire) | `ode_solver_1d.py` | `solve_model_a()` | v-transform + tridiag |
| Eq. 5.13 (d=2) | `ode_solver_2d.py` | `solve_2d()` | EDO multi-asset |
| Eqs. 4.6-4.7 | `closed_form.py` | `approx_quotes()` | Quotes approchées |
| Eq. 4.8 | `closed_form.py` | `approx_spread()` | Spread approché |
| Eq. 4.9 | `closed_form.py` | `approx_skew()` | Skew approché |
| Eqs. 3.14, 3.16, 5.16 | `ode_solver_1d.py` | `_extract_quotes()` | δ^b = δ*(p_bid) |
| Section 6, Table | `params.py` | `IG`, `HY`, ... | Paramètres numériques |
| Figs 1–5 | `01_single_asset.py` | `plot_asset()` | IG quotes |
| Figs 10–14 | `01_single_asset.py` | `plot_asset()` | HY quotes |
| Figs 2–7 | `02_closed_form.py` | `overlay_plots()` | Approx vs exact |
| Figs 8–9, 15–16 | `03_model_a_vs_b.py` | `compare_ab()` | Model A vs B |
| Figs 17–19 | `04_multi_asset.py` | `surface_plot()` | Multi-asset |

---

## 16. Hypothèses et écarts par rapport au papier

### Hypothèses fidèles au papier

1. **Prix = Brownien géométrique sans drift** : dS = σ dW. Pas de tendance, pas de jumps, pas de mean-reversion.
2. **Intensités exponentielles symétriques** : Λ^b = Λ^a = Ae^{-kδ}. Les fills bid et ask ont la même loi.
3. **Fills Poisson indépendants** : à chaque dt, la probabilité de fill est Λ(δ)·dt.
4. **Pas d'impact permanent** : un trade ne déplace pas le mid price S.
5. **Trading continu** : pas de tick size, pas de queue de priorité.
6. **Pas de coûts de transaction** : pas de frais, pas de latence.
7. **Corrélation constante** : ρ ne change pas au cours du temps.
8. **Pénalité terminale ℓ = 0** : pas de pénalité sur l'inventaire final (sauf si `ell_func` est spécifié).

### Écarts par rapport au papier

| Point | Papier | Notre code | Impact |
|---|---|---|---|
| Convention H_ξ | Section 3 : H sans Δ / Section 5 : H avec Δ | Toujours avec Δ (convention Section 5) | Aucun — les quotes sont identiques |
| Schéma numérique | Non spécifié | Euler implicite backward + Newton | Standard, pas d'erreur |
| N_t = 7200 | Non spécifié | 1 step/seconde | Suffisamment fin (convergé) |
| N_t = 720 pour 2D | — | Réduit pour la vitesse | Légère perte de précision mais convergé |
| Simulateur | Absent | Monte Carlo avec lookup | Extension propre |
| Figures exactes | — | Reproduites en PNG pas en LaTeX | Cosmétique seulement |

### Choses que le papier fait et qu'on ne fait PAS

1. **Approximations multi-asset fermées (Section 5.4)** : la matrice Ω, les Eqs. (5.18)–(5.19). On ne les implémente pas (seul le solveur 2D exact est codé).
2. **Preuve du théorème de vérification** (Theorem 3.2) : on ne vérifie pas formellement que la solution HJB est bien la fonction valeur.
3. **Étude de la comparative statics** (Section 4.3) : on ne trace pas l'effet de chaque paramètre individuellement.

---

## 17. Guide de debugging

### Symptôme → Cause → Fix

| Symptôme | Cause probable | Où regarder |
|---|---|---|
| Quotes toutes NaN | `_extract_quotes` ne trouve pas les voisins | Vérifier que `lots` est bien {-Q,...,+Q} |
| Spread NÉGATIF à n=0 | Bug de signe dans v-transform | `ode_solver_1d.py` ligne 88 : doit être `+(Δ/k)·ln(v)` |
| Quotes d'ordre de grandeur absurde (>1 ou <10⁻¹⁰) | ξ utilisé au lieu de ξΔ dans `delta_star` ou `C_coeff` | Vérifier que `H_val` passe bien `xi*Delta` à `C_coeff` |
| 2D(ρ=0) ≠ 1D | Bug dans la matrice Σ ou dans l'indexation 2D | Vérifier `_build_grid` et l'assemblage du terme `q^T Σ q` |
| Fig 19 plate à ρ=0.9 | Le terme ρσ₁σ₂ n'est pas pris en compte | Vérifier la construction de Sig dans `solve_2d` |
| Simulateur : P&L diverge | δ traité comme relatif au lieu d'absolu | Vérifier `X -= (S - db) * Delta` (pas `S * (1-db) * Delta`) |
| Newton ne converge pas | dt trop grand ou mauvaise initialisation | Réduire dt (augmenter N_t) ou vérifier le Jacobien |
| Closed-form très loin de l'exact | Bug dans ω ou dans δ_static | Vérifier que `C_coeff` reçoit `xi_Delta` et pas `xi` |
| `test_intensity.py` échoue | Import cassé | Vérifier que tous les fichiers sont dans le même dossier |

### Tests rapides pour valider l'installation

```bash
# 1. Paramètres OK ?
python params.py
# → doit afficher ξΔ = 3000 pour IG, 600 pour HY

# 2. Intensity OK ?
python test_intensity.py
# → 21/21 PASSED

# 3. Solveur 1D OK ?
python -c "
from params import IG, GAMMA, T
from ode_solver_1d import solve_general
sol = solve_general(IG, GAMMA, T, xi=GAMMA, N_t=7200)
print(f'δ^b(0,n=0) = {sol[\"delta_bid\"][0, 4]:.6e}')
print(f'δ^a(0,n=0) = {sol[\"delta_ask\"][0, 4]:.6e}')
print(f'spread(0,0) = {sol[\"delta_bid\"][0,4] + sol[\"delta_ask\"][0,4]:.6e}')
"
# → spread > 0, δ^b(0,0) = δ^a(0,0) (symétrie à n=0)
# → δ^b ≈ 1.2e-4, spread ≈ 2.4e-4

# 4. Cross-check v-transform vs Newton
python -c "
from params import IG, GAMMA, T
from ode_solver_1d import solve_model_a, solve_general
import numpy as np
a = solve_model_a(IG, GAMMA, T, N_t=7200)
g = solve_general(IG, GAMMA, T, xi=GAMMA, N_t=7200)
mask = np.isfinite(a['delta_bid'][0]) & np.isfinite(g['delta_bid'][0])
print(f'Max diff = {np.max(np.abs(a[\"delta_bid\"][0,mask] - g[\"delta_bid\"][0,mask])):.2e}')
"
# → Max diff < 1e-10 (sinon bug de signe dans v-transform)
```
