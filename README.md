# TP5 — Autoencoders y VAE

Implementación **manual** (sin frameworks de deep learning) de **Autoencoder (AE)**,
**Denoising Autoencoder (DAE)** y **Variational Autoencoder (VAE)** sobre patrones de
7×5 píxeles: las 32 letras de `font.h` y 32 emojis.

Todo corre **dentro de un entorno virtual (venv)** y entra por un único punto: `main.py`.

---

## Instalación

Desde la raíz del proyecto:

```bash
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

> Activá el venv antes de cada corrida (`source venv/bin/activate`). El prompt muestra `(venv)`.

Dependencias: `numpy`, `pandas`, `matplotlib`, `questionary` (ver `requirements.txt`).

---

## Cómo correr

Hay **dos modos equivalentes**: el **menú interactivo (TUI)** y los **flags**. Los dos
arman exactamente los mismos comandos.

### Menú interactivo

```bash
python3 main.py        # sin argumentos -> abre el menú (questionary, dark mode)
```

### Por flags

Un único entry point con subcomandos:

```bash
python3 main.py ae       [--data letters|emoji] [--noise/--no-noise] [--salt P] [--save] ...
python3 main.py vae      [--data emoji|letters]  [--noise/--no-noise] [--kl W] [--save] ...
python3 main.py generate {ae|vae}  --weights P [--sampling ...] [-n N] [--plot]
python3 main.py plot     latent    --weights P [--sampling normal|bounds]
python3 main.py study    {architecture|hyperparams|architecture-dae|hyperparams-dae|denoising|kl} [--epochs N] [--seeds K]
python3 main.py --help          # ayuda general
python3 main.py ae --help       # flags de cada subcomando
```

Los scripts de `experiments/` conservan su `main(argv)` sólo como detalle interno: la
interfaz pública es siempre `main.py`.

---

## Configuración

`config.json` (en la raíz) es la **única fuente de verdad** de los hiperparámetros. Cualquier
flag los overridea por corrida.

| Clave | Default | Qué es |
|-------|---------|--------|
| `learning_rate` | 0.001 | paso de Adam |
| `epochs` | 7500 | máximo de épocas (corta antes si converge) |
| `training_mode` | batch | online / batch / minibatch |
| `batch_size` | 5 | sólo en minibatch |
| `epsilon` | 0.001 | corte por convergencia (NO es el ε de Adam, que es 1e-8) |
| `salt_p` | 0.1 | nivel de ruido Salt & Pepper del DAE |
| `kl_weight` | 0.03 | β del VAE; elegido del barrido `study kl` (trade-off reconstrucción/generación) |
| `seed` | null | seed por defecto; `--seed` la overridea |

---

## Flags por subcomando

**`ae`** (autoencoder / denoising):

| Flag | Descripción |
|------|-------------|
| `--data letters\|emoji` | dataset (default `letters`) |
| `--noise` / `--no-noise` | corromper la entrada con Salt & Pepper (default on → DAE) |
| `--salt P` | nivel de ruido (default `config.salt_p` = 0.1) |
| `--resample` / `--no-resample` | re-samplear el ruido cada época (default on = denoising real). `--no-resample` usa ruido FIJO: la red memoriza una corrupción puntual, **no** es denoising real |
| `--save` | guardar pesos en `output/<ae\|dae>/weights/<hiperparams>/weights.npz` |
| `--load PATH` | cargar pesos `.npz` y saltear el entrenamiento |
| `--no-viz` | no imprimir las reconstrucciones en ASCII |
| `--seed N` | seed de reproducibilidad |

**`vae`**: igual que `ae` (sin `--salt`/`--resample`/`--no-viz`) más `--kl W` (peso del término KL; default `config.kl_weight` = 0.03; `--kl 0` → sólo reconstrucción).

**`generate ae`**: `--weights P` (pesos del AE) · `--data` · `--sampling normal|bounds` (`normal`: z∼N(0,1) · `bounds`: uniforme dentro del rango latente ocupado) · `-n N` · `--plot` (latente con generados) · `--seed`.

**`generate vae`**: `--weights P` · `--data` · `--sampling prior|posterior` (`prior`: z∼N(0,I) canónico · `posterior`: z∼N(μ_i, σ_i)) · `-n N` · `--grid`/`--grid-n` (malla 2D) · `--scale` (posterior) · `--seed`.

**`plot latent`**: `--weights P` · `--data` · `--sampling normal|bounds` · `--seed` · `--output`.

**`study`**: `--epochs` (presupuesto del grid, default 2000), `--seeds` (corridas por celda → banda media ± σ), `--seed`, `--data`.
- `hyperparams` / `hyperparams-dae`: grid cruzado `opt × lr × init` con flags `--opts adam,gd` · `--lrs 1e-4,…,1e-2` · `--inits he,xavier,normal` · `--smoke` (defaults chicos). El `-dae` agrega `--salts 0.1,0.2`.
- `architecture` / `architecture-dae`: barren la lista de arquitecturas (HP fijos); flags `--hidden-act relu|tanh` · `--smoke`. El `-dae` agrega `--salts 0.1,0.2`.
- `denoising` agrega `--realizations`; `kl` agrega `--gen-samples`.

---

## Dónde caen las salidas

Todo va a `output/`, separado por **tipo** y por **combinación de hiperparámetros** (así
distintas corridas no se pisan):

```
output/
├── ae/    <kind>/<slug-hp>/...     # autoencoder sin ruido (1.a)
├── dae/   <kind>/<slug-hp>/...     # denoising AE (1.b)
├── vae/   <kind>/<slug-hp>/...     # variational (2)
└── study/ {architecture,hyperparams,architecture-dae,hyperparams-dae,denoising,kl}/<slug>/...
```
con `<kind>` ∈ `latent_space`, `presentation`, `weights`, `generated`. Los `.npz` son arrays
de NumPy por capa (`layer_0_w`, `layer_0_b`, …). `output/` no se versiona.

Cada estudio de grid (`architecture`, `hyperparams`, y sus `-dae`) deja bajo su `<slug>`:
`raw.csv` (una fila por combinación × [salt] × seed, fuente reproducible), `summary.csv`
(una por combinación con `rank` y marca `best`/`tie`), tabla(s) de ranking PNG, barras
(DAE: dos series por salt), `loss/combo_NN.png` por combinación + `top10_overlay.png`.

---

## Mapeo a la consigna

| Ítem | Comando | Resultado |
|------|---------|-----------|
| **a.1** AE reconstruye las 32 letras | `python3 main.py ae --no-noise` | reporte de px por carácter (≤1 px = objetivo) |
| **a.2** entrada vs reconstrucción | (idem a.1) | `output/ae/presentation/.../reconstructions.png` |
| **a.3** espacio latente 2D | (idem a.1) | `output/ae/latent_space/.../latent_space.png` |
| **a.4** letra nueva generada | `python3 main.py generate ae --weights <ae.npz> --plot` | `output/ae/generated/generated_samples.png` (+ latente con generados) |
| **b.1** DAE (entrada ruidosa → limpia) | `python3 main.py ae --salt 0.1` | reconstrucciones + tríptico |
| **b.2** error vs nivel de ruido | `python3 main.py study denoising` | `output/study/denoising/denoising_sweep.png` (banda media±σ) + tríptico `output/dae/presentation/.../denoising_triptych.png` |
| **2.a** VAE reconstrucción | `python3 main.py vae` | `output/vae/presentation/.../reconstructions.png` |
| **2.b** VAE espacio latente | `python3 main.py vae` · `python3 main.py plot latent --weights <vae.npz>` | `output/vae/latent_space/...` · `output/vae/combined_*.png` |
| **2.c** VAE muestra generada | `python3 main.py generate vae --weights <vae.npz>` | `output/vae/generated/vae_generated_prior.png` |

Ejemplo de flujo completo (AE → guardar pesos → generar):

```bash
python3 main.py ae --no-noise --seed 42 --save
python3 main.py generate ae --weights output/ae/weights/<hiperparams>/weights.npz -n 3 --plot --seed 42
```

---

## Estudios / barridos

Cuatro estudios de grid (AE + DAE) corren sobre un **motor compartido**
(`experiments/study_engine.py`) con un **criterio de selección único**
(`experiments/study_selection.py`): entrenan cada combinación con seed fija (N seeds →
banda media ± σ), evalúan con la métrica canónica, vuelcan CSV crudo + resumen, eligen el
mejor por **menor error medio de píxel** (empate dentro de ±1σ = indistinguible, no se
canta ganador) y plotean tablas/barras/curvas de loss.

```bash
python3 main.py study hyperparams           # grid cruzado opt×lr×init (30 combos), AE puro
python3 main.py study hyperparams-dae        # idem en denoising, × salt 0.1/0.2 (60 celdas)
python3 main.py study architecture           # lista de arquitecturas (bottleneck fijo 2), AE puro
python3 main.py study architecture-dae       # idem en denoising, dos series salt 0.1/0.2
python3 main.py study hyperparams --smoke     # validación rápida del pipeline (épocas/seeds bajas)
python3 main.py study denoising              # DAE: error de reconstrucción vs nivel de ruido
python3 main.py study kl                     # VAE: trade-off reconstrucción vs generación según kl_weight
```

- **DAE**: el error se mide contra el patrón **limpio** (no el ruidoso); ruido re-sampleado
  por época (`resample=on`). El `<=1px` es descriptivo, no criterio.
- El grid corre a `--epochs` fijo (default 2000) para comparar en igualdad; los CSV son
  **reanudables** (al rearrancar saltea celdas ya presentes por `label+salt+seed`).
- `study kl` es la base con la que se eligió `kl_weight = 0.03`. Despachan a
  `experiments/grid_architecture.py`, `grid_hyperparams.py`, `architecture_dae.py`,
  `hyperparams_dae.py`, `sweep_denoising.py`, `sweep_kl.py`.

---

## Estructura del proyecto

```
├── main.py                 # entry point (lanzador delgado)
├── config.json             # hiperparámetros (única fuente de verdad)
├── cli/                    # capa de entrada
│   ├── app.py              # subcomandos por flags
│   └── tui.py              # menú interactivo (questionary)
├── experiments/            # orquestación / scripts que producen resultados
│   ├── ae.py  vae.py  experiment.py  trainer.py
│   ├── generate.py  generate_vae.py  plot_latent_combined.py
│   ├── study_engine.py  study_selection.py        # motor + criterio compartidos por los 4 estudios de grid
│   ├── grid_architecture.py  grid_hyperparams.py  # AE: arquitectura / hiperparámetros
│   ├── architecture_dae.py   hyperparams_dae.py   # DAE: idem en denoising (salt 0.1/0.2)
│   └── sweep_denoising.py  sweep_kl.py
├── src/                    # librería reutilizable
│   ├── network/ activation/ cost/ optimizer/ noise/ metric/   # núcleo numérico
│   ├── data/    font.py                                       # carga de patrones
│   └── utils/   config.py evaluation.py weights_io.py sampling.py
├── graphs/                 # toda la visualización (dark mode)
├── assets/                 # datos: font.h, font_emoji.h
└── output/                 # generado: figuras y .npz (no se versiona)
```

Regla de dependencia: las flechas apuntan al núcleo. `src/` no importa de `experiments/`;
`experiments/` importa de `src/` y `graphs/`; `cli/` despacha a `experiments/`.

---

## Notas

- El entrenamiento puede tardar varios minutos (hasta ~7500 épocas, con corte por convergencia).
- Con `--load` se evalúa y grafica sin reentrenar (el modelo debe tener la misma arquitectura).
- El notebook `fonts.ipynb` es exploratorio; la corrida oficial es vía `main.py`.

---

## Fuentes y herramientas — TODO
