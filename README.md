# TP5 — Autoencoders y VAE

Implementación de **Autoencoder (AE)**, **Denoising Autoencoder (DAE)** y **Variational Autoencoder (VAE)** sobre patrones de 7×5 píxeles (letras ASCII y emojis).

Todo el proyecto se ejecuta **dentro de un entorno virtual (venv)**.

---

## Requisitos

- Python 3.10 o superior
- Dependencias: `numpy`, `pandas`, `matplotlib` (ver `requirements.txt`)

---

## Instalación

Clonar o descargar el repositorio y, desde la raíz del proyecto:

### Windows (PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> **Importante:** activar el venv antes de cada corrida. El prompt debería mostrar `(venv)`.

---

## Cómo correr

El punto de entrada es `main.py`. Tiene dos subcomandos: `ae` y `vae`.

```powershell
# Ver ayuda general
python main.py --help

# Autoencoder / Denoising AE (dataset por defecto: letras)
python main.py ae

# Variational Autoencoder (dataset por defecto: emojis)
python main.py vae
```

---

## Comandos de ejemplo

### Autoencoder sobre letras (con ruido, modo DAE)

```powershell
python main.py ae
```

### Autoencoder sin ruido

```powershell
python main.py ae --data letters --no-noise
```

### Autoencoder sobre emojis

```powershell
python main.py ae --data emoji
```

### VAE sobre emojis

```powershell
python main.py vae
```

### Entrenar y guardar pesos

Guarda los pesos en `weights_<data>.npz` (por ejemplo `weights_letters.npz`):

```powershell
python main.py ae --data letters --save
python main.py vae --data emoji --save
```

### Cargar pesos sin reentrenar

El modelo debe tener **la misma arquitectura** con la que se guardaron los pesos:

```powershell
python main.py ae --load weights_letters.npz
python main.py vae --load weights_emoji.npz
```

### Sin visualización ASCII (solo AE)

```powershell
python main.py ae --no-viz
```

---

## Opciones disponibles

| Opción                   | Subcomando   | Descripción |
|--------------------------|--------------|-------------|
| `--data letters\|emoji`  | `ae`, `vae`  | Dataset a usar (`letters` por defecto en `ae`, `emoji` en `vae`) |
| `--noise` / `--no-noise` | `ae`, `vae`  | Corromper la entrada con ruido sal y pimienta (activado por defecto) |
| `--save`                 | `ae`, `vae`  | Guardar pesos en `weights_<data>.npz` al terminar el entrenamiento |
| `--load PATH`            | `ae`, `vae`  | Cargar pesos desde un `.npz` y saltear el entrenamiento |
| `--no-viz`               | `ae`         | No imprimir reconstrucciones en ASCII |
| `--seed`                 | `ae`, `vae`  | seed opcional para reproducibilidad |

---

## Qué produce cada corrida

Al finalizar, el programa imprime en consola:

- Épocas de entrenamiento (o 0 si se cargaron pesos)
- Error de reconstrucción (BCE)
- Reporte de **error por carácter en píxeles** (objetivo del TP: ≤ 1 píxel en el peor caso)
- Métricas extra del VAE (KL divergence y loss total)

Además genera archivos en la raíz del proyecto:

| Archivo | Cuándo |
|---------|--------|
| `latent_space_<data>.png` | Corrida de `ae` |
| `latent_space_<data>_vae.png` | Corrida de `vae` |
| `weights_<data>.npz` | Si se usó `--save` |

Los pesos `.npz` son arrays de NumPy guardados por capa (`layer_0_w`, `layer_0_b`, …).

---

## Estructura del proyecto

```
├── main.py              # Punto de entrada
├── cli/app.py           # Argumentos de línea de comandos
├── ae.py                # Autoencoder / DAE
├── vae.py               # Variational Autoencoder
├── experiment.py        # Pipeline compartido (datos, entrenamiento, gráficos)
├── evaluation.py        # Reporte de error por píxel
├── weights_io.py        # Guardar / cargar pesos (.npz)
├── font.py              # Carga de patrones desde font.h / font_emoji.h
├── trainer.py           # Loop de entrenamiento
├── network/             # Modelos (AE, VAE, capas, neuronas)
├── activation/          # Funciones de activación
├── cost/                # Funciones de costo
├── optimizer/           # Optimizadores (Adam, etc.)
└── noise/               # Ruido sal y pimienta
```

---

## Notas

- El entrenamiento puede tardar varios minutos (hasta ~7500 épocas con convergencia temprana).
- Con `--load`, se evalúa y grafica sin volver a entrenar.
- El notebook `fonts.ipynb` es exploratorio; la corrida oficial del TP es vía `main.py`.
