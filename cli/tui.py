"""
cli/tui.py — menú interactivo (questionary), en dark mode.

Se abre cuando corrés `python3 main.py` sin argumentos. No tiene lógica propia:
junta opciones, arma la MISMA lista de flags que escribirías a mano y se la pasa al
entry point de siempre (cli.app.main para ae/vae/study, o el main() del script para
generar/visualizar). Por eso TUI y flags hacen exactamente lo mismo.

La paleta espeja la de graphs/style.py (se duplican sólo los hex para no arrastrar
matplotlib al abrir el menú). Pensado para terminal de fondo oscuro; símbolos ASCII.
"""
import questionary

from src.utils.config import load_config

# Paleta (espejo de graphs/style.py).
BLUE, ORANGE, RED = "#4aa3ff", "#ffb454", "#ff6b6b"
FG, FG_DIM, MUTED = "#e6e9ef", "#3a4150", "#8b94a6"

QMARK = "?"
POINTER = ">"

THEME = questionary.Style([
    ("qmark", f"fg:{ORANGE} bold"),        # el "?" de cada pregunta
    ("question", f"fg:{FG} bold"),         # texto de la pregunta
    ("answer", f"fg:{BLUE} bold"),         # lo que quedó elegido
    ("pointer", f"fg:{BLUE} bold"),        # ">" sobre la opción actual
    ("highlighted", f"fg:{BLUE} bold"),    # opción bajo el cursor
    ("selected", f"fg:{ORANGE}"),          # opción marcada (checkbox)
    ("separator", f"fg:{FG_DIM}"),
    ("instruction", f"fg:{MUTED} italic"),
    ("text", f"fg:{FG}"),
    ("disabled", f"fg:{FG_DIM} italic"),
])


def _color(text, hexcolor, bold=False):
    """Envuelve text en un color truecolor ANSI (para el banner y los avisos)."""
    r, g, b = (int(hexcolor[i:i + 2], 16) for i in (1, 3, 5))
    prefix = ("\x1b[1m" if bold else "") + f"\x1b[38;2;{r};{g};{b}m"
    return f"{prefix}{text}\x1b[0m"


def _banner():
    """Banner ASCII en dark mode, una sola vez al abrir el menú."""
    bar = _color("+" + "-" * 48 + "+", FG_DIM)

    def row(text, color, bold=False):
        return _color("|", FG_DIM) + _color(text.ljust(48), color, bold) + _color("|", FG_DIM)

    print("\n" + bar)
    print(row("   TP5  -  Autoencoders / VAE   (AE - DAE - VAE)", BLUE, bold=True))
    print(row("   Deep Learning   *   SIA - ITBA", MUTED))
    print(bar + "\n")


# --- preguntas con el tema ya aplicado ------------------------------------------------

def _select(message, choices, **kw):
    return questionary.select(message, choices=choices, style=THEME,
                              qmark=QMARK, pointer=POINTER, **kw)


def _text(message, **kw):
    return questionary.text(message, style=THEME, qmark=QMARK, **kw)


def _confirm(message, **kw):
    return questionary.confirm(message, style=THEME, qmark=QMARK, **kw)


def _path(message, **kw):
    return questionary.path(message, style=THEME, qmark=QMARK, **kw)


def _ask(question):
    """Corre una pregunta; si el usuario cancela (Ctrl-C / Esc → None), aborta el flujo."""
    answer = question.ask()
    if answer is None:
        raise KeyboardInterrupt
    return answer


# --- helpers de flags -----------------------------------------------------------------

def _seed_flag(cfg):
    """Pide la seed (vacío = usa la de config.json). Devuelve [] o ['--seed', valor]."""
    default = "" if cfg.seed is None else str(cfg.seed)
    seed = _ask(_text("Seed (vacío = default de config):", default=default)).strip()
    return ["--seed", seed] if seed else []


def _confirm_and_run(display, run):
    """Pantalla de confirmación: muestra el comando equivalente y, si acepta, lo ejecuta."""
    print("\n  " + _color(display, BLUE) + "\n")
    if _ask(_confirm("¿Ejecutar?", default=True)):
        run()
    else:
        print(_color("Cancelado.", MUTED))


def _dispatch_main(argv):
    """Ejecuta argv por el MISMO main() que usan los flags (python3 main.py ...)."""
    from cli.app import main as app_main
    _confirm_and_run("python3 main.py " + " ".join(argv), lambda: app_main(argv))


# --- menús por acción -----------------------------------------------------------------

def _menu_ae(cfg, denoising):
    data = _ask(_select("Dataset:", choices=["letters", "emoji"], default="letters"))
    argv = ["ae", "--data", data]

    if denoising:
        salt = _ask(_text("Nivel de ruido salt:", default=str(cfg.salt_p)))
        print(_color("  no-resample = ruido FIJO: la red memoriza una corrupción puntual, no es denoising real", MUTED))
        resample = _ask(_confirm("¿Re-samplear el ruido en cada época? (recomendado)", default=True))
        argv += ["--noise", "--salt", salt, "--resample" if resample else "--no-resample"]
    else:
        argv += ["--no-noise"]

    argv += _seed_flag(cfg)
    if _ask(_confirm("¿Guardar los pesos?", default=False)):
        argv += ["--save"]
    if not _ask(_confirm("¿Mostrar las reconstrucciones (ASCII)?", default=True)):
        argv += ["--no-viz"]

    _dispatch_main(argv)


def _menu_vae(cfg):
    data = _ask(_select("Dataset:", choices=["emoji", "letters"], default="emoji"))
    noise = _ask(_confirm("¿Con ruido?", default=True))
    kl = _ask(_text("kl_weight (β):", default=str(cfg.kl_weight)))

    argv = ["vae", "--data", data, "--noise" if noise else "--no-noise", "--kl", kl]
    argv += _seed_flag(cfg)
    if _ask(_confirm("¿Guardar los pesos?", default=False)):
        argv += ["--save"]

    _dispatch_main(argv)


def _menu_generate(cfg):
    what = _ask(_select("¿Qué querés hacer?", choices=[
        questionary.Choice("Generar muestras del AE", "ae",
                           description="z del latente -> decode -> letras nuevas (a.4)"),
        questionary.Choice("Generar muestras del VAE", "vae",
                           description="z ~ N(0,I) (o posterior) -> decode -> emojis nuevos (2.c)"),
        questionary.Choice("Plot del latente + generados", "plot",
                           description="nubes q(z|x) + medias + generados sobre un VAE entrenado (2.b)"),
    ]))

    if what == "ae":
        weights = _ask(_path("Pesos .npz del AE entrenado:", default="output/ae/"))
        data = _ask(_select("Dataset:", choices=["letters", "emoji"], default="letters"))
        sampling = _ask(_select("Muestreo:", choices=[
            questionary.Choice("normal", description="z ~ N(0,1): generación canónica"),
            questionary.Choice("bounds", description="uniforme dentro del rango latente ocupado"),
        ]))
        argv = ["generate", "ae", "--weights", weights, "--data", data, "--sampling", sampling]
        if _ask(_confirm("¿Plot del latente con los generados?", default=False)):
            argv += ["--plot"]
        _dispatch_main(argv + _seed_flag(cfg))

    elif what == "vae":
        weights = _ask(_path("Pesos .npz del VAE entrenado:", default="output/vae/"))
        data = _ask(_select("Dataset:", choices=["emoji", "letters"], default="emoji"))
        sampling = _ask(_select("Muestreo:", choices=[
            questionary.Choice("prior", description="z ~ N(0,I): generación canónica desde cero"),
            questionary.Choice("posterior", description="z ~ N(μ_i, σ_i): alrededor de patrones aprendidos"),
        ]))
        argv = ["generate", "vae", "--weights", weights, "--data", data, "--sampling", sampling]
        if _ask(_confirm("¿Grilla 2D del manifold?", default=False)):
            argv += ["--grid"]
        _dispatch_main(argv + _seed_flag(cfg))

    else:  # plot latent
        weights = _ask(_path("Pesos .npz del VAE entrenado:", default="output/vae/"))
        data = _ask(_select("Dataset:", choices=["emoji", "letters"], default="emoji"))
        sampling = _ask(_select("Muestreo:", choices=[
            questionary.Choice("normal", description="generados z ~ N(0,1)"),
            questionary.Choice("bounds", description="uniforme dentro del rango latente ocupado"),
        ]))
        argv = ["plot", "latent", "--weights", weights, "--data", data, "--sampling", sampling]
        _dispatch_main(argv + _seed_flag(cfg))


def _menu_study(cfg):
    kind = _ask(_select("Estudio:", choices=[
        questionary.Choice("Arquitectura", "architecture",
                           description="varía profundidad/ancho del encoder (bottleneck fijo en 2); AE puro"),
        questionary.Choice("Hiperparámetros", "hyperparams",
                           description="grid cruzado opt×lr×init (30 combos); AE puro"),
        questionary.Choice("Arquitectura (DAE)", "architecture-dae",
                           description="igual que arquitectura pero denoising; default salt 0.1, error vs limpio"),
        questionary.Choice("Hiperparámetros (DAE)", "hyperparams-dae",
                           description="grid opt×lr×init × salt 0.1; error vs limpio"),
        questionary.Choice("Barrido KL (β-VAE)", "kl",
                           description="VAE: trade-off reconstrucción vs generación según kl_weight"),
        questionary.Choice("Barrido denoising", "denoising",
                           description="DAE: error de reconstrucción vs nivel de ruido"),
    ]))

    grid_kinds = ("architecture", "hyperparams", "architecture-dae", "hyperparams-dae")
    if kind in grid_kinds:
        smoke = _ask(_confirm("¿Modo smoke (épocas/seeds bajas, ejes recortados)?", default=False))
        argv = ["study", kind]
        if smoke:
            argv += ["--smoke"]
        else:
            epochs = _ask(_text("Épocas (presupuesto del grid):", default="2000"))
            seeds = _ask(_text("Seeds por celda:", default="3"))
            argv += ["--epochs", epochs, "--seeds", seeds]
        _dispatch_main(argv)
        return

    # kl / denoising: barridos con su propio main.
    epochs = _ask(_text("Épocas:", default=str(cfg.epochs)))
    seeds = _ask(_text("Seeds por celda:", default="3"))
    argv = ["study", kind, "--epochs", epochs, "--seeds", seeds]
    if kind == "denoising":
        argv += ["--realizations", _ask(_text("Realizaciones por punto:", default="5"))]
    _dispatch_main(argv)


def run_tui():
    """Menú principal. Dispatcha solo; no devuelve nada útil."""
    _banner()
    cfg = load_config()
    handlers = {
        "Autoencoder (AE)": lambda: _menu_ae(cfg, denoising=False),
        "Denoising AE": lambda: _menu_ae(cfg, denoising=True),
        "VAE": lambda: _menu_vae(cfg),
        "Generar / Visualizar": lambda: _menu_generate(cfg),
        "Grid search / Estudios": lambda: _menu_study(cfg),
    }
    choices = [
        questionary.Choice("Autoencoder (AE)",
                           description="reconstruye las 32 letras; espacio latente 2D (a.1-a.4)"),
        questionary.Choice("Denoising AE",
                           description="entrada ruidosa, objetivo limpio; ruido re-sampleado (b.1-b.2)"),
        questionary.Choice("VAE",
                           description="variational sobre emojis: μ, logσ², KL (2.a-2.c)"),
        questionary.Choice("Generar / Visualizar",
                           description="muestras nuevas o plot del latente de un VAE entrenado"),
        questionary.Choice("Grid search / Estudios",
                           description="arquitectura, hiperparámetros y barridos comparativos"),
    ]
    try:
        choice = _ask(_select("¿Qué querés correr?", choices, instruction="(flechas + enter)"))
        handlers[choice]()
    except KeyboardInterrupt:
        print(_color("Cancelado.", MUTED))
