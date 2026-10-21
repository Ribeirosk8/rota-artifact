#!/usr/bin/env python3
"""Fig. 1 do P1 : a divergencia entre vazao e potencia.

DECISAO DE ENCODING, e ela E' a contribuicao:
  NAO usamos eixo duplo. Duas escalas arbitrarias no mesmo grafico permitem
  fabricar (ou esconder) divergencia so' escolhendo os limites -- e' a critica
  classica ao dual-axis. Em vez disso, normalizamos AMBAS as series ao seu valor
  no ponto de menor J/GB (16 MiB) e plotamos as duas no MESMO eixo adimensional.
  A divergencia passa a ser lida como razao, sem escala escolhida por nos.

  Consequencia: a distancia vertical entre as curvas E' o achado, e ela nao
  depende de nenhuma decisao estetica.

Cores: Okabe-Ito (CVD-safe). Vazao = azul, potencia = vermelho-alaranjado.
Barras de erro = desvio-padrao entre replicas (n de 3 a 11 por ponto).
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "curve_data.json")))

MIBS = [1, 4, 16, 20, 24, 28, 32, 40, 48, 64, 256, 1024]
REF = 16                      # ponto de referencia = minimo de J/GB

BLUE, VERM, CINZA = "#0072B2", "#D55E00", "#999999"

def series(rota, fill, key):
    xs, ys, es, ns = [], [], [], []
    for m in MIBS:
        k = f"{rota}|{fill}|{m}"
        if k not in D: continue
        xs.append(m); ys.append(D[k][key]); es.append(D[k][key + "_sd"]); ns.append(D[k]["n"])
    return xs, ys, es, ns

fig, ax = plt.subplots(figsize=(5.2, 1.95))

for key, cor, rot, marc in [("G", BLUE, "Throughput", "o"),
                            ("P", VERM, "Power (net)", "s")]:
    xs, ys, es, _ = series("nvlink", "const", key)
    ref = ys[xs.index(REF)]
    yn = [v / ref for v in ys]
    en = [e / ref for e in es]
    ax.errorbar(xs, yn, yerr=en, marker=marc, ms=4.5, lw=1.7, capsize=2.5,
                color=cor, label=rot, zorder=3)

ax.axhline(1.0, color=CINZA, lw=0.8, ls=":", zorder=1)
ax.axvline(REF, color=CINZA, lw=0.8, ls=":", zorder=1)

# AS DUAS FASES, rotuladas pelo que acontece com J/GB -- conferido no dado:
#   1 -> 16 MiB : vazao +113%, potencia  +11%  => J/GB MELHORA (0,757 -> 0,395)
#  16 -> 1024   : vazao  +10%, potencia  +47%  => J/GB PIORA   (0,395 -> 0,528)
# NOTE: NAO rotular a 1a fase de "coupled": as curvas NAO sobem juntas nela.
ax.axvspan(0.8, REF, color=CINZA, alpha=0.09, lw=0, zorder=0)
ax.text(3.5, 0.50, "J/GB improves", fontsize=8, color="#555555", ha="center")
ax.text(150, 0.50, "J/GB degrades", fontsize=8, color="#555555", ha="center")
ax.plot([REF], [1.0], marker="v", ms=7, color="black", zorder=5)
ax.text(REF, 1.07, "min J/GB", fontsize=8, ha="center", color="black")

# a anotacao E' a mensagem: onde as curvas se separam
xs, ys, _, _ = series("nvlink", "const", "G"); refg = ys[xs.index(REF)]
xs2, ys2, _, _ = series("nvlink", "const", "P"); refp = ys2[xs2.index(REF)]
gi, pi = ys[xs.index(1024)] / refg, ys2[xs2.index(1024)] / refp
ax.annotate("", xy=(1024, pi), xytext=(1024, gi),
            arrowprops=dict(arrowstyle="<->", color="black", lw=1.1))
ax.text(760, (gi + pi) / 2, f"{(pi/gi - 1)*100:.0f}%\ngap", ha="right",
        va="center", fontsize=8.5, linespacing=1.1)

ax.set_xscale("log", base=2)
ax.set_xticks([1, 4, 16, 64, 256, 1024])
ax.set_xticklabels(["1", "4", "16", "64", "256", "1024"])
ax.set_xlabel("Message size (MiB)", fontsize=9.5)
ax.set_ylabel(f"Relative to {REF} MiB", fontsize=9.5)
ax.tick_params(labelsize=8.5)
ax.legend(fontsize=8.5, frameon=False, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
# ylim inclui TODOS os pontos: cortar 1 e 4 MiB esconderia a fase acoplada
ax.set_ylim(0.42, 1.58)

fig.tight_layout(pad=0.4)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(HERE, f"fig1_divergence.{ext}"), dpi=200,
                bbox_inches="tight")

# --- o numero que a figura afirma, impresso para conferencia ---
print("Fig.1 : NVLink, const. Normalizado em 16 MiB:")
print(f"{'MiB':>6}{'GB/s':>9}{'norm':>8}{'W':>9}{'norm':>8}{'gap':>8}")
for m in MIBS:
    k = f"nvlink|const|{m}"
    if k not in D: continue
    g, p = D[k]["G"] / refg, D[k]["P"] / refp
    print(f"{m:>6}{D[k]['G']:>9.1f}{g:>8.3f}{D[k]['P']:>9.1f}{p:>8.3f}{(p/g-1)*100:>7.1f}%")
