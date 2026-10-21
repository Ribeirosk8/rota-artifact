#!/usr/bin/env python3
"""Fig. 2 do P1 : o modelo canonico de coletivos falha em energia.

O QUE A FIGURA AFIRMA:
  O custo de um coletivo e' modelado na literatura como T = alpha + beta*n
  (Wikipedia/Collective_operation da' O(alpha log p + beta n) para broadcast,
  reduce, all-reduce, scan; O(alpha log p + beta p n) para gather/scatter).
  Todas LINEARES em n => o custo POR BYTE decresce monotonicamente ate beta.
  Medimos um MINIMO seguido de subida: ha' um termo super-linear que o modelo
  nao possui.

DECISOES DE DESENHO, e cada uma tem motivo:

  (1) DOIS PAINEIS lado a lado (NVLink | SYS), nao 4 curvas sobrepostas.
      E' o idioma do campo -- Ismayilov (ICS'23) Figs. 5/6/8, Li et al. TPDS.
      Quatro curvas num painel so' obrigariam o leitor a desembaralhar duas
      variaveis (rota e entropia) ao mesmo tempo.

  (2) O MODELO E' CALIBRADO EM DOIS PONTOS (1 e 16 MiB), nao por minimos
      quadrados sobre todos. Ajustar em todos daria ao modelo a chance de
      espalhar o erro e esconder a falha ESTRUTURAL. Calibrar no regime onde
      ele e' valido (custo fixo dominante) e extrapolar e' o teste justo.
      NOTE: E' uma escolha, e vai declarada na legenda e no texto.

  (3) EIXO Y INDEPENDENTE por painel. NVLink e SYS diferem por ~4x em J/GB;
      um eixo comum achataria o NVLink e esconderia o efeito da entropia.
      O preco e' nao poder comparar alturas entre paineis a olho -- por isso a
      razao entre rotas vai no TEXTO, nao na figura.

Cores: Okabe-Ito (CVD-safe).
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "curve_data.json")))
MIBS = [1, 4, 16, 20, 24, 28, 32, 40, 48, 64, 256, 1024]

AZUL, VERM, CINZA = "#0072B2", "#D55E00", "#666666"

def serie(rota, fill):
    xs, ys, es = [], [], []
    for m in MIBS:
        k = f"{rota}|{fill}|{m}"
        if k in D:
            xs.append(m); ys.append(D[k]["J"]); es.append(D[k]["J_sd"])
    return xs, ys, es

def modelo(xs, ys, cal=(1, 16)):
    """J/GB = A/n + B, calibrado nos dois pontos onde o modelo DEVE valer."""
    i1, i2 = xs.index(cal[0]), xs.index(cal[1])
    A = (ys[i1] - ys[i2]) / (1 / cal[0] - 1 / cal[1])
    B = ys[i1] - A / cal[0]
    return A, B

fig, axes = plt.subplots(1, 2, figsize=(6.0, 1.9))

for ax, rota, titulo in zip(axes, ["nvlink", "sys"],
                            ["NVLink  (1 hop)", "SYS  (PCIe$\\to$UPI$\\to$PCIe, 3 hops)"]):
    xc, yc, ec = serie(rota, "const")
    A, B = modelo(xc, yc)
    fino = [1 * (1.06 ** i) for i in range(int(__import__("math").log(1024, 1.06)) + 1)]
    ax.plot(fino, [A / n + B for n in fino], color=CINZA, lw=1.4, ls="--",
            label=r"model $\alpha/n+\beta$", zorder=2)

    for fill, cor, marc in [("const", AZUL, "o"), ("random", VERM, "s")]:
        xs, ys, es = serie(rota, fill)
        ax.errorbar(xs, ys, yerr=es, marker=marc, ms=4, lw=1.6, capsize=2,
                    color=cor, label=fill, zorder=3)

    # sombreia onde o modelo erra >10%
    falha = [n for n in xc if abs(yc[xc.index(n)] - (A / n + B)) / (A / n + B) > 0.10]
    if falha:
        ax.axvspan(min(falha) * 0.87, 1024 * 1.35, color=CINZA, alpha=0.10, lw=0, zorder=0)
        pior = max(falha, key=lambda n: (yc[xc.index(n)] - (A / n + B)) / (A / n + B))
        err = 100 * (yc[xc.index(pior)] - (A / pior + B)) / (A / pior + B)
        ax.text(0.97, 0.06, f"model off by\nup to {err:.0f}%", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=10, color="black", linespacing=1.15)

    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 4, 16, 64, 256, 1024])
    ax.set_xticklabels(["1", "4", "16", "64", "256", "1024"])
    ax.set_xlim(0.75, 1500)
    ax.set_xlabel("Message size (MiB)", fontsize=11)
    ax.set_title(titulo, fontsize=11, pad=4)
    ax.tick_params(labelsize=10)
    ax.spines[["top", "right"]].set_visible(False)

axes[0].set_ylabel("Energy per byte (J/GB)", fontsize=11)
axes[0].legend(fontsize=9.5, frameon=False, loc="upper center", ncol=1)

fig.tight_layout(pad=0.5, w_pad=1.5)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(HERE, f"fig2_model.{ext}"), dpi=200, bbox_inches="tight")

# --- os numeros que a figura afirma ---
for rota in ["nvlink", "sys"]:
    xc, yc, _ = serie(rota, "const")
    A, B = modelo(xc, yc)
    print(f"\n{rota.upper():7} modelo J/GB = {A:.4f}/n + {B:.4f}  (calibrado em 1 e 16 MiB)")
    print(f"{'MiB':>6}{'medido':>10}{'modelo':>10}{'erro':>9}")
    for n in xc:
        med = yc[xc.index(n)]; pre = A / n + B
        print(f"{n:>6}{med:>10.4f}{pre:>10.4f}{100*(med-pre)/pre:>8.1f}%")
