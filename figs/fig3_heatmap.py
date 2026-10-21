#!/usr/bin/env python3
"""Fig. 3 do P1 : o campo J/GB sobre rota x tamanho, com regua razao-vs-mediana.

O QUE A FIGURA MOSTRA: onde o custo esta', num campo bidimensional
(rota x tamanho de mensagem x entropia), com uma regua que permite comparar
celulas entre linhas cujas magnitudes diferem por ~4x.

AS TRES DECISOES DE ENCODING, e cada uma tem fonte:

  (1) REGUA RAZAO-VS-MEDIANA, nao min-max por painel.
      A celula carrega log2(J/GB / mediana da linha). Eisen et al. (PNAS 1998)
      estabeleceram esta regua para expressao genica pelo mesmo motivo que ela
      serve aqui: linhas com magnitudes muito diferentes precisam ser
      comparaveis em FORMA sem que a de maior magnitude domine a rampa.
      NOTE: Eisen SATURA em ±3 (fator 8). Nossa faixa util e' muito menor, entao
      usamos compressao tanh -- satura suave em vez de cortar.

  (2) DIVERGENTE com ponto neutro na mediana, nao sequencial.
      O zero (= mediana da rota) e' um valor com significado: "o custo tipico
      desta rota". Moreland (ISVC'09) e a regra da casa: mapa divergente exige
      ponto medio ancorado em valor significativo, nunca no meio do intervalo.

  (3) NENHUM ARCO-IRIS. Luminancia monotonica em cada metade.
      Rainbow/jet nao tem luminancia monotonica e fabrica fronteiras onde o
      dado e' liso.

NOTE: O QUE A FIGURA NAO PODE FAZER: comparar magnitude ABSOLUTA entre as duas
   rotas. A regua e' por linha, de proposito. A razao NVLink-vs-SYS (~4x) vai no
   TEXTO e na Fig. 2, nao aqui. Anotamos a mediana de cada linha para que o
   leitor possa reconstruir o valor absoluto.
"""
import json, os, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "curve_data.json")))
MIBS = [1, 4, 16, 20, 24, 28, 32, 40, 48, 64, 256, 1024]
LINHAS = [("nvlink", "const"), ("nvlink", "random"),
          ("sys", "const"), ("sys", "random")]
ROTULO = {("nvlink", "const"): "NVLink · const", ("nvlink", "random"): "NVLink · random",
          ("sys", "const"): "SYS · const",       ("sys", "random"): "SYS · random"}

# --- monta a matriz e a regua ---
M = np.full((len(LINHAS), len(MIBS)), np.nan)
ABS = np.full_like(M, np.nan)
medianas = []
for i, (rota, fill) in enumerate(LINHAS):
    vals = [D[f"{rota}|{fill}|{m}"]["J"] if f"{rota}|{fill}|{m}" in D else np.nan
            for m in MIBS]
    med = float(np.nanmedian(vals))
    medianas.append(med)
    for j, v in enumerate(vals):
        ABS[i, j] = v
        if not math.isnan(v):
            # razao-vs-mediana em log2, com compressao tanh (satura suave)
            M[i, j] = math.tanh(math.log2(v / med) / 0.45) * 0.45

VMAX = float(np.nanmax(np.abs(M)))
fig, ax = plt.subplots(figsize=(6.4, 1.55))
im = ax.imshow(M, cmap="RdBu_r", vmin=-VMAX, vmax=VMAX, aspect="auto")

for i in range(len(LINHAS)):
    for j in range(len(MIBS)):
        if math.isnan(ABS[i, j]): continue
        # texto claro sobre celula escura, escuro sobre clara
        forte = abs(M[i, j]) > 0.62 * VMAX
        ax.text(j, i, f"{ABS[i,j]:.2f}", ha="center", va="center", fontsize=6.4,
                color="white" if forte else "#222222")

ax.set_xticks(range(len(MIBS)))
ax.set_xticklabels([str(m) for m in MIBS], fontsize=7.5)
ax.set_yticks(range(len(LINHAS)))
ax.set_yticklabels([f"{ROTULO[k]}\n(med {medianas[i]:.2f})" for i, k in enumerate(LINHAS)],
                   fontsize=7.2)
ax.set_xlabel("Message size (MiB)", fontsize=9)
ax.tick_params(length=0)
for s in ax.spines.values(): s.set_visible(False)

cb = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.036)
cb.set_label("$\\log_2$(J/GB ÷ row median), tanh-compressed", fontsize=7.5)
cb.ax.tick_params(labelsize=7)

fig.tight_layout(pad=0.35)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(HERE, f"fig3_heatmap.{ext}"), dpi=200, bbox_inches="tight")

# --- diagnostico da regua: ela usa a rampa ou deixa tudo liso? ---
print("REGUA razao-vs-mediana + tanh : diagnostico")
print(f"{'linha':<18}{'mediana':>9}{'min':>8}{'max':>8}{'spread':>9}{'ocupacao':>10}")
for i, k in enumerate(LINHAS):
    v = ABS[i][~np.isnan(ABS[i])]
    occ = (np.nanmax(M[i]) - np.nanmin(M[i])) / (2 * VMAX)
    print(f"{ROTULO[k]:<18}{medianas[i]:>9.3f}{v.min():>8.3f}{v.max():>8.3f}"
          f"{v.max()/v.min():>8.2f}x{100*occ:>9.1f}%")
print(f"\n  saturacao (celulas em |M|>0.95*VMAX): "
      f"{100*np.nanmean(np.abs(M) > 0.95*VMAX):.1f}%")
