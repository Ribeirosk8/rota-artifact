#!/bin/bash
#SBATCH --job-name=mcpp_curve
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --output=mcpp_curve_%j.out
#SBATCH --error=mcpp_curve_%j.err
#
# ============================================================================
#  mercury-cpp : VARREDURA DE TAMANHO: a curva J/GB x tamanho de mensagem
# ============================================================================
#  O que este job entrega e os anteriores nao entregaram:
#    - a CURVA J/GB x tamanho, para as 2 rotas -> a figura-tese
#    - o joelho de saturacao (Li et al. TPDS mede ~4 MB para P2P)
#    - o efeito da ENTROPIA com poder estatistico (n grande por condicao)
#    - C1 com dado proprio, em varios pontos em vez de um
#
#  HERANCA DOS ERROS ANTERIORES -- os tres controles estao aqui:
#   1. `srun ... < /dev/null` e leitura do plano por DESCRITOR 9.
#      Sem isso o srun drena o stdin e o laco morre na 1a iteracao: foi o que
#      fez o 135896 rodar 1 de 12 bracos reportando COMPLETED exit 0.
#   2. AQUECIMENTO descartado antes de t0. Sem isso o 1o braco roda com a GPU
#      em clock ocioso e a comparacao mede DVFS, nao a variavel (job 135723).
#   3. ASSERCAO DE COMPLETUDE no fim. As assercoes por braco verificam que
#      cada braco executado esta' bom; NAO que todos executaram.
#
#  NOTE: --exclusive e' passado na LINHA DE COMANDO. O job 135951 mediu ruido 3x
#  maior no SYS (±3,24%) que no NVLink (±1,22%): o caminho SYS atravessa UPI e
#  root complex, compartilhados com outros jobs; o NVLink nao atravessa nada
#  compartilhado. Mais replicas nao removem vies de vizinho -- exclusividade sim.
# ============================================================================
set -uo pipefail

OUT="$HOME/cluster-redacted/mcpp_curve_${SLURM_JOB_ID}"; mkdir -p "$OUT"
# NOTE: probe3, e a troca de UMA letra decide o numero do paper.
#
# O `route_probe` (sem 3) NAO grava `TIME/epoch_ns`. Sem carimbo nao ha' Δt, sem
# Δt nao ha' P_idle medido, e `E_liq = ΔE − P_idle·Δt` passa a depender de uma
# CONSTANTE: o `viz/data.py` usa `IDLE_S = 15.0`, que e' o sleep declarado no
# script e nao o intervalo real. Medido em 06/08 no hostC_140009 (o unico job
# com carimbo): o intervalo real e' 15,262 s (faixa 15,252-15,283), n=27.
#
# O efeito no numero que o P1 publica: o toggle no NVLink@64 vale +7,6% na base
# bruta da rota e +14,2% na base liquida -- e a base liquida e' a que o
# `main.tex:324` cita. Com a sonda certa, `jgb_liq` deixa de repousar num
# segundo imputado e passa a ser medido nas 96 janelas.
PROBE="$HOME/cluster-redacted/route_probe3"
BENCH="$HOME/cluster-redacted/route_bench"
for b in "$PROBE" "$BENCH"; do
  [[ -x "$b" ]] || { echo "FATAL: $b ausente ou nao executavel."; exit 2; }
done

MIBS="${MCPP_MIBS:-1 4 16 64 256 1024}"
REPS="${MCPP_REPS:-4}"
WARMUP="${MCPP_WARMUP:-20}"
MEASURE="${MCPP_MEASURE:-60}"
IDLE="${MCPP_IDLE:-15}"
SEED="${MCPP_SEED:-20260801}"

echo "==> job $SLURM_JOB_ID em $(hostname -s) $(date -Is)"
echo "==> MIBS=[$MIBS] REPS=$REPS WARMUP=${WARMUP}s MEASURE=${MEASURE}s SEED=$SEED"
[[ -n "${SLURM_JOB_EXCLUSIVE:-}" ]] && echo "==> alocacao EXCLUSIVA" \
  || echo "==> NOTE: NAO exclusiva: UPI/root-complex podem estar contaminados"

srun --gpu-bind=none nvidia-smi --query-gpu=index,name,pci.bus_id --format=csv \
     < /dev/null > "$OUT/gpus.txt" 2>&1
srun --gpu-bind=none nvidia-smi topo -m < /dev/null > "$OUT/topo.txt" 2>&1
echo "-- topologia:"; head -6 "$OUT/topo.txt"

# assercao: o par NVLink e' mesmo (0,1)? Se nao, o desenho inteiro e' invalido.
grep -qE '^GPU0.*NV[0-9]+' "$OUT/topo.txt" || {
  echo "FATAL: GPU0 sem vizinho NVLink. Desenho invalido."; exit 3; }

# -- plano: MIB x rota x entropia x replica, EMBARALHADO -------------------
PLANO="$OUT/plano.txt"; : > "$PLANO"
for m in $MIBS; do
  for r in $(seq 1 "$REPS"); do
    for f in const random; do
      echo "nvlink $f 0 1 $m $r" >> "$PLANO"
      echo "sys    $f 0 2 $m $r" >> "$PLANO"
    done
  done
done
shuf --random-source=<(yes "$SEED") "$PLANO" -o "$PLANO"
N_PLAN=$(wc -l < "$PLANO")
echo; echo "=== plano: $N_PLAN janelas (seed=$SEED) ==="
awk '{print $1, $2, $5"MiB", "r"$6}' "$PLANO" | head -8; echo "  ..."

snap () { srun --gpu-bind=none "$PROBE" < /dev/null 2>/dev/null \
          | sed "s/^/${1},/" >> "$OUT/energy.csv"; }

# -- FALHAR NO MINUTO 1, NAO NO MINUTO 120 ---------------------------------
# Esta campanha existe POR CAUSA do carimbo. Se a sonda nao o gravar, as 96
# janelas saem sem Δt e o job entrega exatamente o dado que veio corrigir --
# custando ~2 h de fila para descobrir. Um retrato de teste responde em 5 s.
snap "sonda_teste_t0"
if ! grep -q "TIME,epoch_ns" "$OUT/energy.csv"; then
  echo "FATAL: $PROBE nao gravou TIME/epoch_ns. Esta campanha existe para"
  echo "       produzir o carimbo; sem ele ela repete o defeito que veio"
  echo "       corrigir. Confira se o binario e' o route_probe3."
  exit 9
fi
echo "==> carimbo de tempo confirmado na sonda"
: > "$OUT/energy.csv"   # descarta o retrato de teste: nao e' medida de braco

n=0
while read -r rota fill s d mib rep <&9; do
  n=$((n+1))
  tag="${rota}_${fill}_m${mib}_r${rep}"
  printf "\n--- [%d/%d] %s\n" "$n" "$N_PLAN" "$tag"

  snap "idlepre_${tag}_t0"; sleep "$IDLE"; snap "idlepre_${tag}_t1"

  srun --gpu-bind=none "$BENCH" --src "$s" --dst "$d" --mib "$mib" \
       --seconds "$WARMUP" --fill "$fill" < /dev/null > /dev/null 2>&1

  snap "${tag}_t0"
  linha=$(srun --gpu-bind=none "$BENCH" --src "$s" --dst "$d" --mib "$mib" \
               --seconds "$MEASURE" --fill "$fill" < /dev/null 2>&1 | tail -1)
  snap "${tag}_t1"
  echo "$tag,$linha" >> "$OUT/bench.csv"
  echo "    $linha"

  bytes=$(echo "$linha" | cut -d, -f9); elapsed=$(echo "$linha" | cut -d, -f10)
  [[ -n "$bytes" && "$bytes" != "0" ]] || { echo "FATAL: $tag moveu 0 bytes."; exit 4; }
  awk -v e="$elapsed" 'BEGIN{exit !(e < 0.9*'"$MEASURE"')}' \
    && { echo "FATAL: $tag rodou ${elapsed}s, esperado ~${MEASURE}s."; exit 5; }
done 9< "$PLANO"

# -- ASSERCAO DE COMPLETUDE ------------------------------------------------
N_EXEC=$(wc -l < "$OUT/bench.csv")
echo; echo "=== completude: $N_EXEC de $N_PLAN ==="
[[ "$N_EXEC" -eq "$N_PLAN" ]] || { echo "FATAL: $N_EXEC de $N_PLAN bracos."; exit 7; }
[[ "$n" -eq "$N_PLAN" ]] || { echo "FATAL: laco iterou $n de $N_PLAN."; exit 8; }

echo; echo "==> artefatos em $OUT"; ls -la "$OUT"
echo "==> FIM $(date -Is)"
