#!/bin/bash
#SBATCH --job-name=mcpp_cache
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:45:00
#SBATCH --output=mcpp_cache_%j.out
#SBATCH --error=mcpp_cache_%j.err
#
# ============================================================================
#  O TESTE QUE DECIDE: capacidade de CACHE ou tamanho de MENSAGEM?
# ============================================================================
#  O job 136281 mediu, no NVLink, de 16 para 64 MiB:
#      potencia  +44,2%  (95,5 -> 137,7 W)
#      vazao      +8,1%
#      clock      CONSTANTE em 1560 MHz
#  Duas explicacoes concorrem, e elas fazem previsoes OPOSTAS:
#
#   (a) CAPACIDADE DE CACHE. O laco reusa o mesmo buffer; se ele cabe no L2, a
#       iteracao seguinte le do cache. O que importa e' o WORKING SET.
#   (b) TAMANHO DA MENSAGEM. Algo na transferencia de 64 MiB custa mais por
#       byte que a de 16 MiB, independentemente de reuso.
#
#  `--buffers N` separa as duas: mantem a MENSAGEM em 16 MiB e faz o WORKING
#  SET virar N*16 MiB, rodiziando entre N buffers distintos.
#
#  NOTE: PREVISAO REGISTRADA ANTES DE RODAR (ver run_136281/VEREDITO.md):
#      16 MiB x 1 buffer  -> ~95 W   (ja' medido)
#      16 MiB x 4 buffers -> ~137 W  SE for cache
#                         -> ~95 W   SE for tamanho de mensagem
#  Qualquer valor intermediario e' resultado parcial e sera' reportado como tal.
#
#  CONTROLE: 64 MiB x 1 buffer roda no MESMO job, para ancorar os ~137 W nas
#  condicoes de hoje em vez de comparar com outro job.
# ============================================================================
set -uo pipefail
OUT="$HOME/cluster-redacted/mcpp_cache_${SLURM_JOB_ID}"; mkdir -p "$OUT"
PROBE="$HOME/cluster-redacted/route_probe"
BENCH="$HOME/cluster-redacted/route_bench3"
for b in "$PROBE" "$BENCH"; do
  [[ -x "$b" ]] || { echo "FATAL: $b ausente."; exit 2; }
done

REPS=3
WARMUP=20
MEASURE=60
IDLE=15
SEED="${MCPP_SEED:-20260801}"

echo "==> $SLURM_JOB_ID em $(hostname -s) $(date -Is)"
srun --gpu-bind=none nvidia-smi topo -m < /dev/null > "$OUT/topo.txt" 2>&1
grep -qE '^GPU0.*NV[0-9]+' "$OUT/topo.txt" || { echo "FATAL: sem NVLink."; exit 3; }

# plano: (mib, buffers). working_set = mib*buffers
PLANO="$OUT/plano.txt"; : > "$PLANO"
for r in $(seq 1 $REPS); do
  echo "16 1 $r"  >> "$PLANO"   # base    : ws=16   -> esperado ~95 W
  echo "16 2 $r"  >> "$PLANO"   # ws=32                dose-resposta
  echo "16 4 $r"  >> "$PLANO"   # ws=64   -> CHAVE: ~137 W se for cache
  echo "16 8 $r"  >> "$PLANO"   # ws=128               satura?
  echo "64 1 $r"  >> "$PLANO"   # controle: ws=64   -> ancora os ~137 W
done
shuf --random-source=<(yes "$SEED") "$PLANO" -o "$PLANO"
N_PLAN=$(wc -l < "$PLANO")
echo "=== $N_PLAN janelas (seed=$SEED) ==="

snap () { srun --gpu-bind=none "$PROBE" < /dev/null 2>/dev/null \
          | sed "s/^/${1},/" >> "$OUT/energy.csv"; }

n=0
while read -r mib bufs rep <&9; do
  n=$((n+1))
  tag="m${mib}_b${bufs}_r${rep}"
  printf "\n--- [%d/%d] %s  (working set = %d MiB)\n" "$n" "$N_PLAN" "$tag" "$((mib*bufs))"
  snap "idlepre_${tag}_t0"; sleep "$IDLE"; snap "idlepre_${tag}_t1"
  srun --gpu-bind=none "$BENCH" --src 0 --dst 1 --mib "$mib" --buffers "$bufs" \
       --seconds "$WARMUP" --fill const < /dev/null > /dev/null 2>&1
  snap "${tag}_t0"
  linha=$(srun --gpu-bind=none "$BENCH" --src 0 --dst 1 --mib "$mib" --buffers "$bufs" \
               --seconds "$MEASURE" --fill const < /dev/null 2>&1 | tail -1)
  snap "${tag}_t1"
  echo "$tag,$linha" >> "$OUT/bench.csv"; echo "    $linha"
  by=$(echo "$linha" | cut -d, -f13); el=$(echo "$linha" | cut -d, -f15)
  [[ -n "$by" && "$by" != "0" ]] || { echo "FATAL: $tag moveu 0 bytes."; exit 4; }
  awk -v e="$el" 'BEGIN{exit !(e < 0.9*'"$MEASURE"')}' \
    && { echo "FATAL: $tag rodou ${el}s."; exit 5; }
done 9< "$PLANO"

N_EXEC=$(wc -l < "$OUT/bench.csv")
echo; echo "=== completude: $N_EXEC de $N_PLAN ==="
[[ "$N_EXEC" -eq "$N_PLAN" ]] || { echo "FATAL: $N_EXEC de $N_PLAN."; exit 7; }
echo "==> artefatos em $OUT"; echo "==> FIM $(date -Is)"
