#!/bin/bash
#SBATCH --job-name=mcpp_nccl
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --output=mcpp_nccl_%j.out
#SBATCH --error=mcpp_nccl_%j.err
#
# ============================================================================
#  A CARGA RECONHECIVEL: NCCL AllReduce sobre 4 GPUs
# ============================================================================
#  POR QUE EXISTE: o `route_bench` isola a variavel (rota, entropia, tamanho)
#  porque nada mais varia -- e' o INSTRUMENTO. Mas ele nao e' carga que um
#  revisor reconheca. Este job responde "e numa carga de verdade?" usando
#  `nccl-tests`, o benchmark canonico da NVIDIA, ja' compilado no cluster.
#
#  NOTE: DELIBERADAMENTE NAO escrevemos um AllReduce proprio. Usar a ferramenta
#  padrao e' mais defensavel que a nossa, e o ponto aqui e' RECONHECIMENTO, nao
#  controle -- o controle ja' vem do route_bench.
#
#  NOTE: O QUE ESTE JOB NAO PODE FAZER: dizer por onde o byte passou. O NCCL
#  escolhe canais, algoritmos e transportes sozinho (Li et al., TPDS'20:
#  "NCCL-V2 is closed-source... introduces difficulty in independently
#  measuring the CL communication efficiency for a particular interconnect").
#  Por isso ele VALIDA, nao MEDE a rota. A rota vem do contador, no outro job.
#
#  MESMOS CONTROLES do resto da campanha: aquecimento descartado, ordem
#  aleatorizada, replicas, idle antes de cada braco, assercao de completude.
# ============================================================================
set -uo pipefail

OUT="$HOME/cluster-redacted/mcpp_nccl_${SLURM_JOB_ID}"; mkdir -p "$OUT"
PROBE="$HOME/cluster-redacted/route_probe3"   # v3: registra epoch_ns (ver 138091)
NT="$HOME/cluster-redacted/nccl-tests/build"
[[ -x "$PROBE" ]] || { echo "FATAL: $PROBE ausente."; exit 2; }
[[ -x "$NT/all_reduce_perf" ]] || { echo "FATAL: all_reduce_perf ausente."; exit 2; }

# NOTE: O 137926 falhou em 32s: "libcudart.so.12: cannot open shared object file".
# O nccl-tests foi compilado contra o CUDA/NCCL do spack, mas o job nao herda o
# LD_LIBRARY_PATH. A assercao pegou (exit 4, 32 s em vez de 12 janelas vazias).
SPACK=/opt/apps/nfs/spack-v0.23/opt/spack/linux-rocky9-sapphirerapids/gcc-11.4.1
CUDA_LIB=$SPACK/cuda-12.6.2-glbfe66lql5geaqhi6vfmly7i3zmqmhl/lib64
NCCL_LIB=$SPACK/nccl-2.22.3-1-6yd74t2k2yp5rlnh66usl575gh2ojawa/lib
export LD_LIBRARY_PATH="$CUDA_LIB:$NCCL_LIB:${LD_LIBRARY_PATH:-}"
# pre-check: o binario RODA? Falhar aqui custa 5 s; falhar no meio custa a fila.
"$NT/all_reduce_perf" -b 8 -e 8 -g 1 -n 1 -w 0 > /tmp/nccl_pre_$$ 2>&1 || {
  echo "FATAL: all_reduce_perf nao executa. Saida:"; cat /tmp/nccl_pre_$$; exit 3; }
echo "==> pre-check OK: all_reduce_perf executa"

# NOTE: INVARIANTE DE PILHA (05/08). O nccl-tests imprime a versao da biblioteca
# que REALMENTE carregou ("nccl-library=22203" = NCCL 2.22.3). Nunca exigimos
# isso, e em 05/08 um `conda install libnvshmem-dev` trouxe NCCL 2.30.7.1 para
# o cuda-env. O prepend do spack acima protege, MAS proteger sem verificar e'
# torcer. Comparar coletivos medidos em versoes diferentes de NCCL seria um
# confundidor silencioso: a versao escolhe algoritmo, protocolo e transporte.
NCCL_ESPERADA="${MCPP_NCCL_LIB:-22203}"
NCCL_REAL=$(grep -o "nccl-library=[0-9]*" /tmp/nccl_pre_$$ | head -1 | cut -d= -f2)
echo "==> NCCL carregado: ${NCCL_REAL:-<nao reportado>} (esperado $NCCL_ESPERADA)"
[[ "$NCCL_REAL" == "$NCCL_ESPERADA" ]] || {
  echo "FATAL: NCCL $NCCL_REAL != $NCCL_ESPERADA da campanha."
  echo "       Rode com MCPP_NCCL_LIB=$NCCL_REAL se a troca for DELIBERADA."
  exit 8; }

REPS="${MCPP_REPS:-3}"
IDLE=15
SEED="${MCPP_SEED:-20260802}"
# tamanhos que CASAM com a curva do route_bench, para comparacao direta
SIZES="${MCPP_SIZES:-16M 64M 256M 1G}"
# NOTE: ITERS ERA HARDCODED (-n 200). O job 140010 recebeu MCPP_ITERS=2000 no
# --export, rodou com 200 assim mesmo, e terminou em 6m20 contra 6m41 do
# 138134 -- identico, porque a variavel nunca era lida. Mesmo erro do braco C:
# o NOME do job prometeu o que o SCRIPT nao entregava.
ITERS="${MCPP_ITERS:-200}"
# NOTE: ITERACOES POR TAMANHO. O 140022 morreu certo: com n global=2000, 64M da'
# 45% de fracao (setup ~4,5s fixos vs 2,7ms/iter). O n certo depende do
# tamanho: mira ~30s de transferencia por janela => fracao ~87%.
iters_for () { case "$1" in
  16M)  echo 54000 ;;
  64M)  echo 11000 ;;
  256M) echo 2800  ;;
  1G)   echo 700   ;;
  *)    echo "$ITERS" ;;
esac; }

echo "==> $SLURM_JOB_ID em $(hostname -s) $(date -Is)"
echo "==> SIZES=[$SIZES] REPS=$REPS SEED=$SEED ITERS=$ITERS"
srun --gpu-bind=none nvidia-smi topo -m < /dev/null > "$OUT/topo.txt" 2>&1

PLANO="$OUT/plano.txt"; : > "$PLANO"
for r in $(seq 1 "$REPS"); do
  for s in $SIZES; do echo "$s $r" >> "$PLANO"; done
done
shuf --random-source=<(yes "$SEED") "$PLANO" -o "$PLANO"
N_PLAN=$(wc -l < "$PLANO")
echo "=== $N_PLAN janelas ==="

snap () { srun --gpu-bind=none "$PROBE" < /dev/null 2>/dev/null \
          | sed "s/^/${1},/" >> "$OUT/energy.csv"; }

n=0
while read -r sz rep <&9; do
  n=$((n+1)); tag="allreduce_${sz}_r${rep}"
  printf "\n--- [%d/%d] %s\n" "$n" "$N_PLAN" "$tag"

  snap "idlepre_${tag}_t0"; sleep "$IDLE"; snap "idlepre_${tag}_t1"

  # aquecimento descartado (mesma regra do route_bench)
  srun --gpu-bind=none "$NT/all_reduce_perf" -b "$sz" -e "$sz" -g 4 -n 20 -w 5 \
       < /dev/null > /dev/null 2>&1

  w0=$(date +%s.%N); snap "${tag}_t0"
  # -n 200 iteracoes para encher a janela; -w 5 warmup interno do proprio teste
  srun --gpu-bind=none "$NT/all_reduce_perf" -b "$sz" -e "$sz" -g 4 -n "$(iters_for "$sz")" -w 5 \
       < /dev/null > "$OUT/${tag}.txt" 2>&1
  snap "${tag}_t1"; wall=$(awk -v a="$w0" -v b="$(date +%s.%N)" 'BEGIN{print b-a}')

  # a linha de dado do nccl-tests: size count type redop root time algbw busbw #wrong
  linha=$(grep -E "^\s+[0-9]+" "$OUT/${tag}.txt" | tail -1)
  echo "$tag,$(echo $linha | tr -s ' ' ',')" >> "$OUT/bench.csv"
  echo "    $linha"
  [[ -n "$linha" ]] || { echo "FATAL: $tag sem linha de dado."; exit 4; }
  # NOTE: ASSERCAO DE FRACAO MEDIDA (a que faltou no 138134): se a transferencia
  # for minoria da janela, o J/GB sai diluido por fator variavel e o dado nao
  # serve. Piso de 60%. Declarar a fracao SEMPRE, abortar abaixo do piso.
  # NOTE: O all_reduce_perf roda CADA janela DUAS vezes: out-of-place ($6) e
  # in-place ($10), com n iteracoes cada. A 1a versao desta formula contava
  # so' a oop e ABORTOU um dado BOM (140030: 46% aparente = 92% real).
  # Consequencia p/ analise: bytes logicos da janela = 2*n*size, nao n*size.
  us=$(echo "$linha" | awk '{print $6+$10}')
  frac=$(awk -v u="$us" -v n="$(iters_for "$sz")" -v w="$wall" 'BEGIN{printf "%.1f", 100*u*1e-6*n/w}')
  echo "    fracao transferindo: ${frac}% da janela"
  awk -v f="$frac" 'BEGIN{exit !(f < 60)}' && {
    echo "FATAL: so' ${frac}% da janela e' transferencia (piso 60%)."
    echo "       J/GB sairia diluido. Aumente --iters."; exit 6; }
  # o nccl-tests reporta #wrong; qualquer coisa != 0 invalida a janela
  wrong=$(echo "$linha" | awk '{print $NF}')
  [[ "$wrong" == "0" || "$wrong" == "N/A" ]] || { echo "FATAL: $tag #wrong=$wrong"; exit 5; }
done 9< "$PLANO"

N_EXEC=$(wc -l < "$OUT/bench.csv")
echo; echo "=== completude: $N_EXEC de $N_PLAN ==="
[[ "$N_EXEC" -eq "$N_PLAN" ]] || { echo "FATAL: $N_EXEC de $N_PLAN."; exit 7; }
echo "==> artefatos em $OUT"; echo "==> FIM $(date -Is)"
