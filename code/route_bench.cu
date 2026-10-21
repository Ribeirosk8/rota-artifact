/* route_bench.cu : mover bytes por UMA rota escolhida, com entropia escolhida.
 *
 * POR QUE ESTE EXISTE, e por que nao e o benchmark antigo:
 *
 *   1. ROTA INEQUIVOCA. O antigo era PyTorch/NCCL: o NCCL escolhe canais,
 *      algoritmos e transportes sozinho, e o conversor tinha que ADIVINHAR os
 *      pares src->dst a partir do algoritmo. Aqui a rota e um argumento:
 *      `--src 0 --dst 1` move de GPU0 para GPU1 e nada mais.
 *      (Li et al., TPDS 2020, dizem o mesmo por outro caminho: "NCCL-V2 is
 *      closed-source ... introduces difficulty in independently measuring the
 *      CL communication efficiency for a particular interconnect".)
 *
 *   2. ENTROPIA CONTROLADA. Adhinarayanan (IISWC'16) mede que energia de
 *      interconexao e LINEAR no toggle rate, e que zeros custam ~10% do que
 *      custa dado aleatorio. O antigo usava `torch.full(..., rank+1)` -- um
 *      padrao de 4 bytes repetido, toggle ~0. Aqui `--fill` e explicito.
 *      MEDIDO no job 135951: no NVLink, `random` custa +14,3% mais energia por
 *      byte que `const`, com vazao identica. A ameaca e' real.
 *
 *   3. SEM CAMADA. Nem Python, nem NCCL, nem MPI. Menos software entre a
 *      intencao e o fio significa menos coisa para explicar quando o numero
 *      surpreender.
 *
 * NAO mede energia: quem mede e o `route_probe`, chamado fora, em t0/t1. Este
 * so move bytes e diz quantos e por quanto tempo. Separacao deliberada -- o que
 * move o dado nao deve ser o que o cronometra.
 *
 * build: make -C src
 * uso:   ./route_bench --src 0 --dst 1 --mib 256 --seconds 60 --fill random
 *        ./route_bench --src 0 --dst 2 --route host --hostmem pinned
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <ctime>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) {                 \
    fprintf(stderr, "FATAL %s:%d  %s -> %s\n", __FILE__, __LINE__, #x,          \
            cudaGetErrorString(e)); return 2; } } while (0)

/* Preenche o buffer no HOST e copia uma vez. O conteudo e o experimento, entao
 * ele nasce aqui e nao num kernel -- um kernel de preenchimento teria seu
 * proprio custo energetico dentro da janela que queremos medir. */
static void fill_buffer(uint32_t *h, size_t n, const char *mode, unsigned seed) {
    if (!strcmp(mode, "zero")) {
        memset(h, 0, n * sizeof(uint32_t));                    /* toggle ~0 */
    } else if (!strcmp(mode, "const")) {
        for (size_t i = 0; i < n; i++) h[i] = 0x3F800000u;     /* 1.0f, o do antigo */
    } else {                                                    /* random */
        uint64_t s = seed ? seed : 0x9E3779B97F4A7C15ull;       /* toggle ~0.5 */
        for (size_t i = 0; i < n; i++) {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;            /* xorshift64 */
            h[i] = (uint32_t)(s >> 32);
        }
    }
}

int main(int argc, char **argv) {
    int src = 0, dst = 1, mib = 256, seconds = 60, buffers = 1;
    const char *fill = "random", *route = "peer", *hostmem = "pinned";
    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--src")     && i+1 < argc) src     = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dst")     && i+1 < argc) dst     = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--mib")     && i+1 < argc) mib     = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seconds") && i+1 < argc) seconds = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--fill")    && i+1 < argc) fill    = argv[++i];
        else if (!strcmp(argv[i], "--route")   && i+1 < argc) route   = argv[++i];
        else if (!strcmp(argv[i], "--hostmem") && i+1 < argc) hostmem = argv[++i];
        else if (!strcmp(argv[i], "--buffers") && i+1 < argc) buffers = atoi(argv[++i]);
    }
    if (buffers < 1) { fprintf(stderr, "FATAL: --buffers >= 1\n"); return 2; }
    if (src == dst) { fprintf(stderr, "FATAL: --src e --dst iguais; nao ha rota.\n"); return 2; }

    const int host_staged = !strcmp(route, "host");
    const int pinned      = !strcmp(hostmem, "pinned");
    if (!host_staged && strcmp(route, "peer")) {
        fprintf(stderr, "FATAL: --route aceita 'peer' ou 'host', recebeu '%s'\n", route);
        return 2;
    }

    int ndev = 0; CK(cudaGetDeviceCount(&ndev));
    if (src >= ndev || dst >= ndev) {
        fprintf(stderr, "FATAL: pedido src=%d dst=%d mas so ha %d GPUs visiveis "
                        "(cgroup do Slurm? use --gpu-bind=none)\n", src, dst, ndev);
        return 3;
    }

    /* A ROTA E' EVIDENCIA, nao suposicao: registra se ha P2P e por onde.
     * No modo `host` o P2P e' DELIBERADAMENTE nao usado -- o ponto do braco C
     * e' forcar o bounce pela DRAM do host. Registramos can_peer mesmo assim,
     * para que o consumidor saiba que havia um caminho direto e nao foi usado. */
    int canPeer = 0; CK(cudaDeviceCanAccessPeer(&canPeer, src, dst));
    int atomics = 0;
    cudaDeviceGetP2PAttribute(&atomics, cudaDevP2PAttrNativeAtomicSupported, src, dst);
    int perfRank = -1;
    cudaDeviceGetP2PAttribute(&perfRank, cudaDevP2PAttrPerformanceRank, src, dst);

    if (canPeer && !host_staged) {
        CK(cudaSetDevice(src)); cudaDeviceEnablePeerAccess(dst, 0); cudaGetLastError();
        CK(cudaSetDevice(dst)); cudaDeviceEnablePeerAccess(src, 0); cudaGetLastError();
    }

    const size_t bytes  = (size_t)mib * 1024ull * 1024ull;
    const size_t nwords = bytes / sizeof(uint32_t);

    uint32_t *hbuf = (uint32_t *)malloc(bytes);
    if (!hbuf) { fprintf(stderr, "FATAL: malloc de %zu bytes falhou\n", bytes); return 4; }
    fill_buffer(hbuf, nwords, fill, 12345u);

    /* N buffers por lado, rodiziados a cada iteracao. POR QUE ISTO EXISTE:
     * o job 136281 mediu a potencia subir +44% de 16 para 64 MiB com a vazao
     * quase parada (+8%) e o clock CONSTANTE. A hipotese e' reuso de L2 entre
     * iteracoes -- com um buffer so', a iteracao seguinte le do cache; se o
     * buffer nao couber, le da HBM, que custa muito mais por byte.
     * `--buffers N` faz o working set virar N*mib SEM mudar o tamanho da
     * transferencia: e' o teste que separa CAPACIDADE DE CACHE de TAMANHO DE
     * MENSAGEM. Se a potencia em 16 MiB x 4 buffers igualar a de 64 MiB x 1,
     * a explicacao e' cache. Se ficar baixa, a hipotese cai. */
    void **pSrc = (void **)calloc(buffers, sizeof(void *));
    void **pDst = (void **)calloc(buffers, sizeof(void *));
    if (!pSrc || !pDst) { fprintf(stderr, "FATAL: calloc falhou\n"); return 4; }
    for (int i = 0; i < buffers; i++) {
        CK(cudaSetDevice(src)); CK(cudaMalloc(&pSrc[i], bytes));
        CK(cudaMemcpy(pSrc[i], hbuf, bytes, cudaMemcpyHostToDevice));
        CK(cudaSetDevice(dst)); CK(cudaMalloc(&pDst[i], bytes));
    }
    free(hbuf);

    /* Buffer de STAGING, so' no modo host. PINNED vs PAGEABLE nao e' detalhe:
     * memoria pageable obriga o driver a fazer uma copia intermediaria para um
     * buffer fixado, entao o dado atravessa a DRAM do host mais vezes. E' o
     * "imposto pageable" -- por isso e' um argumento, nao uma escolha calada. */
    void *hstage = nullptr;
    if (host_staged) {
        if (pinned) CK(cudaHostAlloc(&hstage, bytes, cudaHostAllocDefault));
        else { hstage = malloc(bytes);
               if (!hstage) { fprintf(stderr, "FATAL: malloc staging falhou\n"); return 4; } }
    }

    cudaStream_t sSrc, sDst;
    CK(cudaSetDevice(src)); CK(cudaStreamCreate(&sSrc));
    CK(cudaSetDevice(dst)); CK(cudaStreamCreate(&sDst));

    /* Uma iteracao = mover `bytes` de src ate dst pela rota escolhida, usando
     * o buffer de indice `k` (rodizio quando --buffers > 1). */
    long long k = 0;
    #define MOVE_ONCE()                                                          \
        do {                                                                     \
            const int b_ = (int)(k % buffers);                                   \
            if (!host_staged) {                                                  \
                CK(cudaMemcpyPeerAsync(pDst[b_], dst, pSrc[b_], src,             \
                                       bytes, sSrc));                            \
                CK(cudaStreamSynchronize(sSrc));                                  \
            } else {                                                             \
                CK(cudaSetDevice(src));                                          \
                CK(cudaMemcpyAsync(hstage, pSrc[b_], bytes,                      \
                                   cudaMemcpyDeviceToHost, sSrc));               \
                CK(cudaStreamSynchronize(sSrc));                                  \
                CK(cudaSetDevice(dst));                                          \
                CK(cudaMemcpyAsync(pDst[b_], hstage, bytes,                      \
                                   cudaMemcpyHostToDevice, sDst));               \
                CK(cudaStreamSynchronize(sDst));                                  \
            }                                                                    \
            k++;                                                                 \
        } while (0)

    /* Aquecimento fora da janela medida: o Yang (SC24) manda descartar o tempo
     * de subida, e a primeira copia paga inicializacao de contexto.
     * Com N buffers, aquecer N vezes para que TODOS entrem em regime. */
    for (int i = 0; i < 3 * buffers; i++) MOVE_ONCE();

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    long long iters = 0;
    double elapsed = 0.0;
    do {
        MOVE_ONCE();
        iters++;
        clock_gettime(CLOCK_MONOTONIC, &t1);
        elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    } while (elapsed < (double)seconds);

    /* DOIS denominadores, e a diferenca importa.
     *   bytes_moved: bytes LOGICOS entregues (o que a aplicacao queria mover).
     *   bus_bytes:   bytes que ATRAVESSARAM barramento. No host-staging o dado
     *                cruza duas vezes (device->host, host->device), entao e' 2x.
     * Publicar J/GB sobre um ou outro da' numeros diferentes por um fator 2 no
     * braco C. O consumidor escolhe e DECLARA -- nao escolhemos por ele.
     * (Kronmal 1993: razoes so' se interpretam com os componentes a' vista.) */
    const unsigned long long moved = (unsigned long long)iters * bytes;
    const unsigned long long onbus = host_staged ? 2ull * moved : moved;

    /* `working_set_mib` = mib * buffers. E' a grandeza que a hipotese de cache
     * preve como determinante, e ela e' INDEPENDENTE do tamanho da mensagem. */
    printf("src,dst,mib,buffers,working_set_mib,fill,route,hostmem,can_peer,"
           "p2p_perf_rank,p2p_atomics,iters,bytes_moved,bus_bytes,elapsed_s,GBps\n");
    printf("%d,%d,%d,%d,%d,%s,%s,%s,%d,%d,%d,%lld,%llu,%llu,%.6f,%.3f\n",
           src, dst, mib, buffers, mib * buffers, fill, route,
           host_staged ? hostmem : "-",
           canPeer, perfRank, atomics, iters, moved, onbus, elapsed,
           (double)moved / elapsed / 1e9);

    for (int i = 0; i < buffers; i++) { cudaFree(pSrc[i]); cudaFree(pDst[i]); }
    free(pSrc); free(pDst);
    if (hstage) { if (pinned) cudaFreeHost(hstage); else free(hstage); }
    cudaStreamDestroy(sSrc); cudaStreamDestroy(sDst);
    return 0;
}
