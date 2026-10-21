# route_bench precisa do compilador CUDA (nvcc), nao so do header.
#   make CUDA_HOME=$CONDA_PREFIX
CUDA_HOME ?= /usr/local/cuda
NVCC      ?= $(CUDA_HOME)/bin/nvcc
# H100 = sm_90. Manter explicito: compilar para a arquitetura errada roda por
# JIT e paga compilacao dentro da janela que queremos medir.
ARCH      ?= -gencode arch=compute_90,code=sm_90
NVCCFLAGS ?= -O3 -std=c++14 $(ARCH)

# NOTE: NOME VERSIONADO, de proposito. O binario e' lido em TEMPO DE EXECUCAO, nao
# na submissao: sobrescrever `route_bench` enquanto ha' job na fila troca o
# binario debaixo dele. E a v2 MUDOU O FORMATO DO CSV (colunas route, hostmem,
# bus_bytes), o que quebraria os `cut -f9/-f10` dos scripts ja' submetidos.
# Enquanto 136032/136033/136034 estiverem na fila, `route_bench` fica intocado.
route_bench2: route_bench.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

route_bench: route_bench.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f route_bench route_bench2 route_lat

# latencia: binario separado de proposito : quem cronometra uma copia de 4 B
# nao e' quem satura o link por 60 s. Ver cabecalho do route_lat.cu.
route_lat: route_lat.cu ../probe/events.h
	$(NVCC) $(NVCCFLAGS) -o $@ $<
