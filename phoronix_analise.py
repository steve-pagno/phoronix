#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 phoronix_analise.py
--------------------------------------------------------------------------------
 Lê os arquivos CSV exportados pelo Phoronix Test Suite
 (`phoronix-test-suite result-file-to-csv <resultado>`), monta tabelas
 organizadas de hardware, software e desempenho, e gera todos os gráficos
 usados na análise comparativa.

 Uso básico
 ----------
     python phoronix_analise.py --entrada ./csv --saida ./saida

 Opções úteis
 ------------
     --baseline D4        dispositivo usado como referência 1,0x nos gráficos
                          normalizados (padrão: o mais lento no 7-Zip)
     --formato png        png | pdf | svg
     --dpi 200            resolução das imagens
     --sem-excel          não gera o arquivo .xlsx consolidado

 Configuração dos dispositivos
 -----------------------------
 Por padrão o script descobre todos os .csv da pasta de entrada e os rotula
 automaticamente. Para dar nomes e cores próprias, crie um `dispositivos.json`
 na pasta de entrada (veja MODELO_DISPOSITIVOS_JSON no fim deste arquivo).

 Dependências: pandas, numpy, matplotlib  (openpyxl apenas para o .xlsx)
================================================================================
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # backend sem interface gráfica — funciona em servidor
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


# =============================================================================
# 1. CONFIGURAÇÃO GERAL
# =============================================================================

PALETA_PADRAO = ["#E8792B", "#2E9BB5", "#5FA855", "#8A6FB0",
                 "#C0455E", "#4A5D6B", "#D9A441", "#3F7F6F"]

COR_TEXTO   = "#16202B"
COR_SUAVE   = "#6C7A8A"
COR_GRADE   = "#E3E9EE"
COR_DESTAQUE= "#C0455E"

# A qual subsistema cada teste pertence, e como interpretá-lo.
# "sentido": +1 -> maior é melhor (HIB) | -1 -> menor é melhor (LIB)
SUBSISTEMAS = {
    "SQLite":              dict(subsistema="Disco / I-O",      sentido=-1),
    "GpuTest":             dict(subsistema="GPU",              sentido=+1),
    "RAMspeed SMP":        dict(subsistema="Memória",          sentido=+1),
    "7-Zip Compression":   dict(subsistema="CPU multi-core",   sentido=+1),
    "FLAC Audio Encoding": dict(subsistema="CPU single-core",  sentido=-1),
}

# Campos de inventário classificados como hardware ou software.
CAMPOS_HARDWARE = ["Processor", "Motherboard", "Chipset", "Memory", "Disk",
                   "Graphics", "Audio", "Monitor", "Network"]
CAMPOS_SOFTWARE = ["OS", "Kernel", "Desktop", "Display Server", "Display Driver",
                   "OpenGL", "OpenCL", "Vulkan", "Compiler", "File-System",
                   "Screen Resolution", "System Layer"]

TRADUCAO_CAMPOS = {
    "Processor": "Processador", "Motherboard": "Placa-mãe", "Chipset": "Chipset",
    "Memory": "Memória", "Disk": "Armazenamento", "Graphics": "Gráficos",
    "Audio": "Áudio", "Monitor": "Monitor", "Network": "Rede",
    "OS": "Sistema operacional", "Kernel": "Kernel", "Desktop": "Ambiente gráfico",
    "Display Server": "Servidor gráfico", "Display Driver": "Driver de vídeo",
    "OpenGL": "OpenGL", "OpenCL": "OpenCL", "Vulkan": "Vulkan",
    "Compiler": "Compilador", "File-System": "Sistema de arquivos",
    "Screen Resolution": "Resolução de tela", "System Layer": "Camada de sistema",
}


@dataclass
class Dispositivo:
    """Uma máquina testada. Pode vir de um CSV com várias colunas de resultado."""
    id: str
    arquivo: Path
    rotulo: str = ""
    cor: str = "#888888"
    inventario: dict = field(default_factory=dict)

    @property
    def nome_curto(self) -> str:
        return f"{self.id} · {self.rotulo}" if self.rotulo else self.id


# =============================================================================
# 2. LEITURA E PARSING DOS CSV DO PHORONIX
# =============================================================================
#
# Formato do arquivo exportado pelo PTS:
#
#   ,<nome do result file>,
#   <linha em branco>
#    ,,"identificador-run-1","identificador-run-2"      <- cabeçalho do inventário
#   Processor,,AMD Ryzen ...,AMD Ryzen ...
#   ... (demais campos de sistema)
#   <linha em branco>
#    ,,"identificador-run-1","identificador-run-2"      <- cabeçalho dos resultados
#   "SQLite - Threads / Copies: 1 (sec)",LIB,51.204,
#   ...
#
# A coluna 1 (índice 1) só existe nas linhas de resultado e contém LIB ou HIB:
#   LIB = Lower Is Better  |  HIB = Higher Is Better
# As colunas a partir do índice 2 são execuções distintas da MESMA máquina.
# =============================================================================

REGEX_UNIDADE = re.compile(r"\(([^()]*)\)\s*$")


def ler_csv_bruto(caminho: Path) -> pd.DataFrame:
    """
    Lê o CSV como texto puro, sem inferir cabeçalho nem tipos.

    O arquivo do PTS é "irregular": as linhas têm números diferentes de
    colunas, o que faz o `read_csv` normal falhar. Por isso descobrimos
    antes qual é a linha mais larga e forçamos esse número de colunas.
    """
    with caminho.open(encoding="utf-8-sig", newline="") as f:
        largura = max((len(linha) for linha in csv.reader(f)), default=0)
    if largura == 0:
        raise SystemExit(f"Arquivo vazio: {caminho}")

    return pd.read_csv(caminho, header=None, names=range(largura), dtype=str,
                       keep_default_na=False, na_values=[], engine="python",
                       encoding="utf-8-sig")


def separar_secoes(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide o arquivo em (inventário, resultados).

    A linha de resultados é identificada pela coluna 1 conter LIB ou HIB.
    Tudo que vier antes da primeira linha de resultado e tiver rótulo na
    coluna 0 é tratado como inventário do sistema.
    """
    col1 = df[1].astype(str).str.strip().str.upper() if 1 in df.columns else pd.Series("", index=df.index)
    eh_resultado = col1.isin(["LIB", "HIB"])

    resultados = df[eh_resultado].copy()

    primeira_res = resultados.index.min() if len(resultados) else len(df)
    antes = df.loc[: primeira_res - 1] if len(resultados) else df

    rotulo = antes[0].astype(str).str.strip()
    eh_inventario = rotulo.ne("") & rotulo.isin(CAMPOS_HARDWARE + CAMPOS_SOFTWARE)
    inventario = antes[eh_inventario].copy()

    return inventario, resultados


def extrair_inventario(inventario: pd.DataFrame) -> dict[str, str]:
    """
    Converte o bloco de inventário em {campo: valor}.

    Quando o CSV tem várias colunas (várias execuções da mesma máquina),
    usamos o primeiro valor não vazio — o hardware é o mesmo em todas.
    """
    dados: dict[str, str] = {}
    colunas_valor = [c for c in inventario.columns if c >= 2]
    for _, linha in inventario.iterrows():
        campo = str(linha[0]).strip()
        valores = [str(linha[c]).strip() for c in colunas_valor
                   if str(linha[c]).strip() not in ("", "nan")]
        if valores:
            dados[campo] = valores[0]
    return dados


def desmembrar_nome_teste(nome: str) -> dict:
    """
    Quebra o identificador longo do PTS em partes utilizáveis.

    Exemplos
    --------
    'GpuTest - Test: GiMark - Resolution: 800 x 600 - Mode: Windowed (Points)'
        -> teste='GpuTest', unidade='Points',
           opcoes={'Test': 'GiMark', 'Resolution': '800 x 600', 'Mode': 'Windowed'}

    'FLAC Audio Encoding - WAV To FLAC (sec)'
        -> teste='FLAC Audio Encoding', unidade='sec', descricao='WAV To FLAC'

    'SQLite - Threads / Copies: 1 (sec)'
        -> teste='SQLite', unidade='sec', opcoes={'Threads / Copies': '1'}
    """
    nome = nome.strip().strip('"')

    unidade = ""
    achou = REGEX_UNIDADE.search(nome)
    if achou:
        unidade = achou.group(1).strip()
        nome = nome[: achou.start()].strip()

    partes = [p.strip() for p in nome.split(" - ")]
    teste = partes[0]

    opcoes: dict[str, str] = {}
    descricoes: list[str] = []
    for parte in partes[1:]:
        if ": " in parte:
            chave, valor = parte.split(": ", 1)
            opcoes[chave.strip()] = valor.strip()
        else:
            descricoes.append(parte)

    return dict(teste=teste, unidade=unidade, opcoes=opcoes,
                descricao=" / ".join(descricoes))


def extrair_resultados(resultados: pd.DataFrame, dispositivo_id: str) -> pd.DataFrame:
    """
    Converte o bloco de resultados em formato longo (uma linha por medição).

    Colunas de saída:
        dispositivo, teste, subsistema, metrica, unidade, sentido, valor,
        opcoes (dict) e uma coluna por chave de opção encontrada.
    """
    linhas = []
    colunas_valor = [c for c in resultados.columns if c >= 2]

    for _, linha in resultados.iterrows():
        info = desmembrar_nome_teste(str(linha[0]))
        sentido = -1 if str(linha[1]).strip().upper() == "LIB" else +1

        # várias colunas = várias execuções da mesma máquina; ficamos com a
        # primeira não vazia e guardamos quantas execuções existiam.
        brutos = []
        for c in colunas_valor:
            texto = str(linha[c]).strip().replace(",", ".")
            if texto in ("", "nan"):
                continue
            try:
                brutos.append(float(texto))
            except ValueError:
                continue
        if not brutos:
            continue

        registro = dict(
            dispositivo=dispositivo_id,
            teste=info["teste"],
            subsistema=SUBSISTEMAS.get(info["teste"], {}).get("subsistema", "Outro"),
            descricao=info["descricao"],
            unidade=info["unidade"],
            sentido=sentido,
            valor=brutos[0],
            n_execucoes=len(brutos),
            desvio_entre_execucoes=(np.std(brutos, ddof=0) if len(brutos) > 1 else 0.0),
            metrica=str(linha[0]).strip().strip('"'),
        )
        registro.update(info["opcoes"])
        linhas.append(registro)

    return pd.DataFrame(linhas)


def carregar_dispositivos(pasta: Path) -> list[Dispositivo]:
    """
    Descobre os CSV da pasta e aplica o `dispositivos.json`, se existir.

    Formato do JSON (todas as chaves além de "arquivo" são opcionais):
        [{"arquivo": "leonardo.csv", "id": "D1",
          "rotulo": "Ryzen 5 5600G", "cor": "#E8792B"}, ...]
    """
    arquivos = sorted(p for p in pasta.glob("*.csv"))
    if not arquivos:
        raise SystemExit(f"Nenhum .csv encontrado em {pasta}")

    config: dict[str, dict] = {}
    caminho_cfg = pasta / "dispositivos.json"
    if caminho_cfg.exists():
        for item in json.loads(caminho_cfg.read_text(encoding="utf-8")):
            config[item["arquivo"]] = item
        print(f"  configuração lida de {caminho_cfg.name}")

    dispositivos = []
    for i, arq in enumerate(arquivos):
        cfg = config.get(arq.name, {})
        dispositivos.append(Dispositivo(
            id=cfg.get("id", f"D{i + 1}"),
            arquivo=arq,
            rotulo=cfg.get("rotulo", ""),
            cor=cfg.get("cor", PALETA_PADRAO[i % len(PALETA_PADRAO)]),
        ))

    # ordena por id para que D1, D2, D3... apareçam sempre na mesma sequência
    dispositivos.sort(key=lambda d: d.id)
    return dispositivos


def montar_base(pasta: Path) -> tuple[list[Dispositivo], pd.DataFrame]:
    """Lê todos os CSV e devolve os dispositivos (com inventário) e o DataFrame longo."""
    dispositivos = carregar_dispositivos(pasta)
    blocos = []

    for disp in dispositivos:
        bruto = ler_csv_bruto(disp.arquivo)
        inventario, resultados = separar_secoes(bruto)
        disp.inventario = extrair_inventario(inventario)

        # rótulo automático a partir do processador, se não veio do JSON
        if not disp.rotulo:
            proc = disp.inventario.get("Processor", disp.arquivo.stem)
            disp.rotulo = re.sub(r"\s*@.*$", "", proc).replace("AMD ", "").replace("Intel ", "").strip()

        df = extrair_resultados(resultados, disp.id)
        blocos.append(df)
        print(f"  {disp.id:<3} {disp.arquivo.name:<22} "
              f"{len(df):>3} medições · {len(disp.inventario):>2} campos de inventário")

    longo = pd.concat(blocos, ignore_index=True)

    # ordem categórica estável para os gráficos
    ordem = [d.id for d in dispositivos]
    longo["dispositivo"] = pd.Categorical(longo["dispositivo"], categories=ordem, ordered=True)
    return dispositivos, longo


# =============================================================================
# 3. TABELAS DERIVADAS
# =============================================================================

def tabela_inventario(dispositivos: list[Dispositivo], campos: list[str]) -> pd.DataFrame:
    """Monta a tabela de HW ou SW: uma linha por campo, uma coluna por dispositivo."""
    linhas = []
    for campo in campos:
        if not any(campo in d.inventario for d in dispositivos):
            continue  # ninguém reportou esse campo
        linha = {"Item": TRADUCAO_CAMPOS.get(campo, campo)}
        for d in dispositivos:
            linha[d.nome_curto] = d.inventario.get(campo, "—")
        linhas.append(linha)
    return pd.DataFrame(linhas).set_index("Item")


def tabela_cobertura(longo: pd.DataFrame) -> pd.DataFrame:
    """Quantas medições cada dispositivo tem por teste — revela lacunas."""
    return (longo.pivot_table(index="teste", columns="dispositivo",
                              values="valor", aggfunc="count", observed=False)
            .fillna(0).astype(int))


def normalizar(df: pd.DataFrame, baseline: str,
               chave: list[str], coluna_valor: str = "valor") -> pd.DataFrame:
    """
    Acrescenta a coluna `relativo`: quantas vezes cada dispositivo é melhor
    que o `baseline` na mesma métrica.

    Respeita o sentido do teste — em testes onde menor é melhor a razão é
    invertida, de modo que >1 sempre significa "melhor".
    """
    ref = (df[df["dispositivo"] == baseline]
           .set_index(chave)[coluna_valor].rename("ref"))
    saida = df.join(ref, on=chave)
    razao = saida[coluna_valor] / saida["ref"]
    saida["relativo"] = np.where(saida["sentido"] > 0, razao, 1 / razao)
    return saida


def indice_por_subsistema(longo: pd.DataFrame, baseline: str) -> pd.DataFrame:
    """
    Índice sintético por subsistema: média geométrica do desempenho relativo
    de todas as métricas daquele subsistema, considerando apenas as métricas
    que TODOS os dispositivos mediram (comparação justa).
    """
    comum = (longo.groupby("metrica", observed=False)["dispositivo"]
             .nunique()
             .pipe(lambda s: s[s == longo["dispositivo"].nunique()].index))
    base = longo[longo["metrica"].isin(comum)].copy()
    if base.empty:
        return pd.DataFrame()

    base = normalizar(base, baseline, chave=["metrica"])
    return (base.groupby(["subsistema", "dispositivo"], observed=False)["relativo"]
            .apply(lambda s: float(np.exp(np.log(s).mean())))
            .unstack("dispositivo")
            .round(2))


def resumo_geral(longo: pd.DataFrame) -> pd.DataFrame:
    """Melhor, pior e amplitude por métrica — a base do quadro comparativo."""
    linhas = []
    for metrica, grupo in longo.groupby("metrica", observed=False):
        sentido = grupo["sentido"].iloc[0]
        melhor = grupo.loc[grupo["valor"].idxmax() if sentido > 0 else grupo["valor"].idxmin()]
        pior = grupo.loc[grupo["valor"].idxmin() if sentido > 0 else grupo["valor"].idxmax()]
        amplitude = (melhor["valor"] / pior["valor"]) if sentido > 0 else (pior["valor"] / melhor["valor"])
        linhas.append(dict(
            subsistema=grupo["subsistema"].iloc[0],
            teste=grupo["teste"].iloc[0],
            metrica=metrica,
            unidade=grupo["unidade"].iloc[0],
            sentido="maior é melhor" if sentido > 0 else "menor é melhor",
            n_dispositivos=grupo["dispositivo"].nunique(),
            melhor=f'{melhor["dispositivo"]} ({formatar_br(melhor["valor"], 2)})',
            pior=f'{pior["dispositivo"]} ({formatar_br(pior["valor"], 2)})',
            amplitude=round(float(amplitude), 2),
        ))
    return (pd.DataFrame(linhas)
            .sort_values(["subsistema", "teste", "metrica"])
            .reset_index(drop=True))


# =============================================================================
# 4. INFRAESTRUTURA DOS GRÁFICOS
# =============================================================================

def aplicar_estilo() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": COR_GRADE,
        "axes.labelcolor": COR_SUAVE,
        "axes.titlecolor": COR_TEXTO,
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.titlepad": 14,
        "axes.labelsize": 11,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": COR_GRADE,
        "grid.linewidth": 0.9,
        "xtick.color": COR_SUAVE,
        "ytick.color": COR_SUAVE,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "font.size": 11,
        "figure.autolayout": False,
    })


def limpar_eixos(ax, eixo_x: bool = True) -> None:
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.spines["left"].set_color(COR_GRADE)
    ax.spines["bottom"].set_color(COR_GRADE)
    ax.grid(axis="x" if not eixo_x else "y")
    ax.grid(axis="y" if not eixo_x else "x", visible=False)


def formatar_br(valor: float, casas: int = 1) -> str:
    """Formata número no padrão brasileiro: 1.234,5"""
    texto = f"{valor:,.{casas}f}"
    return texto.replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def rotular_barras(ax, containers, casas=1, sufixo="", tamanho=7.5) -> None:
    for cont in containers:
        ax.bar_label(cont, padding=2, fontsize=tamanho, color="#3D4A58",
                     fmt=lambda v: (formatar_br(v, casas) + sufixo) if v else "")


def salvar(fig, saida: Path, nome: str, args) -> Path:
    saida.mkdir(parents=True, exist_ok=True)
    caminho = saida / f"{nome}.{args.formato}"
    fig.savefig(caminho, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    ✓ {caminho.name}")
    return caminho


def barras_agrupadas(ax, dados: pd.DataFrame, cores: dict[str, str],
                     casas=1, sufixo="", rotulos=True):
    """
    `dados`: index = categorias do eixo X, columns = dispositivos.
    Desenha um grupo de barras por categoria.
    """
    categorias = list(dados.index)
    series = list(dados.columns)
    n = len(series)
    x = np.arange(len(categorias))
    largura = min(0.8 / max(n, 1), 0.28)

    conts = []
    for i, serie in enumerate(series):
        desloc = (i - (n - 1) / 2) * largura
        conts.append(ax.bar(x + desloc, dados[serie].values, largura,
                            label=serie, color=cores.get(serie, "#999999"),
                            edgecolor="none"))

    ax.set_xticks(x)
    ax.set_xticklabels(categorias)
    if rotulos:
        rotular_barras(ax, conts, casas=casas, sufixo=sufixo)
    return conts


# =============================================================================
# 5. GRÁFICOS — UM POR SUBSISTEMA
# =============================================================================

def grafico_sqlite(longo, disp_map, cores, saida, args):
    """Tempo por número de threads + curva de escalabilidade."""
    df = longo[longo["teste"] == "SQLite"].copy()
    if df.empty:
        return
    coluna = "Threads / Copies"
    if coluna not in df:
        return
    df["threads"] = pd.to_numeric(df[coluna], errors="coerce")
    df = df.dropna(subset=["threads"])

    tabela = df.pivot_table(index="threads", columns="dispositivo",
                            values="valor", observed=False).sort_index()
    # mantém só os níveis de thread medidos por todos — comparação justa
    completos = tabela.dropna(how="any")
    if completos.empty:
        completos = tabela

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.4),
                                   gridspec_kw={"width_ratios": [1.35, 1]})

    dados = completos.copy()
    dados.index = [f"{int(t)} thread" + ("s" if t > 1 else "") for t in dados.index]
    dados.columns = [disp_map[c] for c in dados.columns]
    barras_agrupadas(ax1, dados, cores, casas=1, sufixo=" s")
    ax1.set_title("SQLite — tempo por nível de concorrência")
    ax1.set_ylabel("segundos (menor é melhor)")
    ax1.legend(ncols=2, loc="upper left")
    limpar_eixos(ax1)

    # escalabilidade: quanto o tempo cresce em relação a 1 thread
    rel = tabela / tabela.iloc[0]
    for col in rel.columns:
        serie = rel[col].dropna()
        ax2.plot(serie.index, serie.values, marker="o", linewidth=2.2,
                 markersize=6, color=cores[disp_map[col]], label=disp_map[col])
    ax2.set_xscale("log", base=2)
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v)}"))
    ax2.set_xticks(sorted(tabela.index))
    ax2.set_title("Degradação relativa a 1 thread")
    ax2.set_xlabel("threads / cópias simultâneas")
    ax2.set_ylabel("× o tempo de 1 thread")
    ax2.legend()
    limpar_eixos(ax2)

    fig.suptitle("Disco e I/O", x=0.008, ha="left", fontsize=12,
                 color=COR_SUAVE, weight="bold")
    fig.tight_layout()
    salvar(fig, saida, "01_sqlite_disco", args)


def grafico_7zip(longo, disp_map, cores, saida, args):
    """Compressão e descompressão em MIPS, mais MIPS por thread."""
    df = longo[longo["teste"] == "7-Zip Compression"].copy()
    if df.empty:
        return
    df["modo"] = df["Test"].str.replace(" Rating", "", regex=False) if "Test" in df else "Rating"
    df["modo"] = df["modo"].replace({"Compression": "Compressão",
                                     "Decompression": "Descompressão"})

    tabela = df.pivot_table(index="modo", columns="dispositivo",
                            values="valor", observed=False)
    tabela.columns = [disp_map[c] for c in tabela.columns]

    fig, ax = plt.subplots(figsize=(11, 5.6))
    barras_agrupadas(ax, tabela, cores, casas=0)
    ax.set_title("7-Zip Compression — desempenho multi-core")
    ax.set_ylabel("MIPS (maior é melhor)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: formatar_br(v, 0)))
    ax.legend(ncols=4, loc="upper center", bbox_to_anchor=(0.5, -0.06))
    limpar_eixos(ax)
    fig.tight_layout()
    salvar(fig, saida, "02_7zip_cpu_multicore", args)


def grafico_flac(longo, disp_map, cores, saida, args):
    """Tempo de codificação — carga essencialmente single-core."""
    df = longo[longo["teste"] == "FLAC Audio Encoding"]
    if df.empty:
        return
    serie = (df.groupby("dispositivo", observed=False)["valor"].mean()
             .dropna().sort_values(ascending=False))

    fig, ax = plt.subplots(figsize=(10, 4.6))
    rotulos = [disp_map[i] for i in serie.index]
    barras = ax.barh(rotulos, serie.values,
                     color=[cores[r] for r in rotulos], height=0.62)
    ax.bar_label(barras, padding=4, fontsize=10, color="#3D4A58",
                 fmt=lambda v: formatar_br(v, 1) + " s")
    ax.set_title("FLAC Audio Encoding — desempenho single-core")
    ax.set_xlabel("segundos (menor é melhor)")
    ax.set_xlim(0, serie.max() * 1.18)
    limpar_eixos(ax, eixo_x=False)
    fig.tight_layout()
    salvar(fig, saida, "03_flac_cpu_singlecore", args)


def grafico_ramspeed(longo, disp_map, cores, saida, args):
    """Banda de memória por operação, Integer e Floating Point lado a lado."""
    df = longo[longo["teste"] == "RAMspeed SMP"].copy()
    if df.empty or "Type" not in df or "Benchmark" not in df:
        return
    ordem = ["Copy", "Scale", "Add", "Triad", "Average"]
    traducao = {"Average": "Média"}

    fig, eixos = plt.subplots(1, 2, figsize=(14, 5.4), sharey=True)
    for ax, modo, titulo in zip(eixos, ["Integer", "Floating Point"],
                                ["Integer", "Ponto flutuante"]):
        sub = df[df["Benchmark"] == modo]
        if sub.empty:
            ax.set_visible(False)
            continue
        tabela = sub.pivot_table(index="Type", columns="dispositivo",
                                 values="valor", observed=False)
        tabela = tabela.reindex([o for o in ordem if o in tabela.index])
        tabela.index = [traducao.get(i, i) for i in tabela.index]
        tabela.columns = [disp_map[c] for c in tabela.columns]
        barras_agrupadas(ax, tabela, cores, casas=0, rotulos=False)
        ax.set_title(titulo, fontsize=13)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: formatar_br(v, 0)))
        limpar_eixos(ax)

    eixos[0].set_ylabel("MB/s (maior é melhor)")
    manipuladores, etiquetas = eixos[0].get_legend_handles_labels()
    fig.legend(manipuladores, etiquetas, ncols=4, loc="lower center",
               bbox_to_anchor=(0.5, -0.04), frameon=False)
    fig.suptitle("RAMspeed SMP — banda de memória", fontsize=15,
                 weight="bold", color=COR_TEXTO, y=1.0)
    fig.tight_layout()
    salvar(fig, saida, "04_ramspeed_memoria", args)


def grafico_gputest(longo, disp_map, cores, saida, args, baseline, resolucao=None, modo=None):
    """
    Duas visões da GPU:
      esquerda  — pontuação absoluta em escala log (as demos têm ordens de
                  grandeza muito diferentes entre si);
      direita   — desempenho normalizado pelo dispositivo de referência.
    """
    df = longo[longo["teste"] == "GpuTest"].copy()
    if df.empty or "Test" not in df:
        return

    # Sem filtro explícito, escolhemos automaticamente a combinação de
    # resolução e modo que TODOS os dispositivos mediram e que cobre o maior
    # número de demos. Sem isso, uma máquina que rodou quatro resoluções
    # entraria na média com quatro vezes mais pontos que as demais.
    if not resolucao and not modo and {"Resolution", "Mode"} <= set(df.columns):
        n_disp = df["dispositivo"].nunique()
        candidatos = (df.groupby(["Resolution", "Mode"], observed=False)
                      .agg(dispositivos=("dispositivo", "nunique"),
                           demos=("Test", "nunique"))
                      .reset_index())
        candidatos = candidatos[candidatos["dispositivos"] == n_disp]
        if not candidatos.empty:
            escolha = candidatos.sort_values("demos", ascending=False).iloc[0]
            resolucao, modo = escolha["Resolution"], escolha["Mode"]
            print(f"      GpuTest: comparação feita em {resolucao} · {modo} "
                  f"({int(escolha['demos'])} demos em todos os dispositivos)")

    if resolucao and "Resolution" in df:
        df = df[df["Resolution"] == resolucao]
    if modo and "Mode" in df:
        df = df[df["Mode"] == modo]
    if df.empty:
        return

    ordem = ["GiMark", "Plot3D", "Furmark", "TessMark", "Triangle",
             "Pixmark Piano", "Pixmark Volplosion"]
    curto = {"Pixmark Piano": "Piano", "Pixmark Volplosion": "Volplosion"}

    tabela = df.pivot_table(index="Test", columns="dispositivo",
                            values="valor", observed=False)
    tabela = tabela.reindex([o for o in ordem if o in tabela.index])

    rel = normalizar(df, baseline, chave=["metrica"])
    tabela_rel = rel.pivot_table(index="Test", columns="dispositivo",
                                 values="relativo", observed=False)
    tabela_rel = tabela_rel.reindex(tabela.index)

    for t in (tabela, tabela_rel):
        t.index = [curto.get(i, i) for i in t.index]
        t.columns = [disp_map[c] for c in t.columns]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.6))

    barras_agrupadas(ax1, tabela, cores, casas=0, rotulos=False)
    ax1.set_yscale("log")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: formatar_br(v, 0)))
    ax1.set_title("Pontuação absoluta (escala log)")
    ax1.set_ylabel("pontos (maior é melhor)")
    ax1.tick_params(axis="x", rotation=20)
    limpar_eixos(ax1)

    barras_agrupadas(ax2, tabela_rel, cores, casas=1, sufixo="×", rotulos=True)
    ax2.axhline(1.0, color=COR_DESTAQUE, linewidth=1.2, linestyle="--", zorder=0)
    ax2.set_title(f"Relativo ao {disp_map[baseline]} (= 1,0×)")
    ax2.set_ylabel("× o dispositivo de referência")
    ax2.tick_params(axis="x", rotation=20)
    limpar_eixos(ax2)

    manipuladores, etiquetas = ax1.get_legend_handles_labels()
    fig.legend(manipuladores, etiquetas, ncols=4, loc="lower center",
               bbox_to_anchor=(0.5, -0.05), frameon=False)
    contexto = " · ".join(filter(None, [resolucao, modo]))
    fig.suptitle(f"GpuTest em OpenGL{' — ' + contexto if contexto else ''}",
                 fontsize=15, weight="bold", color=COR_TEXTO, y=1.0)
    fig.tight_layout()
    salvar(fig, saida, "05_gputest_gpu", args)


def grafico_escalonamento_resolucao(longo, disp_map, cores, saida, args):
    """
    Como cada demo reage ao aumento de resolução, e a comparação entre
    modo janela e tela cheia. Usa o dispositivo com a varredura mais completa.
    """
    df = longo[(longo["teste"] == "GpuTest")].copy()
    if df.empty or "Resolution" not in df or "Mode" not in df:
        return

    # escolhe o dispositivo que mediu mais resoluções distintas
    contagem = df.groupby("dispositivo", observed=False)["Resolution"].nunique()
    alvo = contagem.idxmax()
    if contagem.max() < 2:
        return
    df = df[df["dispositivo"] == alvo]

    def ordenar(res):
        a, b = re.findall(r"\d+", res)[:2]
        return int(a) * int(b)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.4))
    paleta = PALETA_PADRAO + ["#2B3846"]

    for ax, modo, titulo in zip((ax1, ax2), ("Windowed", "Fullscreen"),
                                ("Modo janela", "Modo tela cheia")):
        sub = df[df["Mode"] == modo]
        if sub.empty:
            ax.set_visible(False)
            continue
        tabela = sub.pivot_table(index="Resolution", columns="Test",
                                 values="valor", observed=False)
        tabela = tabela.reindex(sorted(tabela.index, key=ordenar))
        tabela = tabela.dropna(axis=1, how="any")
        if tabela.empty:
            ax.set_visible(False)
            continue
        normal = tabela / tabela.iloc[0] * 100
        rotulos = [r.replace(" x ", "×") for r in normal.index]
        for i, col in enumerate(normal.columns):
            ax.plot(rotulos, normal[col].values, marker="o", markersize=5,
                    linewidth=2.1, color=paleta[i % len(paleta)],
                    label=col.replace("Pixmark ", ""))
        ax.set_title(titulo, fontsize=13)
        ax.set_ylim(0, 115)
        ax.set_ylabel("% do resultado na menor resolução")
        ax.tick_params(axis="x", rotation=15)
        limpar_eixos(ax)

    manipuladores, etiquetas = ax1.get_legend_handles_labels()
    fig.legend(manipuladores, etiquetas, ncols=7, loc="lower center",
               bbox_to_anchor=(0.5, -0.06), frameon=False)
    fig.suptitle(f"Escalonamento por resolução — {disp_map[alvo]}",
                 fontsize=15, weight="bold", color=COR_TEXTO, y=1.0)
    fig.tight_layout()
    salvar(fig, saida, "06_gputest_escalonamento_resolucao", args)


def grafico_indice_geral(longo, disp_map, cores, saida, args, baseline):
    """Índice sintético por subsistema (média geométrica das métricas comuns)."""
    indice = indice_por_subsistema(longo, baseline)
    if indice.empty:
        return
    ordem = ["CPU single-core", "CPU multi-core", "Memória", "Disco / I-O", "GPU"]
    indice = indice.reindex([o for o in ordem if o in indice.index])
    indice.columns = [disp_map[c] for c in indice.columns]

    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    barras_agrupadas(ax, indice, cores, casas=2, sufixo="×")
    ax.axhline(1.0, color=COR_DESTAQUE, linewidth=1.2, linestyle="--", zorder=0)
    ax.set_title("Índice sintético por subsistema")
    ax.set_ylabel(f"× o {disp_map[baseline]} (referência = 1,0)")
    ax.legend(ncols=4, loc="upper center", bbox_to_anchor=(0.5, -0.06))
    limpar_eixos(ax)
    excluidos = sorted(set(longo["subsistema"]) - set(indice.index) - {"Outro"})
    nota = ("Média geométrica do desempenho relativo, considerando apenas as métricas "
            "medidas por todos os dispositivos.\nValores acima de 1,0 indicam desempenho "
            "superior ao da referência, em qualquer teste.")
    if excluidos:
        nota += ("\nSubsistema(s) fora do gráfico por cobertura incompleta: "
                 + ", ".join(excluidos) + ".")
    fig.text(0.01, -0.12, nota, fontsize=9, color=COR_SUAVE, ha="left")
    fig.tight_layout()
    salvar(fig, saida, "07_indice_por_subsistema", args)


def grafico_cobertura(longo, disp_map, saida, args):
    """Mapa de calor mostrando quais testes cada máquina conseguiu rodar."""
    cobertura = tabela_cobertura(longo)
    if cobertura.empty:
        return
    cobertura.columns = [disp_map[c] for c in cobertura.columns]

    fig, ax = plt.subplots(figsize=(2.6 + 1.35 * len(cobertura.columns),
                                    1.1 + 0.45 * len(cobertura)))
    dados = cobertura.values.astype(float)
    ax.imshow(np.where(dados > 0, 1, 0), cmap="Greens", vmin=0, vmax=2.6, aspect="auto")

    ax.set_xticks(range(len(cobertura.columns)), cobertura.columns, fontsize=9)
    ax.set_yticks(range(len(cobertura.index)), cobertura.index, fontsize=9)
    for i in range(dados.shape[0]):
        for j in range(dados.shape[1]):
            n = int(dados[i, j])
            ax.text(j, i, f"{n}" if n else "—", ha="center", va="center",
                    fontsize=10, color=COR_TEXTO if n else COR_DESTAQUE,
                    weight="bold" if n else "normal")
    ax.set_title("Cobertura: número de medições por teste", fontsize=12)
    ax.grid(False)
    for lado in ax.spines.values():
        lado.set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    salvar(fig, saida, "08_cobertura_dos_testes", args)


# =============================================================================
# 6. TABELAS DE HARDWARE E SOFTWARE COMO IMAGEM
# =============================================================================

def _largura_texto(texto: str) -> int:
    """Comprimento aproximado desconsiderando acentos (para dimensionar colunas)."""
    return len(unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore"))


def quebrar(texto: str, largura: int) -> str:
    """Quebra manual de linha respeitando palavras."""
    palavras, linhas, atual = str(texto).split(), [], ""
    for palavra in palavras:
        candidato = f"{atual} {palavra}".strip()
        if _largura_texto(candidato) <= largura:
            atual = candidato
        else:
            if atual:
                linhas.append(atual)
            atual = palavra
    if atual:
        linhas.append(atual)
    return "\n".join(linhas) or "—"


def tabela_como_imagem(df: pd.DataFrame, titulo: str, cores: dict[str, str],
                       saida: Path, nome: str, args,
                       largura_coluna_pol: float = 2.55, fonte: float = 9.5):
    """
    Renderiza um DataFrame como tabela formatada em imagem.

    A quebra de linha é calculada a partir da largura real da coluna em
    polegadas e do tamanho da fonte, para o texto nunca vazar da célula.
    """
    if df.empty:
        return

    # ~2,05 caracteres por ponto de fonte por polegada, com folga de 8%
    largura_celula = int(largura_coluna_pol * 72 / (fonte * 0.62) * 0.92)
    corpo = df.map(lambda v: quebrar(v, largura_celula))
    n_linhas = len(corpo)
    altura_linhas = [max(1, max(str(v).count("\n") + 1 for v in corpo.iloc[i]))
                     for i in range(n_linhas)]

    altura = 0.85 + 0.32 * sum(altura_linhas)
    largura = largura_coluna_pol * (len(corpo.columns) + 1)

    fig, ax = plt.subplots(figsize=(largura, altura))
    ax.axis("off")

    tabela = ax.table(
        cellText=corpo.values,
        rowLabels=corpo.index,
        colLabels=corpo.columns,
        cellLoc="center", rowLoc="left", loc="center",
    )
    tabela.auto_set_font_size(False)
    tabela.set_fontsize(fonte)
    tabela.scale(1, 1.0)

    # normaliza as alturas para a tabela ocupar toda a figura
    unidade = 0.94 / (sum(altura_linhas) + 1.35)

    for (linha, coluna), celula in tabela.get_celld().items():
        celula.set_edgecolor("#FFFFFF")
        celula.set_linewidth(1.2)
        if linha == 0:  # cabeçalho: cor do dispositivo
            rotulo = corpo.columns[coluna] if coluna >= 0 else ""
            celula.set_facecolor(cores.get(rotulo, "#4A5D6B"))
            celula.set_text_props(color="white", weight="bold", fontsize=10)
            celula.set_height(unidade * 1.35)
        elif coluna == -1:  # coluna de rótulos
            celula.set_facecolor("#FFFFFF")
            celula.set_text_props(color=COR_SUAVE, weight="bold")
            celula.set_height(unidade * altura_linhas[linha - 1])
        else:
            celula.set_facecolor("#F5F7F9" if linha % 2 else "#FFFFFF")
            celula.set_text_props(color="#2B3846")
            celula.set_height(unidade * altura_linhas[linha - 1])

    ax.set_title(titulo, fontsize=14, weight="bold", color=COR_TEXTO, pad=10)
    fig.tight_layout()
    salvar(fig, saida, nome, args)


# =============================================================================
# 7. EXPORTAÇÃO
# =============================================================================

def exportar_tabelas(pasta: Path, tabelas: dict[str, pd.DataFrame], args) -> None:
    pasta.mkdir(parents=True, exist_ok=True)
    for nome, df in tabelas.items():
        caminho = pasta / f"{nome}.csv"
        df.to_csv(caminho, encoding="utf-8-sig", sep=";", decimal=",")
        print(f"    ✓ {caminho.name}")

    if args.sem_excel:
        return
    try:
        caminho = pasta / "phoronix_consolidado.xlsx"
        with pd.ExcelWriter(caminho, engine="openpyxl") as escritor:
            for nome, df in tabelas.items():
                df.to_excel(escritor, sheet_name=nome[:31])
        print(f"    ✓ {caminho.name}")
    except ImportError:
        print("    ! openpyxl não instalado — pulei o .xlsx (pip install openpyxl)")


# =============================================================================
# 8. PROGRAMA PRINCIPAL
# =============================================================================

def principal(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Análise e gráficos a partir dos CSV do Phoronix Test Suite.")
    parser.add_argument("--entrada", type=Path, default=Path("."),
                        help="pasta com os arquivos .csv exportados pelo PTS")
    parser.add_argument("--saida", type=Path, default=Path("saida"),
                        help="pasta onde gravar gráficos e tabelas")
    parser.add_argument("--baseline", default=None,
                        help="id do dispositivo de referência (ex.: D4)")
    parser.add_argument("--resolucao", default=None,
                        help="filtra o GpuTest por resolução (ex.: '1920 x 1080')")
    parser.add_argument("--modo", default=None,
                        help="filtra o GpuTest por modo (Windowed | Fullscreen)")
    parser.add_argument("--formato", default="png", choices=["png", "pdf", "svg"])
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--sem-excel", action="store_true")
    args = parser.parse_args(argv)

    aplicar_estilo()

    print("\n[1/5] Lendo os arquivos do Phoronix Test Suite")
    dispositivos, longo = montar_base(args.entrada)
    disp_map = {d.id: d.nome_curto for d in dispositivos}
    cores = {d.nome_curto: d.cor for d in dispositivos}
    cores.update({d.id: d.cor for d in dispositivos})

    # --- escolha do baseline -------------------------------------------------
    if args.baseline and args.baseline in disp_map:
        baseline = args.baseline
    else:
        ref = longo[longo["teste"] == "7-Zip Compression"]
        baseline = (ref.groupby("dispositivo", observed=False)["valor"].mean().idxmin()
                    if not ref.empty else dispositivos[-1].id)
    print(f"      referência para normalização: {disp_map[baseline]}")

    # --- tabelas -------------------------------------------------------------
    print("\n[2/5] Montando tabelas")
    hardware = tabela_inventario(dispositivos, CAMPOS_HARDWARE)
    software = tabela_inventario(dispositivos, CAMPOS_SOFTWARE)
    resumo = resumo_geral(longo)
    indice = indice_por_subsistema(longo, baseline)
    cobertura = tabela_cobertura(longo)
    cobertura.columns = [disp_map[c] for c in cobertura.columns]

    largo = longo.pivot_table(index=["subsistema", "teste", "metrica", "unidade"],
                              columns="dispositivo", values="valor", observed=False)
    largo.columns = [disp_map[c] for c in largo.columns]

    print(f"      {len(longo)} medições · {len(longo['metrica'].unique())} métricas distintas")

    # --- gráficos ------------------------------------------------------------
    pasta_graficos = args.saida / "graficos"
    print("\n[3/5] Gerando gráficos de desempenho")
    grafico_sqlite(longo, disp_map, cores, pasta_graficos, args)
    grafico_7zip(longo, disp_map, cores, pasta_graficos, args)
    grafico_flac(longo, disp_map, cores, pasta_graficos, args)
    grafico_ramspeed(longo, disp_map, cores, pasta_graficos, args)
    grafico_gputest(longo, disp_map, cores, pasta_graficos, args, baseline,
                    resolucao=args.resolucao, modo=args.modo)
    grafico_escalonamento_resolucao(longo, disp_map, cores, pasta_graficos, args)
    grafico_indice_geral(longo, disp_map, cores, pasta_graficos, args, baseline)
    grafico_cobertura(longo, disp_map, pasta_graficos, args)

    print("\n[4/5] Gerando tabelas de hardware e software em imagem")
    tabela_como_imagem(hardware, "Hardware dos dispositivos testados",
                       cores, pasta_graficos, "09_tabela_hardware", args,
                       largura_coluna_pol=2.8)
    tabela_como_imagem(software, "Software dos dispositivos testados",
                       cores, pasta_graficos, "10_tabela_software", args,
                       largura_coluna_pol=2.5)

    print("\n[5/5] Exportando tabelas")
    exportar_tabelas(args.saida / "tabelas", {
        "hardware": hardware,
        "software": software,
        "resultados_longo": longo.drop(columns=["sentido"]).set_index("dispositivo"),
        "resultados_largo": largo,
        "resumo_por_metrica": resumo.set_index("metrica"),
        "indice_subsistema": indice,
        "cobertura": cobertura,
    }, args)

    # --- achados automáticos -------------------------------------------------
    print("\n" + "=" * 70)
    print("ACHADOS AUTOMÁTICOS")
    print("=" * 70)

    maiores = resumo.nlargest(3, "amplitude")
    print("\nMaiores diferenças entre dispositivos:")
    for _, linha in maiores.iterrows():
        print(f"  · {linha['amplitude']:>5.1f}×  {linha['metrica']}")
        print(f"           melhor {linha['melhor']} | pior {linha['pior']}")

    faltando = cobertura[(cobertura == 0).any(axis=1)]
    if not faltando.empty:
        print("\nTestes com cobertura incompleta:")
        for teste, linha in faltando.iterrows():
            ausentes = [c for c in linha.index if linha[c] == 0]
            print(f"  · {teste}: sem dados em {', '.join(ausentes)}")

    # métricas em que um dispositivo repetiu exatamente o mesmo valor em
    # configurações diferentes — sinal de parâmetro ignorado pelo teste
    suspeitas = (longo.groupby(["dispositivo", "teste", "valor"], observed=False)
                 .size().reset_index(name="n"))
    suspeitas = suspeitas[suspeitas["n"] >= 3]
    if not suspeitas.empty:
        print("\nValores idênticos repetidos em configurações diferentes")
        print("(possível parâmetro ignorado pelo benchmark):")
        for _, linha in suspeitas.iterrows():
            print(f"  · {disp_map[linha['dispositivo']]} — {linha['teste']}: "
                  f"{formatar_br(linha['valor'], 0)} aparece {linha['n']}×")

    print(f"\nTudo pronto em: {args.saida.resolve()}\n")
    return 0


MODELO_DISPOSITIVOS_JSON = """
[
  {"arquivo": "leonardo.csv",   "id": "D1", "rotulo": "Ryzen 5 5600G",  "cor": "#E8792B"},
  {"arquivo": "luciano.csv",    "id": "D2", "rotulo": "Ryzen 7 7800X3D","cor": "#2E9BB5"},
  {"arquivo": "result.csv",     "id": "D3", "rotulo": "Ryzen 5 5600",   "cor": "#5FA855"},
  {"arquivo": "result__1_.csv", "id": "D4", "rotulo": "Core i7-1255U",  "cor": "#8A6FB0"}
]
"""


if __name__ == "__main__":
    sys.exit(principal())
