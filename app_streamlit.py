#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 app_streamlit.py — Painel interativo dos resultados do Phoronix Test Suite
--------------------------------------------------------------------------------
 Lê os CSV exportados pelo PTS que estiverem na mesma pasta deste arquivo e
 monta um painel navegável. Qualquer alteração nos CSV é detectada
 automaticamente: o cache é invalidado pela data de modificação dos arquivos.

 Como rodar
 ----------
     pip install streamlit pandas numpy altair openpyxl
     streamlit run app_streamlit.py

 Requer o `phoronix_analise.py` na mesma pasta — é dele que vem toda a
 lógica de leitura e parsing dos CSV, para não haver duas versões da
 mesma regra.
================================================================================
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

PASTA_APP = Path(__file__).parent.resolve()
sys.path.insert(0, str(PASTA_APP))

try:
    from phoronix_analise import (
        CAMPOS_HARDWARE,
        CAMPOS_SOFTWARE,
        formatar_br,
        indice_por_subsistema,
        montar_base,
        normalizar,
        resumo_geral,
        tabela_cobertura,
        tabela_inventario,
    )
except ImportError as erro:  # pragma: no cover
    st.error(
        "Não encontrei o `phoronix_analise.py`. Ele precisa estar na mesma "
        f"pasta deste app (`{PASTA_APP}`).\n\nDetalhe do erro: {erro}"
    )
    st.stop()


# =============================================================================
# CONFIGURAÇÃO DA PÁGINA
# =============================================================================

st.set_page_config(
    page_title="Phoronix Test Suite — análise comparativa",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem; padding-bottom: 3rem;}
      [data-testid="stMetricValue"] {font-size: 1.65rem;}
      [data-testid="stMetricLabel"] {color: #6C7A8A;}
      div[data-testid="stDataFrame"] {border-radius: 8px;}
    </style>
    """,
    unsafe_allow_html=True,
)

ORDEM_SUBSISTEMAS = ["CPU single-core", "CPU multi-core", "Memória",
                     "Disco / I-O", "GPU", "Outro"]

# O parâmetro que faz um elemento ocupar toda a largura mudou de nome no
# Streamlit 1.49. Detectamos a versão para o app funcionar nas duas.
def _largura_total() -> dict:
    try:
        maior, menor = (int(p) for p in st.__version__.split(".")[:2])
    except Exception:
        return {"use_container_width": True}
    return {"width": "stretch"} if (maior, menor) >= (1, 49) else {"use_container_width": True}


LARGURA = _largura_total()


# =============================================================================
# CARREGAMENTO COM DETECÇÃO DE MUDANÇA NOS ARQUIVOS
# =============================================================================

def assinatura_da_pasta(pasta: Path) -> tuple:
    """
    Impressão digital dos CSV da pasta: nome, tamanho e data de modificação.

    É isso que faz o painel "puxar sozinho" quando um CSV muda — a assinatura
    entra como argumento da função cacheada, então qualquer alteração num
    arquivo gera uma chave de cache nova e força a releitura.
    """
    arquivos = sorted(pasta.glob("*.csv")) + sorted(pasta.glob("dispositivos.json"))
    return tuple(
        (p.name, p.stat().st_size, int(p.stat().st_mtime))
        for p in arquivos
    )


@st.cache_data(show_spinner="Lendo os resultados do Phoronix…")
def carregar(pasta_txt: str, assinatura: tuple):
    """Lê todos os CSV da pasta e devolve tudo já mastigado."""
    pasta = Path(pasta_txt)
    dispositivos, longo = montar_base(pasta)

    apelidos = {d.id: d.nome_curto for d in dispositivos}
    cores = {d.nome_curto: d.cor for d in dispositivos}

    hardware = tabela_inventario(dispositivos, CAMPOS_HARDWARE)
    software = tabela_inventario(dispositivos, CAMPOS_SOFTWARE)
    resumo = resumo_geral(longo)
    cobertura = tabela_cobertura(longo)

    # a coluna categórica não sobrevive bem ao cache; guardamos a ordem à parte
    ordem_ids = [d.id for d in dispositivos]
    longo = longo.copy()
    longo["dispositivo"] = longo["dispositivo"].astype(str)
    longo["nome"] = longo["dispositivo"].map(apelidos)

    return dict(
        dispositivos=[(d.id, d.nome_curto, d.cor, d.arquivo.name, d.inventario)
                      for d in dispositivos],
        ordem_ids=ordem_ids,
        apelidos=apelidos,
        cores=cores,
        longo=longo,
        hardware=hardware,
        software=software,
        resumo=resumo,
        cobertura=cobertura,
    )


# =============================================================================
# BARRA LATERAL
# =============================================================================

with st.sidebar:
    st.title("⚙️ Configuração")

    pasta_txt = st.text_input(
        "Pasta dos arquivos CSV",
        value=str(PASTA_APP),
        help="Por padrão, a mesma pasta deste arquivo. Os CSV devem ser os "
             "gerados por `phoronix-test-suite result-file-to-csv`.",
    )
    pasta = Path(pasta_txt).expanduser()

    if not pasta.is_dir():
        st.error("Pasta não encontrada.")
        st.stop()

    arquivos = sorted(pasta.glob("*.csv"))
    if not arquivos:
        st.error("Nenhum arquivo .csv nesta pasta.")
        st.stop()

    assinatura = assinatura_da_pasta(pasta)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🔄 Recarregar", **LARGURA):
            st.cache_data.clear()
            st.rerun()
    with col_b:
        vigiar = st.toggle("Vigiar", value=True,
                           help="Verifica os arquivos a cada poucos segundos e "
                                "recarrega sozinho quando algum CSV muda.")

    st.caption(f"{len(arquivos)} arquivo(s) encontrado(s)")
    with st.expander("Arquivos lidos", expanded=False):
        for nome, tamanho, modificado in assinatura:
            quando = pd.Timestamp(modificado, unit="s", tz="UTC").tz_convert(
                "America/Sao_Paulo").strftime("%d/%m/%Y %H:%M")
            st.caption(f"**{nome}** · {tamanho / 1024:.1f} kB · {quando}")


dados = carregar(str(pasta), assinatura)

longo: pd.DataFrame = dados["longo"]
cores: dict[str, str] = dados["cores"]
apelidos: dict[str, str] = dados["apelidos"]
nomes_ordenados = [apelidos[i] for i in dados["ordem_ids"]]


# --- vigia de arquivos ------------------------------------------------------
if vigiar:
    @st.fragment(run_every="4s")
    def vigiar_arquivos(pasta_vigiada: Path, assinatura_atual: tuple):
        """Recarrega o painel inteiro assim que algum CSV é alterado."""
        if assinatura_da_pasta(pasta_vigiada) != assinatura_atual:
            st.cache_data.clear()
            st.rerun(scope="app")

    vigiar_arquivos(pasta, assinatura)


with st.sidebar:
    st.divider()
    st.subheader("Dispositivos")
    selecionados = st.multiselect(
        "Mostrar no painel", nomes_ordenados, default=nomes_ordenados,
        label_visibility="collapsed",
    )
    if not selecionados:
        st.warning("Selecione pelo menos um dispositivo.")
        st.stop()

    st.subheader("Referência")
    baseline_nome = st.selectbox(
        "Dispositivo de referência (= 1,0×)", selecionados,
        index=len(selecionados) - 1,
        help="Usado nos gráficos normalizados e no índice sintético.",
    )
    baseline_id = {v: k for k, v in apelidos.items()}[baseline_nome]

    st.divider()
    st.caption("Feito com os CSV do Phoronix Test Suite. "
               "Edite um arquivo e o painel se atualiza sozinho.")


longo = longo[longo["nome"].isin(selecionados)].copy()
paleta = alt.Scale(domain=selecionados, range=[cores[n] for n in selecionados])


# =============================================================================
# AJUDANTES DE GRÁFICO (Altair — já vem junto com o Streamlit)
# =============================================================================

def rotulo_br(serie: pd.Series, casas: int = 1, sufixo: str = "") -> pd.Series:
    return serie.map(lambda v: formatar_br(v, casas) + sufixo
                     if pd.notna(v) else "—")


def barras(df: pd.DataFrame, categoria: str, valor: str, titulo_y: str,
           casas: int = 1, sufixo: str = "", ordem_x: list | None = None,
           log: bool = False, altura: int = 380, rotulos: bool = True):
    """Barras agrupadas: uma cor por dispositivo, um grupo por categoria."""
    base_df = df.copy()
    base_df["rotulo"] = rotulo_br(base_df[valor], casas, sufixo)

    escala = alt.Scale(type="log", zero=False) if log else alt.Scale(zero=True)
    eixo_x = alt.X(f"{categoria}:N", title=None, sort=ordem_x,
                   axis=alt.Axis(labelAngle=0, labelLimit=180))

    base = alt.Chart(base_df).encode(
        x=eixo_x,
        xOffset=alt.XOffset("nome:N", sort=selecionados),
        y=alt.Y(f"{valor}:Q", title=titulo_y, scale=escala),
        color=alt.Color("nome:N", scale=paleta,
                        legend=alt.Legend(title=None, orient="bottom")),
        tooltip=[alt.Tooltip("nome:N", title="Dispositivo"),
                 alt.Tooltip(f"{categoria}:N", title="Item"),
                 alt.Tooltip("rotulo:N", title=titulo_y)],
    )
    grafico = base.mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)

    if rotulos and not log:
        texto = base.mark_text(dy=-7, fontSize=10, color="#3D4A58").encode(
            text="rotulo:N", color=alt.value("#3D4A58"))
        grafico = grafico + texto

    return grafico.properties(height=altura).configure_view(strokeWidth=0)


def barras_horizontais(df: pd.DataFrame, valor: str, titulo_x: str,
                       casas: int = 1, sufixo: str = "", crescente: bool = True,
                       altura: int = 300):
    base_df = df.copy()
    base_df["rotulo"] = rotulo_br(base_df[valor], casas, sufixo)
    ordem = "x" if crescente else "-x"

    base = alt.Chart(base_df).encode(
        y=alt.Y("nome:N", title=None, sort=ordem),
        x=alt.X(f"{valor}:Q", title=titulo_x),
        color=alt.Color("nome:N", scale=paleta, legend=None),
        tooltip=[alt.Tooltip("nome:N", title="Dispositivo"),
                 alt.Tooltip("rotulo:N", title=titulo_x)],
    )
    texto = base.mark_text(align="left", dx=5, fontSize=11,
                           color="#3D4A58").encode(text="rotulo:N")
    return ((base.mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
             + texto)
            .properties(height=altura).configure_view(strokeWidth=0))


def linhas(df: pd.DataFrame, x: str, valor: str, titulo_x: str, titulo_y: str,
           cor: str = "nome", escala_cor=None, log_x: bool = False,
           altura: int = 380, ordem_x: list | None = None):
    escala_x = alt.Scale(type="log", base=2) if log_x else alt.Scale(zero=False)
    tipo_x = "Q" if log_x else "N"
    enc_x = (alt.X(f"{x}:{tipo_x}", title=titulo_x, scale=escala_x) if log_x
             else alt.X(f"{x}:N", title=titulo_x, sort=ordem_x,
                        axis=alt.Axis(labelAngle=0)))

    base = alt.Chart(df).encode(
        x=enc_x,
        y=alt.Y(f"{valor}:Q", title=titulo_y, scale=alt.Scale(zero=False)),
        color=alt.Color(f"{cor}:N", scale=escala_cor or paleta,
                        legend=alt.Legend(title=None, orient="bottom")),
        tooltip=[alt.Tooltip(f"{cor}:N", title="Série"),
                 alt.Tooltip(f"{x}:{tipo_x}", title=titulo_x),
                 alt.Tooltip(f"{valor}:Q", title=titulo_y, format=".1f")],
    )
    return ((base.mark_line(strokeWidth=2.5) + base.mark_point(size=70, filled=True))
            .properties(height=altura).configure_view(strokeWidth=0))


def aviso_sem_dados(msg: str = "Nenhum dado para esta seleção."):
    st.info(msg, icon="ℹ️")


def contar_threads(processador: str) -> float:
    """Extrai o número de threads da string de CPU detectada pelo PTS."""
    import re
    achou = re.search(r"(\d+)\s*Threads", str(processador))
    return float(achou.group(1)) if achou else np.nan


# =============================================================================
# CABEÇALHO
# =============================================================================

st.title("Phoronix Test Suite — análise comparativa")
st.caption(
    f"{len(selecionados)} dispositivo(s) · {len(longo)} medições · "
    f"{longo['metrica'].nunique()} métricas distintas · "
    f"referência: **{baseline_nome}**"
)

cartoes = st.columns(len(selecionados))
for coluna, nome in zip(cartoes, selecionados):
    disp_id = {v: k for k, v in apelidos.items()}[nome]
    inventario = next(d[4] for d in dados["dispositivos"] if d[0] == disp_id)
    n_medicoes = int((longo["nome"] == nome).sum())
    with coluna:
        st.markdown(
            f"<div style='border-left:4px solid {cores[nome]};padding-left:10px'>"
            f"<b>{nome}</b><br>"
            f"<span style='color:#6C7A8A;font-size:0.82rem'>"
            f"{inventario.get('OS', '—')}<br>{n_medicoes} medições</span></div>",
            unsafe_allow_html=True,
        )

st.divider()


# =============================================================================
# ABAS
# =============================================================================

aba_geral, aba_cpu, aba_mem, aba_disco, aba_gpu, aba_inv, aba_dados = st.tabs(
    ["📈 Visão geral", "🧮 CPU", "🧠 Memória", "💾 Disco / I-O",
     "🎮 GPU", "🖥️ Hardware e software", "🗂️ Dados brutos"]
)


# ----------------------------------------------------------------- VISÃO GERAL
with aba_geral:
    st.subheader("Índice sintético por subsistema")
    st.caption(
        "Média geométrica do desempenho relativo, considerando apenas as "
        "métricas medidas por **todos** os dispositivos selecionados. "
        "Acima de 1,0 significa melhor que a referência, em qualquer teste."
    )

    indice = indice_por_subsistema(longo, baseline_id)
    if indice.empty:
        aviso_sem_dados("Não há métricas medidas por todos os dispositivos "
                        "selecionados. Tente reduzir a seleção.")
    else:
        indice_longo = (indice.rename(columns=apelidos)
                        .reset_index()
                        .melt(id_vars="subsistema", var_name="nome",
                              value_name="indice")
                        .dropna(subset=["indice"]))
        ordem = [s for s in ORDEM_SUBSISTEMAS if s in set(indice_longo["subsistema"])]
        st.altair_chart(
            barras(indice_longo, "subsistema", "indice",
                   f"× o {baseline_nome}", casas=2, sufixo="×", ordem_x=ordem),
            **LARGURA,
        )
        fora = sorted(set(longo["subsistema"]) - set(indice.index) - {"Outro"})
        if fora:
            st.warning(
                "Fora do gráfico por cobertura incompleta: **"
                + "**, **".join(fora) + "**. Veja a aba de dados brutos.",
                icon="⚠️",
            )

    st.divider()
    esquerda, direita = st.columns([1.25, 1])

    with esquerda:
        st.subheader("Maiores diferenças entre dispositivos")
        resumo = resumo_geral(longo)
        somente_completos = st.toggle(
            "Só métricas medidas por todos", value=True, key="resumo_completos")
        visao = resumo.copy()
        if somente_completos:
            visao = visao[visao["n_dispositivos"] == len(selecionados)]
        visao = visao.nlargest(12, "amplitude")[
            ["subsistema", "metrica", "melhor", "pior", "amplitude"]]
        st.dataframe(
            visao, **LARGURA, hide_index=True,
            column_config={
                "subsistema": st.column_config.TextColumn("Subsistema", width="small"),
                "metrica": st.column_config.TextColumn("Métrica", width="large"),
                "melhor": st.column_config.TextColumn("Melhor"),
                "pior": st.column_config.TextColumn("Pior"),
                "amplitude": st.column_config.NumberColumn("Amplitude", format="%.1f×"),
            },
        )

    with direita:
        st.subheader("Cobertura dos testes")
        st.caption("Quantas medições cada máquina tem em cada teste. "
                   "Zero significa que o teste não rodou ali.")
        cobertura = tabela_cobertura(longo).rename(columns=apelidos)
        st.dataframe(
            cobertura.style.background_gradient(cmap="Greens", axis=None)
            .format("{:.0f}"),
            **LARGURA,
        )
        vazios = cobertura[(cobertura == 0).any(axis=1)]
        for teste, linha in vazios.iterrows():
            ausentes = [c for c in linha.index if linha[c] == 0]
            st.warning(f"**{teste}** sem dados em: {', '.join(ausentes)}", icon="⚠️")


# ------------------------------------------------------------------------- CPU
with aba_cpu:
    st.subheader("Multi-core — 7-Zip Compression")
    sete_zip = longo[longo["teste"] == "7-Zip Compression"].copy()

    if sete_zip.empty:
        aviso_sem_dados()
    else:
        sete_zip["modo"] = (sete_zip.get("Test", pd.Series("Rating", index=sete_zip.index))
                            .astype(str).str.replace(" Rating", "", regex=False)
                            .replace({"Compression": "Compressão",
                                      "Decompression": "Descompressão"}))
        st.altair_chart(
            barras(sete_zip, "modo", "valor", "MIPS (maior é melhor)", casas=0,
                   ordem_x=["Compressão", "Descompressão"]),
            **LARGURA,
        )

        with st.expander("Normalizar pelo número de threads", expanded=False):
            st.caption(
                "MIPS dividido pelas threads que o PTS detectou. Separa o ganho "
                "que vem de ter mais núcleos do ganho que vem da arquitetura."
            )
            threads = {}
            for disp_id, nome, _, _, inventario in dados["dispositivos"]:
                threads[nome] = contar_threads(inventario.get("Processor", ""))
            por_thread = sete_zip.copy()
            por_thread["threads"] = por_thread["nome"].map(threads)
            por_thread["valor"] = por_thread["valor"] / por_thread["threads"]
            if por_thread["valor"].notna().any():
                st.altair_chart(
                    barras(por_thread.dropna(subset=["valor"]), "modo", "valor",
                           "MIPS por thread", casas=0,
                           ordem_x=["Compressão", "Descompressão"], altura=320),
                    **LARGURA,
                )
            else:
                aviso_sem_dados("Não consegui detectar o número de threads.")

    st.divider()
    st.subheader("Single-core — FLAC Audio Encoding")
    st.caption("Carga sequencial: depende de IPC, clock e cache, não da "
               "quantidade de núcleos.")
    flac = longo[longo["teste"] == "FLAC Audio Encoding"]

    if flac.empty:
        aviso_sem_dados()
    else:
        agregado = (flac.groupby("nome", as_index=False)["valor"].mean())
        st.altair_chart(
            barras_horizontais(agregado, "valor", "segundos (menor é melhor)",
                               casas=1, sufixo=" s", crescente=True,
                               altura=60 + 55 * len(agregado)),
            **LARGURA,
        )
        melhor = agregado.loc[agregado["valor"].idxmin()]
        pior = agregado.loc[agregado["valor"].idxmax()]
        col1, col2, col3 = st.columns(3)
        col1.metric("Mais rápido", melhor["nome"],
                    f'{formatar_br(melhor["valor"], 1)} s', delta_color="off")
        col2.metric("Mais lento", pior["nome"],
                    f'{formatar_br(pior["valor"], 1)} s', delta_color="off")
        col3.metric("Diferença",
                    f'{formatar_br(pior["valor"] / melhor["valor"], 2)}×',
                    "entre o melhor e o pior", delta_color="off")


# --------------------------------------------------------------------- MEMÓRIA
with aba_mem:
    st.subheader("RAMspeed SMP — banda de memória")
    ram = longo[longo["teste"] == "RAMspeed SMP"].copy()

    if ram.empty or "Benchmark" not in ram.columns:
        aviso_sem_dados("Nenhum resultado de RAMspeed. Vale lembrar que este "
                        "teste não tem perfil compatível com Windows no PTS.")
    else:
        modos = sorted(ram["Benchmark"].dropna().unique())
        escolha = st.radio("Tipo de operação", modos, horizontal=True,
                           format_func=lambda m: {"Integer": "Integer",
                                                  "Floating Point": "Ponto flutuante"}
                           .get(m, m))
        sub = ram[ram["Benchmark"] == escolha].copy()
        traducao = {"Average": "Média"}
        sub["operacao"] = sub["Type"].map(lambda t: traducao.get(t, t))
        ordem = [o for o in ["Copy", "Scale", "Add", "Triad", "Média"]
                 if o in set(sub["operacao"])]

        st.altair_chart(
            barras(sub, "operacao", "valor", "MB/s (maior é melhor)",
                   casas=0, ordem_x=ordem),
            **LARGURA,
        )

        media = sub[sub["operacao"] == "Média"]
        if not media.empty:
            colunas = st.columns(len(media))
            topo = media["valor"].max()
            for coluna, (_, linha) in zip(colunas, media.iterrows()):
                coluna.metric(
                    linha["nome"], f'{formatar_br(linha["valor"], 0)} MB/s',
                    f'{formatar_br(linha["valor"] / topo * 100, 0)}% do melhor',
                    delta_color="off")

        faltando = set(selecionados) - set(ram["nome"])
        if faltando:
            st.warning("Sem dados de RAMspeed em: " + ", ".join(sorted(faltando)),
                       icon="⚠️")


# ------------------------------------------------------------------ DISCO / IO
with aba_disco:
    st.subheader("SQLite — inserções com carga concorrente")
    sqlite = longo[longo["teste"] == "SQLite"].copy()
    coluna_threads = "Threads / Copies"

    if sqlite.empty or coluna_threads not in sqlite.columns:
        aviso_sem_dados()
    else:
        sqlite["threads"] = pd.to_numeric(sqlite[coluna_threads], errors="coerce")
        sqlite = sqlite.dropna(subset=["threads"])

        tabela = sqlite.pivot_table(index="threads", columns="nome",
                                    values="valor").sort_index()
        comuns = tabela.dropna(how="any")

        so_comuns = st.toggle(
            "Mostrar apenas os níveis de thread medidos por todos",
            value=True,
            help="Nem toda máquina rodou os mesmos níveis de concorrência.",
        )
        base_tabela = comuns if (so_comuns and not comuns.empty) else tabela

        absoluto = (base_tabela.reset_index()
                    .melt(id_vars="threads", var_name="nome", value_name="valor")
                    .dropna(subset=["valor"]))
        absoluto["rotulo_threads"] = absoluto["threads"].map(
            lambda t: f"{int(t)} thread" + ("s" if t > 1 else ""))
        ordem = [f"{int(t)} thread" + ("s" if t > 1 else "")
                 for t in sorted(base_tabela.index)]

        esquerda, direita = st.columns(2)
        with esquerda:
            st.markdown("**Tempo absoluto**")
            st.altair_chart(
                barras(absoluto, "rotulo_threads", "valor",
                       "segundos (menor é melhor)", casas=1, sufixo=" s",
                       ordem_x=ordem),
                **LARGURA,
            )
        with direita:
            st.markdown("**Degradação relativa a 1 thread**")
            st.caption("Quanto o tempo cresce conforme a concorrência aumenta. "
                       "Curva mais plana = disco aguenta melhor a carga.")
            relativo = (tabela / tabela.iloc[0]).reset_index().melt(
                id_vars="threads", var_name="nome", value_name="fator"
            ).dropna(subset=["fator"])
            st.altair_chart(
                linhas(relativo, "threads", "fator", "threads simultâneas",
                       "× o tempo de 1 thread", log_x=True),
                **LARGURA,
            )

        uma_thread = sqlite[sqlite["threads"] == 1]
        if len(uma_thread) > 1:
            melhor = uma_thread.loc[uma_thread["valor"].idxmin()]
            pior = uma_thread.loc[uma_thread["valor"].idxmax()]
            st.info(
                f"Com uma única thread, **{melhor['nome']}** leva "
                f"{formatar_br(melhor['valor'], 1)} s e **{pior['nome']}** leva "
                f"{formatar_br(pior['valor'], 1)} s — uma diferença de "
                f"{formatar_br(pior['valor'] / melhor['valor'], 1)}×. "
                "Quando as máquinas usam o mesmo modelo de SSD, uma diferença "
                "desse tamanho aponta para configuração do sistema de arquivos, "
                "não para o hardware.",
                icon="💡",
            )


# ------------------------------------------------------------------------- GPU
with aba_gpu:
    gpu = longo[longo["teste"] == "GpuTest"].copy()

    if gpu.empty or "Test" not in gpu.columns:
        aviso_sem_dados()
    else:
        tem_config = {"Resolution", "Mode"} <= set(gpu.columns)

        if tem_config:
            # combinações que todos os dispositivos selecionados mediram
            candidatos = (gpu.groupby(["Resolution", "Mode"])
                          .agg(n_disp=("nome", "nunique"), demos=("Test", "nunique"))
                          .reset_index())
            completos = candidatos[candidatos["n_disp"] == len(selecionados)]
            sugerido = (completos.sort_values("demos", ascending=False).iloc[0]
                        if not completos.empty else
                        candidatos.sort_values("demos", ascending=False).iloc[0])

            col1, col2 = st.columns(2)
            resolucoes = sorted(gpu["Resolution"].dropna().unique(),
                                key=lambda r: int(r.split("x")[0].strip() or 0)
                                if "x" in r else 0)
            with col1:
                resolucao = st.selectbox(
                    "Resolução", resolucoes,
                    index=resolucoes.index(sugerido["Resolution"]))
            with col2:
                modos = sorted(gpu["Mode"].dropna().unique())
                modo = st.selectbox(
                    "Modo", modos, index=modos.index(sugerido["Mode"]),
                    format_func=lambda m: {"Windowed": "Janela",
                                           "Fullscreen": "Tela cheia"}.get(m, m))

            filtrado = gpu[(gpu["Resolution"] == resolucao) & (gpu["Mode"] == modo)]
            if filtrado["nome"].nunique() < len(selecionados):
                st.warning(
                    "Nem todos os dispositivos mediram esta combinação. "
                    f"A comparação completa está disponível em "
                    f"**{sugerido['Resolution']} · {sugerido['Mode']}**.",
                    icon="⚠️",
                )
        else:
            filtrado = gpu

        if filtrado.empty:
            aviso_sem_dados()
        else:
            ordem_demos = ["GiMark", "Plot3D", "Furmark", "TessMark", "Triangle",
                           "Pixmark Piano", "Pixmark Volplosion"]
            curto = {"Pixmark Piano": "Piano", "Pixmark Volplosion": "Volplosion"}
            filtrado = filtrado.copy()
            filtrado["demo"] = filtrado["Test"].map(lambda t: curto.get(t, t))
            ordem = [curto.get(o, o) for o in ordem_demos
                     if curto.get(o, o) in set(filtrado["demo"])]

            st.markdown("**Pontuação absoluta**")
            st.caption("As demos têm ordens de grandeza muito diferentes entre "
                       "si — a escala logarítmica deixa todas legíveis no mesmo "
                       "gráfico.")
            usar_log = st.toggle("Escala logarítmica", value=True, key="gpu_log")
            st.altair_chart(
                barras(filtrado, "demo", "valor", "pontos (maior é melhor)",
                       casas=0, ordem_x=ordem, log=usar_log, rotulos=not usar_log),
                **LARGURA,
            )

            if baseline_id in set(filtrado["dispositivo"]):
                st.markdown(f"**Relativo ao {baseline_nome}**")
                relativo = normalizar(filtrado, baseline_id, chave=["metrica"])
                st.altair_chart(
                    barras(relativo.dropna(subset=["relativo"]), "demo",
                           "relativo", f"× o {baseline_nome}", casas=1,
                           sufixo="×", ordem_x=ordem),
                    **LARGURA,
                )
            else:
                aviso_sem_dados(f"{baseline_nome} não tem dados nesta "
                                "configuração — sem gráfico normalizado.")

        # ---- escalonamento por resolução ----
        if tem_config:
            st.divider()
            st.subheader("Escalonamento por resolução")
            com_varredura = (gpu.groupby("nome")["Resolution"].nunique()
                             .pipe(lambda s: s[s > 1]).index.tolist())
            if not com_varredura:
                aviso_sem_dados("Nenhum dispositivo rodou mais de uma resolução.")
            else:
                alvo = st.selectbox("Dispositivo", com_varredura, key="gpu_alvo")
                sub = gpu[gpu["nome"] == alvo]

                colunas = st.columns(len(sub["Mode"].unique()))
                for coluna, modo_atual in zip(colunas, sorted(sub["Mode"].unique())):
                    parcial = sub[sub["Mode"] == modo_atual]
                    tabela = parcial.pivot_table(index="Resolution",
                                                 columns="Test", values="valor")
                    if tabela.empty:
                        continue
                    tabela = tabela.reindex(sorted(
                        tabela.index,
                        key=lambda r: np.prod([int(n) for n in r.split("x")[:2]])
                        if "x" in r else 0))
                    tabela = tabela.dropna(axis=1, how="any")
                    if tabela.empty:
                        continue
                    percentual = (tabela / tabela.iloc[0] * 100).reset_index().melt(
                        id_vars="Resolution", var_name="demo", value_name="pct")
                    percentual["Resolution"] = percentual["Resolution"].str.replace(
                        " x ", "×", regex=False)
                    with coluna:
                        rotulo_modo = {"Windowed": "Modo janela",
                                       "Fullscreen": "Modo tela cheia"}.get(
                            modo_atual, modo_atual)
                        st.markdown(f"**{rotulo_modo}**")
                        st.altair_chart(
                            linhas(percentual, "Resolution", "pct", None,
                                   "% do resultado na menor resolução",
                                   cor="demo", escala_cor=alt.Scale(scheme="tableau10"),
                                   ordem_x=list(dict.fromkeys(percentual["Resolution"]))),
                            **LARGURA,
                        )

                planos = []
                for modo_atual, parcial in sub.groupby("Mode"):
                    variacao = (parcial.groupby("Test")["valor"]
                                .agg(lambda s: s.std() / s.mean() if s.mean() else 0))
                    if len(parcial["Resolution"].unique()) > 1 and variacao.max() < 0.05:
                        planos.append(modo_atual)
                if planos:
                    st.info(
                        "Em **" + "**, **".join(planos) + "** a pontuação quase "
                        "não muda entre resoluções: o teste está usando o modo "
                        "nativo do monitor e ignorando o parâmetro de resolução. "
                        "Para estudar escalonamento, use os dados em modo janela.",
                        icon="💡",
                    )


# ---------------------------------------------------- HARDWARE E SOFTWARE
with aba_inv:
    st.subheader("Hardware")
    hardware = dados["hardware"].copy()
    hardware = hardware[[c for c in hardware.columns if c in selecionados]]
    st.dataframe(hardware, **LARGURA)

    st.subheader("Software")
    software = dados["software"].copy()
    software = software[[c for c in software.columns if c in selecionados]]
    st.dataframe(software, **LARGURA)

    st.caption("Todos esses campos foram detectados automaticamente pelo "
               "Phoronix Test Suite e gravados junto com os resultados — nada "
               "foi digitado à mão.")

    st.divider()
    st.subheader("Diferenças de software entre as máquinas")
    st.caption("Linhas em que nem todos os dispositivos têm o mesmo valor. "
               "São as variáveis que podem explicar resultados divergentes em "
               "hardware parecido.")
    divergentes = software[software.nunique(axis=1) > 1]
    if divergentes.empty:
        st.success("Nenhuma divergência de software entre os selecionados.")
    else:
        st.dataframe(divergentes, **LARGURA)


# ------------------------------------------------------------------ DADOS
with aba_dados:
    st.subheader("Todas as medições")

    col1, col2 = st.columns([1, 1])
    with col1:
        testes = st.multiselect("Filtrar por teste",
                                sorted(longo["teste"].unique()),
                                default=sorted(longo["teste"].unique()))
    with col2:
        busca = st.text_input("Buscar na métrica", placeholder="ex.: Furmark, Triad, 1920")

    visao = longo[longo["teste"].isin(testes)]
    if busca:
        visao = visao[visao["metrica"].str.contains(busca, case=False, na=False)]

    colunas_base = ["nome", "subsistema", "teste", "metrica", "unidade", "valor"]
    extras = [c for c in visao.columns if c not in colunas_base +
              ["dispositivo", "sentido", "descricao", "n_execucoes",
               "desvio_entre_execucoes"]]
    exibida = visao[colunas_base + extras].rename(columns={
        "nome": "Dispositivo", "subsistema": "Subsistema", "teste": "Teste",
        "metrica": "Métrica", "unidade": "Unidade", "valor": "Valor"})

    st.dataframe(exibida, **LARGURA, hide_index=True, height=420)
    st.caption(f"{len(exibida)} linha(s)")

    st.divider()
    st.subheader("Exportar")

    largo = visao.pivot_table(index=["subsistema", "teste", "metrica", "unidade"],
                              columns="nome", values="valor")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button(
            "⬇️ Medições (CSV)",
            exibida.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
            file_name="phoronix_medicoes.csv", mime="text/csv",
            **LARGURA,
        )
    with col2:
        st.download_button(
            "⬇️ Matriz métrica × dispositivo (CSV)",
            largo.to_csv(sep=";", decimal=",").encode("utf-8-sig"),
            file_name="phoronix_matriz.csv", mime="text/csv",
            **LARGURA,
        )
    with col3:
        buffer = io.BytesIO()
        try:
            with pd.ExcelWriter(buffer, engine="openpyxl") as escritor:
                dados["hardware"].to_excel(escritor, sheet_name="hardware")
                dados["software"].to_excel(escritor, sheet_name="software")
                exibida.to_excel(escritor, sheet_name="medicoes", index=False)
                largo.to_excel(escritor, sheet_name="matriz")
                resumo_geral(longo).to_excel(escritor, sheet_name="resumo", index=False)
            st.download_button(
                "⬇️ Tudo (Excel)", buffer.getvalue(),
                file_name="phoronix_consolidado.xlsx",
                mime="application/vnd.openxmlformats-officedocument."
                     "spreadsheetml.sheet",
                **LARGURA,
            )
        except ImportError:
            st.caption("Instale `openpyxl` para exportar em Excel.")
