import itertools
import pandas as pd

df = pd.read_csv("data/processed/mustard_prepared.csv")

MODALIDADES = ["text", "context", "audio", "visual"]

# Junta a valencia de cada modalidade em uma unica tabela, uma linha por
# instancia -- merge "left" para nao perder instancias que faltam em
# alguma modalidade (ex.: sem rosto detectado no video)
valencias = df[["KEY"]].copy()
for modalidade in MODALIDADES:
    colunas = [f"{modalidade}_valence"]
    tabela = pd.read_csv(f"data/processed/{modalidade}_valence.csv")
    tabela = tabela.rename(columns={"emocao_predominante": f"{modalidade}_emotion"})
    valencias = valencias.merge(
        tabela[["KEY", f"{modalidade}_emotion"] + colunas], on="KEY", how="left"
    )

# Instancia sem valencia em alguma modalidade (hoje so acontece no visual,
# quando nenhum rosto foi detectado) recebe 0, o mesmo valor que os scripts
# de valencia ja usam para entrada vazia -- sem isso o vetor teria NaN e
# quebraria o calculo de vizinhanca da selecao dinamica
COLUNAS_VALENCIA = [f"{modalidade}_valence" for modalidade in MODALIDADES]
sem_valencia = valencias[valencias[COLUNAS_VALENCIA].isna().any(axis=1)]
for coluna in COLUNAS_VALENCIA:
    faltando = valencias[coluna].isna().sum()
    if faltando > 0:
        print(f"{coluna}: {faltando} instancias sem valor, preenchidas com 0")
valencias[COLUNAS_VALENCIA] = valencias[COLUNAS_VALENCIA].fillna(0.0)

# Incongruencia entre duas modalidades = diferenca de valencia dividida
# por 2. Como cada valencia esta em [-1, 1], a diferenca fica em [-2, 2];
# dividir por 2 devolve o resultado para [-1, 1], a mesma escala das
# valencias, sem nenhum peso arbitrario. So a versao com sinal e guardada:
# positivo quando "a" e mais positiva que "b" (ex.: texto positivo com
# contexto negativo, o padrao de Du et al., 2022). A versao absoluta foi
# retirada porque mistura desacordo com quantidade de emocao -- duas
# modalidades neutras (valencia ~0) sempre dao |diferenca| pequena, e o
# sarcasmo tem mais modalidades neutras
PARES = list(itertools.combinations(MODALIDADES, 2))

# Polaridade de cada modalidade a partir da emocao predominante -- mesma
# divisao da formula de valencia (positivas x negativas x resto). "vazio"
# (entrada vazia ou audio sem emocao valida) conta como neutra, igual a
# valencia 0 que ele ja recebe
POLARIDADE = {
    "happy": "positiva",
    "angry": "negativa",
    "disgust": "negativa",
    "fear": "negativa",
    "sad": "negativa",
    "neutral": "neutra",
    "surprise": "neutra",
    "vazio": "neutra",
}
COLUNAS_POLARIDADE = [f"{modalidade}_polarity" for modalidade in MODALIDADES]

resultado = valencias[["KEY"] + COLUNAS_VALENCIA].copy()

# Hipotese de roteamento: se as 4 modalidades tem a mesma polaridade (todas
# positivas, todas negativas ou todas neutras) elas concordam e a instancia
# vai para o grupo A de classificadores; se pelo menos uma difere, elas
# discordam e a instancia vai para o grupo B. Instancia sem rosto detectado
# tem emocao ausente e conta como neutra, igual a valencia 0
for modalidade in MODALIDADES:
    emocoes = valencias[f"{modalidade}_emotion"].fillna("vazio")
    resultado[f"{modalidade}_polarity"] = emocoes.map(POLARIDADE)

todas_iguais = resultado[COLUNAS_POLARIDADE].nunique(axis=1) == 1
resultado["grupo"] = todas_iguais.map({True: "A", False: "B"})
colunas_com_sinal = []
n_oposicoes = 0

for modalidade_a, modalidade_b in PARES:
    nome = f"inc_{modalidade_a}_{modalidade_b}"
    resultado[nome] = (
        valencias[f"{modalidade_a}_valence"] - valencias[f"{modalidade_b}_valence"]
    ) / 2
    colunas_com_sinal.append(nome)

    # Oposicao de polaridade: 1 so quando as duas modalidades tem emocao com
    # polaridade e elas sao contrarias (positiva x negativa). Modalidade
    # neutra nunca conta como oposicao -- assim a falta de emocao nao se
    # confunde com desacordo
    polaridade_a = resultado[f"{modalidade_a}_polarity"]
    polaridade_b = resultado[f"{modalidade_b}_polarity"]
    oposicao = (polaridade_a != "neutra") & (polaridade_b != "neutra") & (polaridade_a != polaridade_b)
    n_oposicoes = n_oposicoes + oposicao.astype(int)

# Quantos dos 6 pares tem polaridades opostas -- so a soma e guardada, as
# colunas opp_a_b por par nao sao salvas
resultado["n_oppositions"] = n_oposicoes

# "Cara de paisagem": audio e rosto sem emocao com polaridade, como no
# exemplo de Castro et al. (2019) -- separado da incongruencia para nao
# ser lido como concordancia
resultado["deadpan"] = (
    (resultado["audio_polarity"] == "neutra") & (resultado["visual_polarity"] == "neutra")
).astype(int)

# O rotulo de sarcasmo nao vai para o csv, igual aos csvs de valencia --
# assim o arquivo pode ser usado como feature sem risco de vazar o rotulo
resultado.to_csv("data/processed/incongruence.csv", index=False)

print("\nVetor de incongruencia calculado para", len(resultado), "instancias.")
print("Pares de modalidades:", len(PARES))
print("Instancias com alguma valencia ausente:", len(sem_valencia))
print("\nEstatisticas das incongruencias (com sinal):")
print(resultado[colunas_com_sinal].describe().round(3).T)
print("\nGrupos de roteamento (A = modalidades concordam, B = discordam):")
print(resultado["grupo"].value_counts().sort_index())
print("\nPolaridade comum das instancias do grupo A:")
print(resultado.loc[resultado["grupo"] == "A", "text_polarity"].value_counts())
print("Salvo em: data/processed/incongruence.csv")
