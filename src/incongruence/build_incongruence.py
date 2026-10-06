import itertools
import pandas as pd

df = pd.read_csv("data/processed/mustard_prepared.csv")

# Resumos por classe so com treino + DSEL -- o teste nao entra em nenhuma
# analise exploratoria
treino_dsel = pd.concat(
    [pd.read_csv("data/splits/train.csv"), pd.read_csv("data/splits/dsel.csv")]
)

MODALIDADES = ["text", "context", "audio", "visual"]

# Junta a valencia de cada modalidade em uma unica tabela, uma linha por
# instancia -- merge "left" para nao perder instancias que faltam em
# alguma modalidade (ex.: sem rosto detectado no video)
valencias = df[["KEY", "Sarcasm", "SENTENCE", "Explicit_Emotion", "Implicit_Emotion"]].copy()
for modalidade in MODALIDADES:
    colunas = [f"{modalidade}_valence", f"{modalidade}_intensity"]
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
# Intensidade de cada modalidade vai junto no csv -- nao e incongruencia
# nem arousal (correlacao ~0 com o arousal anotado), e sim quanta
# probabilidade foi para emocoes com polaridade. Valor baixo no sarcasmo
# combina com a entrega "cara de paisagem" (Castro et al., 2019)
COLUNAS_INTENSIDADE = [f"{modalidade}_intensity" for modalidade in MODALIDADES]
sem_valencia = valencias[valencias[COLUNAS_VALENCIA].isna().any(axis=1)]
for coluna in COLUNAS_VALENCIA:
    faltando = valencias[coluna].isna().sum()
    if faltando > 0:
        print(f"{coluna}: {faltando} instancias sem valor, preenchidas com 0")
valencias[COLUNAS_VALENCIA + COLUNAS_INTENSIDADE] = (
    valencias[COLUNAS_VALENCIA + COLUNAS_INTENSIDADE].fillna(0.0)
)

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
N_EXEMPLOS = 5

resultado = valencias[["KEY"] + COLUNAS_VALENCIA + COLUNAS_INTENSIDADE].copy()

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
colunas_oposicao = []

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
    resultado[f"opp_{modalidade_a}_{modalidade_b}"] = oposicao.astype(int)
    colunas_oposicao.append(f"opp_{modalidade_a}_{modalidade_b}")

resultado["n_oppositions"] = resultado[colunas_oposicao].sum(axis=1)

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
# Daqui para baixo, resumos por classe so com treino + DSEL
analise = valencias["KEY"].isin(treino_dsel["KEY"])
classe = valencias.loc[analise, "Sarcasm"]
print("\nResumos por classe: treino + DSEL,", analise.sum(), "instancias (teste fora)")
print("\nProporcao de sarcasmo por grupo:")
print(classe.groupby(resultado.loc[analise, "grupo"]).mean().round(3))
print("\nIntensidade media por classe (0 = nao sarcastico, 1 = sarcastico):")
print(resultado[analise].groupby(classe)[COLUNAS_INTENSIDADE].mean().round(3).T)
print("\nIncongruencia com sinal media por classe:")
print(resultado[analise].groupby(classe)[colunas_com_sinal].mean().round(3).T)
print("\nProporcao de oposicao de polaridade por classe:")
print(resultado[analise].groupby(classe)[colunas_oposicao + ["n_oppositions"]].mean().round(3).T)
# Mesma oposicao, mas so entre as instancias em que as duas modalidades tem
# polaridade -- tira o efeito de o sarcasmo ter mais modalidades neutras
print("\nOposicao condicional (so quando as duas modalidades tem polaridade):")
for modalidade_a, modalidade_b in PARES:
    ambas = (resultado[f"{modalidade_a}_polarity"] != "neutra") & (resultado[f"{modalidade_b}_polarity"] != "neutra")
    filtro = analise & ambas
    taxa = resultado.loc[filtro, f"opp_{modalidade_a}_{modalidade_b}"].groupby(valencias.loc[filtro, "Sarcasm"]).mean().round(3)
    print(f"  {modalidade_a:7s} x {modalidade_b:7s}: n = {filtro.sum():4d} | {taxa.to_dict()}")
print("\nProporcao de 'cara de paisagem' (audio e rosto neutros) por classe:")
print(resultado[analise].groupby(classe)["deadpan"].mean().round(3))
print("Salvo em: data/processed/incongruence.csv")

# Exemplos de instancias sarcasticas com o calculo aberto, para conferir a
# formula na mao -- amostra fixa (random_state) para sair sempre igual.
# As emocoes explicita/implicita sao a anotacao humana do MUStARD++, so para
# comparar com o que os classificadores viram
exemplos = valencias[valencias["Sarcasm"] == 1].sample(N_EXEMPLOS, random_state=42)

print("\n" + "=" * 70)
print("Exemplos de instancias sarcasticas:")
for _, row in exemplos.iterrows():
    print("\n" + "-" * 70)
    print("KEY:", row["KEY"])
    print("Fala:", row["SENTENCE"])
    print("Emocao explicita (anotada):", row["Explicit_Emotion"])
    print("Emocao implicita (anotada):", row["Implicit_Emotion"])
    print("Valencias (intensidade) e emocao predominante -> polaridade:")
    for modalidade in MODALIDADES:
        emocao = row[modalidade + "_emotion"] if isinstance(row[modalidade + "_emotion"], str) else "vazio"
        print(
            f"  {modalidade:8s} = {row[modalidade + '_valence']:+.3f}"
            f"  ({row[modalidade + '_intensity']:.3f})  {emocao} -> {POLARIDADE[emocao]}"
        )
    print("Grupo:", resultado.loc[row.name, "grupo"])
    print("Incongruencias (oposicao de polaridade entre colchetes):")
    for modalidade_a, modalidade_b in PARES:
        valor_a = row[f"{modalidade_a}_valence"]
        valor_b = row[f"{modalidade_b}_valence"]
        inc = (valor_a - valor_b) / 2
        oposicao = resultado.loc[row.name, f"opp_{modalidade_a}_{modalidade_b}"]
        print(
            f"  {modalidade_a:7s} x {modalidade_b:7s}: "
            f"({valor_a:+.3f} - ({valor_b:+.3f})) / 2 = {inc:+.3f}  [{oposicao}]"
        )
    print("Cara de paisagem:", resultado.loc[row.name, "deadpan"])
