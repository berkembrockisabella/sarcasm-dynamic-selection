import itertools
import pandas as pd

df = pd.read_csv("data/processed/mustard_prepared.csv")

MODALIDADES = ["text", "context", "audio", "visual"]

# Junta a valencia de cada modalidade em uma unica tabela, uma linha por
# instancia -- merge "left" para nao perder instancias que faltam em
# alguma modalidade (ex.: sem rosto detectado no video)
valencias = df[["KEY", "Sarcasm", "SENTENCE", "Explicit_Emotion", "Implicit_Emotion"]].copy()
for modalidade in MODALIDADES:
    colunas = [f"{modalidade}_valence", f"{modalidade}_intensity"]
    tabela = pd.read_csv(f"data/processed/{modalidade}_valence.csv")
    valencias = valencias.merge(tabela[["KEY"] + colunas], on="KEY", how="left")

# Instancia sem valencia em alguma modalidade (hoje so acontece no visual,
# quando nenhum rosto foi detectado) recebe 0, o mesmo valor que os scripts
# de valencia ja usam para entrada vazia -- sem isso o vetor teria NaN e
# quebraria o calculo de vizinhanca da selecao dinamica
COLUNAS_VALENCIA = [f"{modalidade}_valence" for modalidade in MODALIDADES]
# Intensidade de cada modalidade vai junto no csv -- nao e incongruencia,
# mas foi a feature que mais diferenciou sarcastico de nao sarcastico nos
# testes (as falas sarcasticas tem menos emocao, principalmente no rosto e
# no contexto). Diferencas de intensidade entre modalidades foram testadas
# e nao ajudaram, por isso nao sao calculadas
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
# valencias, sem nenhum peso arbitrario.
#   - inc_a_b (com sinal): guarda a direcao -- positivo quando a modalidade
#     "a" e mais positiva que "b" (ex.: texto positivo com contexto
#     negativo, o padrao classico de sarcasmo)
#   - abs_inc_a_b: so a intensidade do desacordo, em [0, 1]
PARES = list(itertools.combinations(MODALIDADES, 2))
N_EXEMPLOS = 5

resultado = valencias[["KEY"] + COLUNAS_VALENCIA + COLUNAS_INTENSIDADE].copy()
colunas_com_sinal = []
colunas_absolutas = []

for modalidade_a, modalidade_b in PARES:
    nome = f"inc_{modalidade_a}_{modalidade_b}"
    resultado[nome] = (
        valencias[f"{modalidade_a}_valence"] - valencias[f"{modalidade_b}_valence"]
    ) / 2
    resultado[f"abs_{nome}"] = resultado[nome].abs()
    colunas_com_sinal.append(nome)
    colunas_absolutas.append(f"abs_{nome}")

# Resumo escalar por instancia: media das incongruencias absolutas dos
# 6 pares -- util para analise exploratoria, nao substitui o vetor
resultado["incongruencia_media"] = resultado[colunas_absolutas].mean(axis=1)

# O rotulo de sarcasmo nao vai para o csv, igual aos csvs de valencia --
# assim o arquivo pode ser usado como feature sem risco de vazar o rotulo
resultado.to_csv("data/processed/incongruence.csv", index=False)

print("\nVetor de incongruencia calculado para", len(resultado), "instancias.")
print("Pares de modalidades:", len(PARES))
print("Instancias com alguma valencia ausente:", len(sem_valencia))
print("\nEstatisticas das incongruencias (com sinal):")
print(resultado[colunas_com_sinal].describe().round(3).T)
print("\nIntensidade media por classe (0 = nao sarcastico, 1 = sarcastico):")
print(resultado.groupby(valencias["Sarcasm"])[COLUNAS_INTENSIDADE].mean().round(3).T)
print("\nIncongruencia absoluta media por classe (0 = nao sarcastico, 1 = sarcastico):")
print(resultado.groupby(valencias["Sarcasm"])[colunas_absolutas + ["incongruencia_media"]].mean().round(3).T)
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
    print("Valencias (intensidade):")
    for modalidade in MODALIDADES:
        print(
            f"  {modalidade:8s} = {row[modalidade + '_valence']:+.3f}"
            f"  ({row[modalidade + '_intensity']:.3f})"
        )
    print("Incongruencias:")
    abs_da_instancia = []
    for modalidade_a, modalidade_b in PARES:
        valor_a = row[f"{modalidade_a}_valence"]
        valor_b = row[f"{modalidade_b}_valence"]
        inc = (valor_a - valor_b) / 2
        abs_da_instancia.append(abs(inc))
        print(
            f"  {modalidade_a:7s} x {modalidade_b:7s}: "
            f"({valor_a:+.3f} - ({valor_b:+.3f})) / 2 = {inc:+.3f}  |abs| = {abs(inc):.3f}"
        )
    print(f"Incongruencia media: {sum(abs_da_instancia) / len(abs_da_instancia):.3f}")
