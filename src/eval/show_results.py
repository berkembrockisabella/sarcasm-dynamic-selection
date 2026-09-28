def main():

    print()
    print("==============================================")
    print("RESULTADOS")
    print("==============================================")
    print()

    print(f"{'Método':<25} {'F1-macro':>15}")
    print("-" * 42)

    print(f"{'T':<25} {'0.6120 (DSEL)':>15}")
    print(f"{'C':<25} {'0.7025 (DSEL)':>15}")
    print(f"{'A':<25} {'0.6639 (DSEL)':>15}")
    print(f"{'F':<25} {'0.6368 (DSEL)':>15}")

    print("-" * 42)

    print(f"{'DCS multimodal':<25} {'0.5597':>15}")
    print(f"{'DCS incongruência':<25} {'0.6189':>15}")
    print(f"{'DCS combinado':<25} {'0.6327':>15}")
    print(f"{'Ensemble estático':<25} {'0.7048':>15}")

    print()


if __name__ == "__main__":
    main()