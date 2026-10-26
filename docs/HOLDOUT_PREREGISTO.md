# Pré-registo do holdout — detetor de recifes rochosos do Algarve

Data de registo: 2026-10-26. Decidido pelo Rui (opção "LTHBAIF1 + 2025–2026"). Este ficheiro não pode ser alterado
depois de registado; o seu SHA-256 fica em `docs/LEDGER_EXPERIENCIAS.md`. Qualquer alteração exige nova entrada datada
no ledger, com o motivo, e invalida o holdout para os métodos já afinados.

## 1. Holdout espacial

- Área: setor COSMO **LTHBAIF1 (Praia da Barreta – Praia da Ilha do Farol)** com 1 km de buffer.
  Caixa WGS84 fixa: **W −7,905, E −7,835, S 36,940, N 36,990**.
- Verdade: DEM multifeixe COSMO LTHBAIF1 de 2018-10, 2019-10 e 2020-10 (APA, 0,30 m, PT-TM06, cotas ZH),
  em `/Volumes/4G_WD_BLACK/BACKUP/bulk_download_2026-10-02_02-35-48/`.
- Regra: nenhum píxel, estrutura, rótulo ou DEM desta caixa entra em treino, afinação, escolha de limiares,
  seleção de feições ou desenho de figuras de desenvolvimento. Os mapas de satélite podem cobri-la, mas qualquer
  amostra dentro da caixa é excluída por código (`src.constants.HOLDOUT_BBOX_WGS84`).

## 2. Holdout temporal

- Todas as passagens Sentinel-2 com data de aquisição **≥ 2025-01-01**, em todas as zonas.
- Os mapas e pilhas de desenvolvimento usam só **2017-01-01 a 2024-12-31**.
- Uso final: medir se as deteções de desenvolvimento persistem nas passagens reservadas.

## 3. Regras de decisão (fixadas antes de qualquer afinação)

O método novo conta como salto qualitativo se, em simultâneo:

1. Em T2 (litologia de Teixeira) e T3 (COSMO de desenvolvimento: LTHQUGA1, LTHFA1, LTHPEAL1), o limite inferior do
   IC 95 % (bootstrap por blocos de 500 m) do ganho de AUC sobre a melhor linha de base é ≥ +0,05.
2. No holdout espacial (LTHBAIF1), AUC ≥ 0,75 para rocha estável (relevo ≥ 0,3 m acima de um passa-baixo de 25 m,
   estável ≤ 0,3 m entre 2018–2020) contra fundo liso/móvel, com profundidade emparelhada (±1 m).
3. Calibração: ECE ≤ 0,10 (10 classes) nas previsões fora da amostra.
4. A crista da Greta (CRISTSE–CRISTNW) é recuperada (≥ 50 % do comprimento acima do limiar de operação a FPR 5 %)
   sem que a Greta tenha sido usada em treino ou afinação.

Linhas de base a reproduzir com o mesmo protocolo: (i) mediana z da persistência B03 do UV; (ii) `sdb_anomaly_m`
sobre a composição de dias límpidos; (iii) modelo só com a profundidade.

## 4. Outras restrições

- A Greta e o Zimbral não entram como positivos de treino nos modelos avaliados contra a regra 4 (viés de seleção).
- Marcas de pesca: só como rótulos fracos (PU), nunca como teste.
- Recifes artificiais (PSOEM lyr13 + tracejado Fig. 20): máscara/classe à parte; falsos positivos reportados à parte.
- Seeds: 42 para tudo, salvo indicação no ledger.
