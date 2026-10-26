# Pré-registo do holdout v2 — H4 revista (2026-10-26)

O holdout v1 (LTHBAIF1 + passagens ≥ 2025-01-01) foi gasto no teste final de 2026-10-26 (ledger E013). Este
pré-registo é feito ANTES de qualquer afinação da H4 revista. Não pode ser alterado depois de registado; o SHA-256 fica
no ledger.

## 1. Holdout espacial v2
- Setor COSMO **LTHNOCR1 (Praia Nova – Praia da Cova Redonda, Porches)** com 1 km de buffer.
  Caixa WGS84: **W −8,403, E −8,361, S 37,077, N 37,108** (`src.constants.HOLDOUT2_BBOX_WGS84`).
- Verdade: DEM COSMO LTHNOCR1 2018-10 vs 2020-11 (APA, 0,30 m, ZH), em
  `/Volumes/4G_WD_BLACK/BACKUP/bulk_download_2026-09-30_22-06-05/`. Rocha estável = relevo ≥ 0,3 m nos dois anos e
  |Δz| ≤ 0,3 m; liso = relevo < 0,15 m nos dois anos (agregação por máximo a 10 m).
- Nenhum píxel desta caixa entra em treino, afinação, seleção de feições ou figuras de desenvolvimento; as amostras
  da zona VALADO dentro da caixa são excluídas por código. Verificado antes do registo: 0 cumes CCMAR na caixa.
- As passagens ≥ 2025-01-01 continuam fora do desenvolvimento (já usadas no teste v1; não são reutilizadas como teste).

## 2. Verificação prospetiva
- As verificações de sonda do Rui sobre `data/external/persistent_structures/verificacao_sonda.*` (e listas futuras),
  feitas depois desta data, são verdade independente nunca vista: pedra/areia por marca.

## 3. Critérios (fixados agora)
1. Holdout v2: AUC ≥ 0,75 (rocha estável vs liso, estratificação por profundidade 0,5 m ×5, blocos 500 m).
2. ECE ≤ 0,10 no holdout v2.
3. Greta (nunca treinada) com ≥ 50 % da crista acima do limiar FPR 5 %.
4. Desenvolvimento (leave-one-zone-out): não pior do que a H4 congelada em T3_QUGA1 e T3_PEAL1 (limite inferior do
   ganho emparelhado ≥ −0,02).
Decisão: se 1–4 se cumprirem, o modelo revisto substitui a H4 na base costeira. O holdout v1 (LTHBAIF1) só pode
aparecer como diagnóstico, nunca como teste.
