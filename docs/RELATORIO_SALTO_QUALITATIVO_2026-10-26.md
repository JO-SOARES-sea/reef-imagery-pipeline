# Relatório: tentativa de salto qualitativo no detetor de recifes rochosos (2026-10-26)

Plano: `docs/PLANO_SALTO_QUALITATIVO.md`. Pré-registo: `docs/HOLDOUT_PREREGISTO.md` (SHA-256 `ce8835b4…`). Todas as
experiências, incluindo as que falharam: `docs/LEDGER_EXPERIENCIAS.md` (E000–E014). Seed 42 em tudo.

## Resposta curta

**O salto qualitativo NÃO ficou demonstrado.**
- No desenvolvimento, sempre com uma zona inteira deixada de fora, o método novo (H4) ganha de forma clara:
  - no multifeixe COSMO, AUC 0,96 e 0,87;
  - nos cumes do Valado, 0,82;
  - recupera 99 % da crista da Greta sem ter sido treinado nela.
- No **holdout pré-registado** (multifeixe COSMO de Barreta–Farol, nunca usado) **inverte-se: AUC 0,31**.
- O critério 1 também falha por pouco no COSMO e não se cumpre contra a litologia de Teixeira.
- Do lado positivo, as deteções **persistem nas passagens de 2025–2026**, que nunca foram vistas: 53–62 %, contra 26–45 % do fundo à mesma profundidade.

## Critérios pré-registados

| Critério | Resultado | Cumpre? |
|---|---|---|
| 1. IC inferior do ganho ≥ +0,05 sobre a melhor linha de base (T2 e T3) | T3 Quarteira +0,045; T3 Albufeira +0,041; T2 contra sdb_anomaly −0,066 | **Não** (falta 0,005–0,009 em T3; nenhum ganho em T2) |
| 2. AUC ≥ 0,75 no holdout espacial (LTHBAIF1) | **0,312 [0,244–0,387]** | **Não** |
| 3. ECE ≤ 0,10 | 0,077–0,097 no desenvolvimento; **0,39 no holdout** | Só no desenvolvimento |
| 4. Greta recuperada sem treino nela | 99 % da crista acima do limiar FPR 5 % | **Sim** |

## Tabela de resultados (desenvolvimento: zona de teste nunca vista; holdout no fim)

AUC [IC 95 % por blocos de 500 m]; negativos estratificados por profundidade (classes de 0,5 m, ×5); a profundidade sozinha dá ≈ 0,50 em todos os conjuntos.

| Conjunto | Só profundidade | Mediana z do UV | sdb_anomaly | H1 | H3 | **H4** | H4+H2 |
|---|---|---|---|---|---|---|---|
| T3 COSMO Quarteira (LTHQUGA1) | 0,50 | 0,83 [0,73–0,92] | 0,66 | 0,83 | 0,82 | **0,96 [0,94–0,98]** | 0,96 |
| T3 COSMO Albufeira (LTHPEAL1) | 0,51 | 0,72 [0,56–0,84] | — | 0,73 | 0,74 | **0,87 [0,80–0,91]** | 0,82 |
| T1 cumes CCMAR Valado | 0,49 | 0,05 (sinal invertido) | — | 0,05 | 0,28 | **0,82 [0,76–0,88]** | 0,87 |
| T2 Teixeira, rocha vs areia/lodo | 0,50 | 0,49 | 0,55 | 0,49 | 0,52 | 0,58 [0,52–0,65] | 0,58 |
| T2L Teixeira, rocha vs lodos | 0,50 | 0,48 | 0,66 | 0,48 | 0,50 | 0,68 [0,59–0,76] | 0,67 |
| **Holdout: COSMO Barreta–Farol (LTHBAIF1)** | 0,50 | ~0,71 (pós-hoc) | — | — | — | **0,31 [0,24–0,39]** | — |

**Outros resultados:**
- **Sonar do IH (H5'):** rocha estável contra fundo liso, AUC 0,89 [0,84–0,94], +0,38 sobre a intensidade. **A altura não é recuperável** (erro na rocha de 0,49 m, contra 0,12 m de prever a mediana).
- **Sonda própria (H6):** os registos Lowrance SL3 de setembro de 2024 confirmam o Zimbral como fundo rugoso (0,12 m contra 0,06 m). São a única verdade de relevo a 11–15 m.

## O que se aprendeu

1. **"Escuro = rocha" é só metade do sinal.**
   - Rocha baixa coberta de algas (COSMO de Quarteira e Albufeira, Barreta–Farol) é escura no passa-alto B03.
   - Cristas de relevo grande (cumes do Valado) são **claras**, porque a coluna de água por cima é menor (curvatura positiva no EMODnet).
   - A linha de base pré-registada estava, por isso, invertida no Valado (AUC 0,05).
2. **O avaliador tem de estratificar por profundidade.** O emparelhamento a ±1 m deixava a profundidade separar rocha de lodos (0,66–0,73). Com classes de 0,5 m isso desaparece.
3. **A síntese de Teixeira não serve de verdade ao nível do píxel.** 40 % da "rocha" está a 20–25 m e os polígonos estão generalizados. Serve ao nível de objeto, como o UV mostrou com 19/19 anomalias replicadas sobre rocha.
4. **A decomposição temporal (H1) e a estabilidade anual (H3) não acrescentam** sobre a mediana: o filtro de passagens limpas já remove os dias cegos.
5. **Porque o holdout falhou** (diagnóstico pós-hoc, E014): os sinais de rocha estão lá (|z| 0,72, dias escuros 0,72), mas o modelo usava os controlos de superfície B04/B08 como feições. Em Barreta a rocha é rasa (mediana 7,2 m), o vermelho vê o fundo e esses controlos invertem-se (0,19–0,23). Juntam-se duas mudanças de domínio:
   - FZW tem duas órbitas (R037+R137) e o treino só usou a R037;
   - FZW tem rocha mais rasa do que o treino.

## Limites de profundidade e clareza

- **Persistência B03 com verdade densa:**
  - 3–10 m: AUC 0,75 (mediana) a 0,96 (H4);
  - 10–15 m: 0,94–0,95;
  - acima de 15 m: só há verdade em pontos (cumes, Greta);
  - a 20–25 m não há sinal contra Teixeira.
- **Clareza:** o filtro de ≥ 90 % de água válida deixa 77–130 passagens por zona em 2017–2024. A visibilidade estimada (H1) varia 0,4–1,6 entre passagens, sem passagens "cegas".

## Entregáveis

- **`data/external/persistent_structures/estruturas_v2.{geojson,csv,gpx}`:** 3 288 estruturas.
  - Campos: p(rocha) calibrada, classe, relevo medido onde há COSMO, anos de persistência, nível de evidência e fonte, indicador de recife artificial e **`dominio_validado`**.
  - Fora do Ancão, de Santa Eulália e do Valado, as probabilidades **não estão validadas**.
- **`verificacao_sonda.{gpx,csv}`:** 30 candidatas (no máximo 6 por zona), com nomes de 8 caracteres e colunas para devolver hora, profundidade e pedra/areia.
  - A maioria é "clara": testa diretamente a hipótese que falhou no holdout.
- **Código:**
  - `src/analysis/{reef_eval,temporal_decomposition}.py`;
  - `src/navigation/lowrance_sl3.py`;
  - os scripts `build_persistence_map`, `cosmo_truth_rasters`, `eval_detectors`, `run_h1_decomposition`, `run_h3_mobility`, `build_h2_albedo`, `run_h4_classifier`, `final_holdout_test`, `build_structures_db`, `extract_sl3_soundings` e `sidescan_to_relief`;
  - testes offline em `tests/test_reef_eval.py` e `tests/test_temporal_decomposition.py`.
- **Modelo congelado:** `models/frozen/h4_2026-10-26.joblib` (SHA-256 `1122c811…`).

## Iteração depois do holdout v1: integração de todas as fontes (E015–E019)

Pré-registo v2 (`docs/HOLDOUT_PREREGISTO_v2.md`, SHA-256 `2ed51d01…`): holdout LTHNOCR1 (Porches), critério 4 =
"não pior do que a H4 congelada em T3_QUGA1 e T3_PEAL1 (limite inferior ≥ −0,02)" antes de gastar o holdout.

| Variante (declarada antes de correr) | T3 Quarteira vs H4 | T3 Albufeira vs H4 | T1 Valado | Critério 4 |
|---|---|---|---|---|
| **H4b**: todas as fontes (COSMO ×3, cumes, Teixeira, sonar IH, sonda SL3), só R037, sem B04/B08 | 0,895 (−0,065 [−0,129; −0,008]) | 0,799 (−0,063 [−0,131; −0,009]) | 0,358 | Falha |
| **H4c**: H4 congelada a ≥ 7 m + especialista raso sem B04/B08 | 0,959 (−0,001 [−0,003; +0,001]) | 0,850 (−0,017 [−0,048; +0,004]) | 0,823 | Falha (Albufeira) |
| \|z\| sem treino (exploratório, E017) | 0,863 | 0,720 | 0,840 | Não candidato |

**O holdout v2 continua por gastar.** Nenhuma variante cumpriu o critério de desenvolvimento, por isso não houve teste.

O que isto ensina:
- **Juntar mais fontes de rótulos num só modelo piorou.**
  - A H4b junta o regime "rocha escura" (algas, rasa) e o regime "rocha clara" (cristas do Valado), e perde nos dois.
  - Os rótulos fracos (sonar, SL3, Teixeira) não compensam a perda das feições B04/B08.
- **O sinal que transfere é o bilateral |z|.** Sem treino, dá 0,69 no LTHBAIF1 (só diagnóstico), contra 0,31–0,45 dos modelos treinados.
  - Os modelos aprendem relações com sinal (escuro *ou* claro) que mudam de zona para zona.
- **A explicação do E014 estava incompleta.** Com a H4c, a parte rasa do LTHBAIF1 deixa de inverter (0,53), mas a 10–25 m a H4 também inverte (0,14–0,35).
- **Parei de iterar modelos aqui.** Mais variantes contra o mesmo desenvolvimento seriam pesca.
  - O próximo ganho real vem de **verdade nova**, não de outro modelo: as verificações de sonda e o holdout v2.

## H1' — decomposição temporal com pilha alargada (E020–E023)

**Pergunta:** o H1 (E006) não rendeu porque o filtro de ≥ 90 % de água válida já tirava os dias cegos. Rende se
deixarmos entrar as passagens com 50–90 % e as pesarmos pela visibilidade estimada por passagem (v_p)?

**Resposta: não.** A regra foi declarada antes de correr (E020). Não se testam mais variantes H1 nestes conjuntos.

| Conjunto | \|s\| H1' (candidato) | Melhor referência sem treino | Ganho [IC 95 %] | H4 congelada (LOZO) |
|---|---|---|---|---|
| T3 Quarteira (LTHQUGA1) | 0,870 | \|mediana z\| 0,863 | +0,007 [−0,009; +0,026] | 0,958 |
| T3 Albufeira (LTHPEAL1) | 0,732 | mediana z 0,721 | +0,011 [−0,024; +0,042] | 0,855 |
| T2 Teixeira (secundário) | 0,551 | \|mediana z\| 0,569 | — | 0,584 |
| T2L lodos (secundário) | 0,633 | \|mediana z\| 0,612 | — | 0,675 |

O ganho pontual exigido era ≥ +0,02, com limite inferior ≥ 0, nos dois conjuntos de decisão. Não se cumpre em nenhum, e o
critério 4 contra a H4 também falha. A mediana ponderada (A) e o −s são exploratórios e também não decidem nada.

**O que se aproveita: o v_p é um bom índice de visibilidade sem rótulos.**
- Estimado longe do Zimbral (buffer de 1 km), separa os dias em que a pedra se vê dos dias em que não se vê: AUC 0,89 [0,85–0,93], com 46 dias visíveis e 156 não visíveis.
- Correlaciona com o KD490 CMEMS do dia, independente: ρ −0,23, n 318.
- Pode servir para escolher passagens limpas noutras zonas, mas não melhora a deteção de rocha.
- A maioria das passagens novas é quase cega no fundo (v_p mediano 0,03–0,55 consoante a faixa). Por isso acrescentar passagens não traz informação nova.

**Reprodução** (seed 42; pilhas 2017–2024, sem passagens ≥ 2025):
```
python scripts/eval_detectors.py --maps-suffix _dev --methods uv_median_z,depth,h1_static --sets T1,T2,T2L,T3_QUGA1,T3_PEAL1 --label h1prime_step0_reproduce
python scripts/build_persistence_map.py --uv-zone SE --name se_dev50 --emodnet data/processed/EMODnet_batimetria_se_dev.tif --min-valid 0.5 --save-stack --scl-prescreen --reuse-stack se_dev
python scripts/build_persistence_map.py --uv-zone ANCAO --name ancao_dev50 --emodnet data/processed/EMODnet_batimetria_Quarteira_Garrao.tif --min-valid 0.5 --save-stack --scl-prescreen --reuse-stack ancao_dev
python scripts/build_persistence_map.py --bbox -8.000 36.935 -7.905 36.985 --name if_zimbral_dev50 --emodnet data/processed/EMODnet_batimetria_if_dev.tif --min-valid 0.5 --save-stack --scl-prescreen
python scripts/run_h1_decomposition.py --zone se_dev50 --banded
python scripts/run_h1_decomposition.py --zone ancao_dev50 --banded
python scripts/run_h1_decomposition.py --zone if_zimbral_dev50 --banded --exclude-zimbral-km 1.0
python scripts/run_h1_decomposition.py --zone if_zimbral_dev50 --validate-zimbral
python scripts/eval_detectors.py --maps-suffix _dev --methods h1p_abs,uv_median_z,uv_abs_median_z,h4_lozo,h1p_neg,wmed_abs,wmed_neg,depth --compare-to uv_median_z,uv_abs_median_z,h4_lozo --sets T3_QUGA1,T3_PEAL1,T2,T2L --calibrate-lozo T3_QUGA1,T3_PEAL1 --primary h1p_abs --free-refs uv_median_z,uv_abs_median_z --h4-ref h4_lozo --label h1prime_eval
```

## Próximos passos (precisam de holdout novo)

1. **H4 revisto**, sem tocar no holdout gasto:
   - B04/B08 só como critério de exclusão e nunca como feição;
   - pilhas por órbita (R037 e R137 em separado);
   - verdade de treino a 3–7 m (LTHNOCR1 Porches e LTHPEAL1);
   - um modelo por regime ("albedo" contra "relevo"), com o regime decidido por |z| e pela curvatura, em vez de um único modelo a aprender interações frágeis.
2. **Holdout novo, pré-registado antes de qualquer afinação:**
   - recomendação: o COSMO LTHNOCR1 (Porches), ainda sem uso em treino;
   - e/ou as tuas verificações de sonda da lista atual, que são verdade independente e prospetiva.
3. **Mais sonda própria:**
   - os registos `.sl3` (incluindo os canais de sidescan 5/7/8, por explorar) dão relevo a 11–15 m;
   - as passagens deste ano sobre a lista de verificação fecham o ciclo H8;
   - a ingestão dos teus resultados devolvidos pode automatizar-se.
4. **Jev (H9):** não foi chamado. O gate não passou fora do domínio validado, e não há chamadas em lote sem a tua autorização.
