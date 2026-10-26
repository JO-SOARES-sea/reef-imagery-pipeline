# Plano: salto qualitativo no detetor de recifes rochosos do Algarve

Data: 2026-10-26. Estado: **executado**. Holdout decidido pelo Rui ("LTHBAIF1 + 2025–2026") e pré-registado em `docs/HOLDOUT_PREREGISTO.md`; resultados em `docs/RELATORIO_SALTO_QUALITATIVO_2026-10-26.md` e `docs/LEDGER_EXPERIENCIAS.md`.

## 1. O que li (secção 1 do pedido)

- **Repositório:**
  - `git status` e `git log -5`: o HEAD é `64aae10`.
  - Nada da sessão de 22–26/10 está commitado.
  - O outro agente tem trabalho por commitar em `src/analysis/*`, `src/navigation/*`, `src/reports/*` e `scripts/reef_detector.py`. Não lhe toco.
- **Regras e memória:** `AGENTS.md` §1–5, `CLAUDE.md`, `MEMORY.md` e as três notas.
- **Investigação de 26/10:**
  - `docs/external_data/INVESTIGACAO_GRETA_QUARTEIRA_2026-10-26.md`, com a §4 revista;
  - `docs/external_data/LOCAL_SOURCES_INDEX.md`, que cobre 2 633 ficheiros do BACKUP e o texto integral de 470 PDFs.
- **UnderwaterVisibility:**
  - os 21 relatórios;
  - o `decision-log.md`, entradas DL-020 a DL-025, DL-042 e DL-043.
  - Li em pormenor:
    - `2026-09-27_ilha_faro_13m`;
    - `2026-09-27_garrao_candidata2`;
    - `2026-10-20_cosmo_lthfa1_praia_faro`;
    - `2026-10-21_greta_recife`;
    - `2026-09-30_pedra_do_valado`, §3c sobre o LTHNOCR1 e §3d sobre o RENSUB.
- **Últimas execuções (lidas e revistas com olho crítico):**

| Script | Resultado | Problemas encontrados na revisão |
|---|---|---|
| `scripts/sidescan_to_relief.py` | Rocha (relevo COSMO ≥ 0,3 m): AUC 0,92 por blocos de 250 m; só intensidade: 0,59. Relevo: MAE 0,0174 = linha de base, Spearman 0,01 | (1) O MAE é dominado pela areia (relevo ~0 em 99,4 % dos píxeis). É preciso medi-lo **só na rocha** e com PR-AUC. (2) A feição `gain` codifica a posição face ao nadir, o que pode levar a aprender as faixas. Testar sem ela. (3) O alvo usa só o COSMO de 2025, sem a estabilidade 2018–2025. (4) Só uma zona de calibração: falta testar a transferência. |
| `scripts/persistent_structures_ml.py` | Teixeira, blocos de 1 km: AUC 0,85 [0,77–0,92]; à mesma profundidade, 0,82. Transferência à mesma profundidade: Zimbral 99, Santa Eulália 100, Valado 86, Greta 67 | (1) O `p_rock` final usa **todos** os rótulos, incluindo a Greta. O "crista a vermelho" que mostrei é dentro da amostra; só a transferência (67) é legítima. (2) As zonas IF/ANCAO e FZW/FZE sobrepõem-se e duplicam estruturas. (3) Os negativos só existem numa zona. (4) Não há calibração (ECE) nem PR-AUC. |

## 2. Inventário: o que mudou face ao pedido

| Fonte | Estado verificado |
|---|---|
| **COSMO no disco** | LTHQUGA1 (2018/19/20/25), LTHPEAL1 (2018/19/20), LTHFA1 (2018/19/20/25), **LTHNOCR1** (Porches, 2018/19/20) e **LTHBAIF1** (Barreta–Farol, 2018/19/20; −7,890 a −7,849, 36,950–36,978). Nenhum detetor usou o LTHBAIF1 nem o LTHNOCR1 para afinar. O UV extraiu afloramentos do LTHNOCR1, mas este é quase todo plataforma rochosa (relevo ao acaso de 0,51 m) e fica abaixo dos 13 m de validade do método S2. |
| **Sonda do barco (H6)** | Encontrei **10 registos Lowrance `.sl3`** (2,4 GB, 1–8/09/2024) em `/Volumes/4G_WD_BLACK/Marinho_Windows/`: profundidade, GPS, downscan e sidescan ping a ping. Há também o `WaypointsRoutesTrails.usr` (trilhos; já existe `scripts/convert_lowrance_usr.py`). A pasta Olex (15 GB) é sobretudo software e ISOs; os dados de profundidade do Olex ainda estão por encontrar. |
| **GPS_10_2026** | 17 324 marcas, nenhum trilho, nenhuma profundidade. São rótulos fracos (nível b). |
| **RENSUB (CCMAR)** | Mapa de substrato (sonar lateral + mergulhos + arrastos, 2003–2010; EMODnet Seabed Habitats PT003015), já usado pelo UV no Valado. Pode servir de verdade de área para o T1. |
| **Holdout temporal antigo** | **Gasto.** O 2018–2022 foi libertado (DL-022) e os dias da Ilha de Faro foram escolhidos pela visibilidade do Zimbral (DL-024). |

## 3. Hipóteses escolhidas e ordem (com justificação)

0. **Linhas de base reproduzidas, antes de tudo:**
   - a mediana z do UV;
   - `sdb_anomaly_m` sobre a composição;
   - o modelo só com a profundidade.
   - Protocolo igual para todos: T1–T5, blocos, profundidade emparelhada, por faixa de profundidade.
1. **H5' — correção e validação justa do sonar do IH.** Fica primeiro porque já está metade feito e dá **verdade densa a 13–34 m** (onde o COSMO não chega), que é a peça que falta a quase tudo o resto.
   - MAE do relevo só na rocha, PR-AUC, sem a `gain`, alvo estável entre anos.
   - Transferência entre o LTHQUGA1 e o LTHFA1.
   - Sem ganho, a geometria pelos nadires e sombras, sendo honesto sobre o que o JPEG permite.
2. **H1 — decomposição temporal (fundo estático × visibilidade por passagem).** É o maior ganho esperado no sinal de satélite: a mediana trata da mesma forma dias turvos e límpidos.
   - Método: fatorização ponderada de posto 1 / PCA robusta da pilha de z por passagem.
   - Validação da visibilidade estimada: contra os 1 322 rótulos do Zimbral (o vetor v_p não usa esses rótulos).
   - Requer refazer a pilha por passagem (`scripts/build_persistence_map.py` guarda hoje só as medianas).
3. **H3 — mobilidade.** É barata com os DEM do COSMO (2018–2025 dá verdade fixo/móvel) e com a pilha por passagem (estabilidade entre anos). Separa diretamente rocha de formas de areia, que é o falso positivo dominante.
4. **H2 — índice de fundo invariante à profundidade (Lyzenga/Sagawa por passagem).** Complementa o H1 para manchas largas. Fica depois porque a 15–20 m o Kd por passagem é instável.
5. **H4 — rótulos de várias fontes com confiança, e PU learning para as marcas de pesca.** É o classificador final sobre as feições de H1–H3.
6. **H6 — batimetria própria (`.sl3`).** Em paralelo, porque é independente:
   - escrever um parser SL3 (sem dependência nova, se possível);
   - extrair profundidade e rugosidade ao longo dos trilhos;
   - usá-las como verdade a 13–20 m.
7. **H8 — lista de verificação para a sonda.** Entregue no fim de cada iteração (GPX com nomes de 8 caracteres).
8. **H7 — modelo espacial.** Só se H1–H4 não chegarem.
9. **H9 — Jev.** Só no fim, depois do gate, com JSON estruturado e critérios separados (rocha natural, recife artificial, forma de areia). **Chamadas em lote só com autorização.**

## 4. Holdout (PRÉ-REGISTO, a decidir pelo Rui)

Proposta:
- **Espacial:** o setor **COSMO LTHBAIF1 (Barreta–Farol) + 1 km de buffer** (≈ −7,905 a −7,835 W, 36,94–36,99 N).
  - Fica fora de todo o desenvolvimento: sem afinação, sem escolha de limiares, sem seleção de feições.
  - Verdade densa: multifeixe de 2018, 2019 e 2020.
  - Duas órbitas (R037 e R137).
  - Nenhum detetor foi afinado nele. O meu ML da sessão passada só lhe deu pontuações, sem rótulos nem afinação.
- **Temporal:** as passagens Sentinel-2 de **2025-01-01 em diante**, em todas as zonas.
  - Os mapas de desenvolvimento são refeitos só com 2017–2024.
  - No fim mede-se a persistência das deteções nas passagens reservadas.
- **Regras de decisão:** são as da secção 6 do pedido, fixadas aqui antes de qualquer afinação.
  - Salto = em T2 e T3, IC inferior do ganho ≥ +0,05 de AUC sobre a melhor linha de base;
  - AUC ≥ 0,75 no holdout espacial;
  - ECE ≤ 0,1;
  - crista da Greta recuperada sem afinação nela.
- **Hash:** `docs/HOLDOUT_PREREGISTO.md` será gravado com a data e o SHA-256 depois da decisão.

## 5. Conjuntos de teste (fixos)

- **T1 — Valado:** os 36 cumes do CCMAR contra fundo emparelhado, mais o RENSUB. Falta verificar o efeito de borda: distância dos cumes ao limite da máscara de profundidade e controlo com a máscara erodida.
- **T2 — Teixeira (Fig. 20):** rocha contra areia e rocha contra lodos, com bootstrap por blocos de 500 m.
- **T3 — COSMO (desenvolvimento):** LTHQUGA1, LTHFA1 e LTHPEAL1. Rocha = relevo ≥ 0,3 m estável 2018–2025; não rocha = fundo liso ou móvel.
- **T4 — confirmados** (Greta, Zimbral, Pedra do Alto, VILHA, Santa Eulália): só para verificar o tipo de fundo (viés de seleção, regra 5).
- **T5 — negativos:**
  - os lodos da Fig. 20 (incluindo a #7);
  - o fundo liso do COSMO;
  - as amostras de areia (S1–S26 e as 45 da Fig. 21, quando forem georreferenciadas);
  - os 10 alvos da AIA 3660 (falsos positivos).
- **Holdout (§4):** só no fim.
- **Máscara de recife artificial:** PSOEM lyr13 + tracejado da Fig. 20. Os falsos positivos dentro dela são reportados à parte.

## 6. Pedidos de decisão ao Rui

1. **Holdout** (§4).
2. **Downloads:**
   - refazer as pilhas por passagem nas 7 zonas (Valado → Cacela) com leitura em janela de COG: estimo 1–3 GB transferidos no total;
   - o Landsat/HLS (H3) fica para depois e eu pergunto antes.
3. **Jev em lote:** não está previsto antes do fim.
