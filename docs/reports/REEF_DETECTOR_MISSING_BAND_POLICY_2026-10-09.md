# Detector de recifes — política de bandas em falta, radiometria e georreferenciação

Relatório de 2026-10-09. Substitui o relatório anterior desta conversa como estado atual, mas a secção 2.1 guarda a história completa das conclusões anteriores.

**Estado final: parcialmente concluído por falta de dados e de acesso.** As correções de software estão feitas e testadas. Não houve validação científica, porque nenhum pixel real pôde ser lido neste ambiente.

Legenda de evidência usada em todo o relatório:

- **[código]** observado no código-fonte acessível
- **[dados]** observado em pixels ou ficheiros de dados acessíveis
- **[metadados]** indicado apenas por metadados ou catálogo
- **[doc]** documentação oficial do fornecedor
- **[inferido]** dedução, não verificada diretamente
- **[inverificável]** impossível de verificar com o acesso atual

---

## 1. Repositório, branch e ambiente utilizados

| Item | Valor |
|---|---|
| Repositório alterado | `JO-SOARES-sea/reef-imagery-pipeline`, branch `main` @ `a590479`, clone superficial, alterações só no working tree (sem commit, stage ou push) |
| Repositórios só lidos | `JO-SOARES-sea/UnderwaterVisibility` (`origin/main` @ `4164c9b`, mais todos os ramos), `JO-SOARES-sea/ALGAS` @ `a56463a` |
| Ambiente | Contentor Linux na cloud, Python 3.12 num venv isolado fora do repositório (`uv`), dependências de `requirements.txt` |
| Mac | Sem acesso. Nenhuma pasta ligada e nenhum shell remoto. Não foi usado nenhum ficheiro local do Mac |
| Rede | Os metadados STAC chegaram pela ferramenta de leitura web. Os anfitriões de imagem foram recusados pelo proxy: `sentinel-cogs.s3.us-west-2.amazonaws.com`, `sentinel2l2a01.blob.core.windows.net`, `stac.dataspace.copernicus.eu`, `zipper.dataspace.copernicus.eu` |
| Identidade Git | Não alterada |

---

## 2. Diagnóstico atualizado

### 2.1 Histórico das conclusões (preservado)

| # | Afirmação anterior | Onde surgiu | Estado atual |
|---|---|---|---|
| H1 | O código local tem `B04 = 0.8 * B02` e `B08 = B02` | Sessão anterior com acesso ao Mac | **Não verificada nesta sessão.** Não aparece em nenhum ramo nem no histórico completo dos três repositórios (153 commits no reef-imagery-pipeline). Não aparece nos 341 rasters de treino. A ausência no Git **não a refuta**: pode ter existido em código local ou em outputs não versionados. [inverificável] |
| H2 | A substituição está provavelmente em código não commitado no Mac | Primeiro relatório desta conversa | Mantém-se **[inferido]**. Reforço: o plano de rebuild anterior incluía `assert_not_surrogate()` (em `src/sdb/rebuild.py`, só no Mac) para detetar exatamente estas cópias. Isso indica que a sessão anterior as viu ou suspeitou delas, mas a evidência não está acessível aqui |
| H3 | Nenhum loader subtrai o offset BOA; os dados do Planetary Computer ficam cerca de +0,1 acima | Primeiro relatório ([Likely]) | **Confirmado [código]** em todos os loaders. Para o Earth Search fica **confirmado [metadados + doc]**: o item exato publica `scale=0.0001` e `offset=-0.1`, e o detector dividia só por 10000. Para o Planetary Computer fica **[metadados + doc + inferido]**: o item não tem `raster:bands` e a documentação da Microsoft diz que o offset "propagates to the L2A data hosted by the Planetary Computer". Os pixels não foram verificados |
| H4 | O Earth Search talvez já aplique o offset | Primeiro relatório ([Guessing]) | **Corrigido.** O item tem `earthsearch:boa_offset_applied: true` **e** `raster:bands.offset = -0.1`. A regra oficial do README do Earth Search é aplicar o `offset` sempre que for diferente de 0. Seguimos `raster:bands`. A semântica exata da flag fica como incerteza residual (ver 8) |
| H5 | O detector grava num grid lat/lon esticado sobre uma janela UTM | Primeiro relatório | **Confirmado [código]** e agora coberto por testes |
| H6 | nodata (DN 0) é tratado como reflectância 0 em `measure_clarity` | Primeiro relatório | **Confirmado [código]** |
| H7 | `Kd_B04 = Kd_B03 * aw_B04/aw_B03 * 0.8` quando não há B04 | Primeiro relatório | **Confirmado [código]**. Foram encontradas mais duas invenções do mesmo tipo (ver 2.2) |
| H8 | Sem B08, todo o detector a jusante fica bloqueado | Primeiro relatório | **Confirmado [código]**, agora formalizado no registry |

### 2.2 Problemas reproduzidos e corrigidos

| ID | Problema | Evidência | Correção |
|---|---|---|---|
| P1 | O detector convertia `DN / 10000` e ignorava `raster:bands.offset = -0.1`. A reflectância de todas as bandas ficava cerca de +0,1 acima. Com `GLINT_NIR_MAX = 0.015`, o NIR sobre água (≈0,10 + 0,005) nunca passava o limiar | [código] + [metadados] | `src/s2_radiometry.py`: a conversão vem só de metadados explícitos e é aplicada uma única vez |
| P1b | Consequência [inferido]: com a escala antiga, a máscara de água ficaria praticamente vazia nas cenas do Earth Search com baseline ≥ 04.00. Não há outputs do detector no repositório para confirmar | [inferido] | — |
| P2 | Georreferenciação. Os lon/lat eram indexados diretamente numa transformada fictícia. A janela UTM era rotulada com `from_bounds(bbox WGS84)`. O recorte nas bordas do tile não atualizava a transformada | [código] | CRS nativo e transformada exata da janela lida. Os pontos são projetados para o CRS da cena. A janela é alinhada a pixels inteiros e cortada ao tile |
| P3 | nodata. Não lia `src.nodata`. Zeros entravam nas médias de clareza. Valores negativos válidos de NIR eram rejeitados (`nir >= 0`) | [código] | Máscaras de validade explícitas por banda. Inválido passa a NaN e é excluído de todas as estatísticas. Os negativos válidos são mantidos |
| P4 | Projeção da profundidade EMODnet sem `src_nodata`: o NaN podia contaminar a interpolação bilinear | [código] | `src_nodata`/`dst_nodata = NaN`. O teste confirma que não há mistura |
| P5 | `Kd_B04` inventado de três formas: (a) `Kd_B03·aw_B04/aw_B03·0.8` sem B04; (b) `kd_b04 = kd_seas` (prior de Kd490 copiado para 665 nm) quando falta B03 ou a inversão falha; (c) `kd_prior·0.1` com menos de 10 pixels | [código] | `kd_b04 = None` sempre que não há B04 genuína. O resumo grava `kd_b04_estimated` vazio. **O que depende disto:** só o campo `kd_b04_estimated` do `summary.csv` e uma linha de log. Não entra no ranker nem no SDB |
| P6 | `_robust_minmax` devolvia **zeros** quando não havia contraste, ou seja, uma "probabilidade 0" falsa | [código] | Devolve NaN (indefinido) |
| P7 | Sem B08 o código rebentava com `KeyError` ou, com uma máscara vazia, com `IndexError`. Não havia estado explícito | [código] | O registry (`src/band_registry.py`) decide o resultado. Sem B08 válida o detector devolve `status="unavailable"`, com o motivo, `validation=None` e nenhum output escrito |
| P8 | Downloader multibanda: bandas de 10 m lidas com janelas fracionárias; bandas de 20/60 m reamostradas para **outro** grid (de `transform_bounds` arredondado); sobrescrevia ficheiros sem aviso; não gravava radiometria nem proveniência | [código] | Grid de referência único (B02 nativo). As bandas de 20/60 m são reprojetadas exatamente para esse grid: bilinear para DN contínuo, nearest para camadas categóricas (SCL), nodata excluído do kernel. Proveniência e radiometria ficam por banda no JSON e nas tags do GeoTIFF. Os hrefs são gravados sem o token SAS. Recusa sobrescrever exceto com `--overwrite` |
| P9 | Fixture inconsistente nos testes antigos (EPSG:32629 com transformada em graus) | [código] | Corrigida para EPSG:4326, com o motivo documentado no teste |

### 2.3 Problemas encontrados mas **não** alterados (fora do âmbito A–D)

- **[código]** Cerca de 25 scripts convertem com `DN / 10000` sem offset. Entre eles: `multiband_reef_analysis.py`, `reef_image_comparator.py`, `santa_eulalia_best_dates.py`, `pedra_do_alto_best_images.py`, `local_cloud_analysis.py`, `evaluate_all_shortlist.py`, `hybrid_stac_physical_orchestrator.py`, `east_reef_audit.py`, `reef_audit_viewer.py`, `west_candidates_audit.py`, `sprint1_algarve_bathymetry.py`, `reef_finder_ortho.py`, `generate_*_1200dpi.py`, `src/enhancer.py`. Para cenas com baseline ≥ 04.00, os outputs destes caminhos estão **radiometry_unknown** e devem considerar-se bloqueados para uso científico até serem migrados para `s2_radiometry`.
- **[código]** `src/reef_ml_predictor_acolite.py` (caminho do `orchestrator_run`) decide se o input é DN com a heurística `max > REFLECTANCE_DN_THRESHOLD → /10000`, sem offset. Os inputs locais `reef_Output_Master/.../S2_B0x_20250925.tif` não têm metadados, por isso ficam **radiometry_unknown**.
- **[código]** Falhas pré-existentes nos testes: `src/dgt_sentinel_integrator.py:331` usa `MemoryFile` sem import (F821). Os testes de exportação de drift também falham. Ambos existiam antes destas alterações.

---

## 3. Ficheiros alterados e resumo dos diffs

| Ficheiro | Tipo | Resumo |
|---|---|---|
| `src/s2_radiometry.py` | novo (165 l.) | `Radiometry`, `radiometry_from_stac_asset` (exige `scale` e `offset`; para baseline < 04.00 aceita só `scale`; recusa float), `radiometry_from_product_metadata` (`BOA_QUANTIFICATION_VALUE` e `BOA_ADD_OFFSET` por banda), `dn_to_reflectance` (nodata primeiro, uma única conversão, recusa input float, mantém negativos) |
| `src/band_registry.py` | novo (101 l.) | Registry feature → bandas, dependências, pré-processamento e se é gating. `evaluate()` resolve as dependências transitivamente |
| `src/s2_grid.py` | novo (43 l.) | `bbox_window` (janela inteira no grid nativo, bbox densificada, cortada ao tile) e `unsigned_href` (remove tokens) |
| `scripts/reef_detector.py` | alterado (+307/−112) | `load_scene_from_item` com grid nativo, radiometria, validade e `band_status`; `lonlat_to_rowcol` e `rowcol_to_lonlat`; `measure_clarity` só com pixels válidos (`None` em vez de 0); `water_mask` exige B08 e validade; `compute_dii` recusa máscaras vazias ou com menos de 50 px; `process_scene_loaded` com gating e `status`; cada execução escreve numa pasta própria `run_<UTC>` e não sobrescreve |
| `scripts/download_multiband_s2.py` | alterado (+140/−110) | Grid único B02, reprojeção exata, nearest para camadas categóricas, proveniência e radiometria por banda, não sobrescreve. A assinatura de `download_multiband` é compatível e só ganha `overwrite` |
| `src/reef_ml_predictor_acolite.py` | alterado (+19/−9) | Remove as três invenções de `Kd_B04`. `kd_b04_estimated` fica vazio quando não há B04 |
| `tests/test_reef_detector.py` | alterado (+11/−4) | Fixture CRS corrigida; o caso de array constante passa a esperar NaN |
| `tests/test_missing_band_policy.py` | novo (397 l.) | 40 testes (ver 4) |
| `docs/reports/REEF_DETECTOR_MISSING_BAND_POLICY_2026-10-09.md` | novo | Este relatório |

Nenhum raster nem output legado foi alterado. Não há commit, stage, push nem merge, e nada foi integrado em V5.

---

## 4. Testes executados e resultados reais

Todos os testes novos usam **dados sintéticos**: GeoTIFFs e arrays feitos à mão, identificados como tal no ficheiro. Verificam o comportamento do software. **Não são evidência de desempenho científico.**

| Execução | Resultado |
|---|---|
| Suite offline antes das alterações (`pytest tests/ -m "not network"`) | 635 passed, **6 failed**, 14 skipped, 24 deselected |
| Suite offline depois das alterações | **675 passed**, as mesmas **6 failed** (pré-existentes, ver 2.3), 14 skipped, 24 deselected |
| `tests/test_missing_band_policy.py` | 40 passed |
| `test_missing_band_policy` + `test_reef_detector` + `test_orchestrator_run` | 110 passed |
| Teste com falhas injetadas no código (8 regressões deliberadas, cada uma revertida a seguir) | **8/8 detetadas**: escala `DN/10000`; nodata sem máscara; janela deslocada; lon/lat indexado diretamente (bug antigo); clareza sem filtros de validade; gating desativado; fallback de `Kd_B04` reposto; bilinear sem `src_nodata` |
| `ruff --select F,B` nos ficheiros tocados | limpo |
| `flake8 --select=E9,F63,F7,F82` | só o F821 pré-existente em `dgt_sentinel_integrator.py` |

O que os testes cobrem: conversão com offset; nodata antes da aritmética; negativos válidos; recusa de input float e de dupla conversão; ativo do PC sem `raster:bands` tratado como desconhecido; parsing do MTD; registry (B08 bloqueia a cadeia, B04 não); CRS e transformada nativos; marcador de pixel encontrado nas suas coordenadas; ida e volta coordenada→pixel→coordenada; janela de borda cortada com transformada correta; cobertura da bbox; bbox fora do tile; clareza sem nodata; janela toda nodata dá `None` e não 0; máscara exclui inválidos; profundidade sem contaminação; reamostragem que exclui nodata; `unavailable` sem B08 (ativo em falta, radiometria desconhecida, todo nodata); sem B04 o detector corre sem fingir true colour; recusa de sobrescrita; `Kd_B04` indisponível.

---

## 5. Testes e verificações não executados

| Verificação | Motivo |
|---|---|
| Executar o detector numa cena real (`S2B_29SNB_20250925_0_L2A`) | Anfitriões COG bloqueados pelo proxy. Pedido de rede negado por política, não contornado |
| Confirmar nos pixels se o NIR sobre mar aberto tem DN ≈ 1000–1100 (o que confirmaria o offset embutido) | Mesmo motivo |
| Ler `MTD_MSIL2A.xml` (`BOA_ADD_OFFSET` real por banda) | Blob do Planetary Computer bloqueado |
| Verificar os rasters do 2025-09-25 já guardados (B04/B08 nas pastas multibanda) | Estão só no Mac. Os metadados listam os ativos mas não permitem verificar os pixels |
| Testar relações entre bandas nos 341 patches de treino | São single-band float32, não contêm B04 nem B08. **Isto não prova que haja B04/B08 autênticas noutro lado** |
| 24 testes marcados `network` | Exigem rede para STAC/COG |
| Teste end-to-end do downloader contra o Planetary Computer | Rede bloqueada. Só as funções de grid e reamostragem foram testadas offline |
| Rebuild (treino e validação RENSUB) | Ver 7 e 8 |

---

## 6. Proveniência e estado radiométrico dos inputs

| Input | Produto / fornecedor | Evidência | Transformação já aplicada | Estado |
|---|---|---|---|---|
| Item Earth Search `S2B_29SNB_20250925_0_L2A` | `S2B_MSIL2A_20250925T112109_N0511_R037_T29SNB_20250925T151904.SAFE`; datatake `GS2B_20250925T112109_044679_N05.11`; baseline 05.11; `sentinel2-to-stac 2025.03.06` | [metadados] | DN uint16, nodata 0; `raster:bands` scale 1e-4, offset −0,1; `earthsearch:boa_offset_applied: true` | **known** (pela regra `raster:bands`). Pixels não lidos |
| Item Planetary Computer `S2B_MSIL2A_20250925T112109_R037_T29SNB_20250925T151904` | Mesmo `product_uri` e datatake; granule `..._A044679_T29SNB_N05.11`; baseline 05.11 | [metadados] | Sem `raster:bands`. Pela doc da Microsoft, o offset está nos dados | **radiometry_unknown** sem o MTD (bloqueado aqui). O mesmo `product_uri` **não** prova valores equivalentes aos do Earth Search |
| 4 pastas multibanda 2025-09-25 (`santa_eulalia_*`, `pedra_eulalia_sep2025_multiband`) | O JSON diz PC e o produto acima | Só [metadados]; rasters no Mac | DN guardado; 20/60 m bilinear num grid próprio (código antigo); radiometria não registada | **radiometry_unknown** e grid não verificado → bloqueadas para uso científico |
| `reef_Output_Master/.../S2_B02/B03_20250925.tif` (orchestrator) | Desconhecido | Sem metadados | Heurística `max>limiar → /10000` | **radiometry_unknown**; produto não identificado |
| `data/master_ml_dataset/images` (341 patches) | Desconhecido (single-band float32, EPSG:32629, 10 m, nodata NaN) | [dados] | Já float, origem da conversão desconhecida | **radiometry_unknown**; não convertíveis outra vez |
| ACOLITE `rhos` 2025-09-25 (citado no plano anterior) | ACOLITE | Não acessível | — | Separado do ensaio Sen2Cor; não usado |

---

## 7. Resultados científicos

**Nenhum.** Não foi lido nenhum pixel real e não houve treino, validação ou métricas. Nenhuma frase deste relatório deve ser lida como desempenho do detector. Uma correção de software testada não equivale a um detector validado. **O detector não está pronto para produção.**

### Rebuild: regras mantidas e estado

- Regras registadas e não executadas: labels RENSUB com proveniência e exclusões acordadas; Pedra da Galé `unresolved_field_site`, excluída do treino; os dois blocos de Santa Eulália (`E570_N4102` e `E570_N4103`) juntos como holdout permanente; blocos de 1 km com buffer de 250 m; ACOLITE separado do Sen2Cor; holdout nunca usado para tuning; intervalos por bootstrap de blocos.
- **Os inputs do rebuild não existem neste ambiente [verificado]:** `src/sdb/rebuild.py`, `scripts/build_rebuild_labels.py`, `scripts/fetch_rebuild_inputs.py`, `data/calibration/sdb/reef_detector/rebuild/` e o commit `7c12eb2` estão só no Mac. Nenhum ramo remoto tem esse commit.
- **Contradição do DTM, resolvida documentalmente.** No plano anterior, a "profundidade" de M1 e M3 é `psdb`: o rácio de Stumpf B02/B03 não calibrado, do mesmo produto, a 10 m. O DTM EMODnet só serve para o filtro de 2–25 m e nunca é feature (havia um teste para isso). Neste repositório, o detector também só usa o EMODnet como **máscara**, nunca como feature.
- **Decisão conservadora:** não encontrei aprovação registada de `psdb` como origem de profundidade para features. Por isso a única variante autorizada é **M2_no_depth**. M1_full e M3_depth_only ficam **não executadas**. Também M2 não correu, por falta de imagem, labels e DTM (este último necessário para o filtro de 2–25 m).

---

## 8. Bloqueios remanescentes e impacto

| Bloqueio | Impacto |
|---|---|
| Proxy recusa os anfitriões de imagem (S3 do Earth Search, blob do PC, CDSE) | Sem pixels reais: não é possível validar o detector, confirmar o offset nos dados, ler o MTD nem executar o rebuild |
| Sem acesso ao Mac | Não é possível inspecionar as alegações H1/H2, os rasters do 2025-09-25 nem o código e labels do rebuild |
| Semântica de `earthsearch:boa_offset_applied: true` junto com `offset −0,1` | Incerteza residual. Seguimos a regra oficial (aplicar o `offset` de `raster:bands`). Só um teste em pixels sobre água escura a pode fechar |
| `GLINT_NIR_MAX = 0.015` nunca foi calibrado com a escala correta | Com a reflectância correta o limiar passa a ter efeito real, mas o seu valor não está validado |
| ~25 scripts legados sem offset | Os outputs desses caminhos ficam radiometry_unknown até migrarem para `s2_radiometry` |
| Aprovação de `psdb` como profundidade | M1 e M3 bloqueadas |

---

## 9. Estado final

**Parcialmente concluído por falta de dados e de acesso.**

- Concluído e testado: radiometria explícita; georreferenciação nativa; nodata e reamostragem; remoção do `Kd_B04` inventado; registry de dependências com `status="unavailable"`; downloader com grid único e proveniência.
- Não concluído: verificação em pixels reais, rebuild e validação científica. Tudo isto está bloqueado pelo acesso, não pelo código.

### Fontes

- [Earth Search — README (gain/offset após 25 jan 2022)](https://github.com/Element84/earth-search/blob/main/README.md)
- [Item Earth Search S2B_29SNB_20250925_0_L2A](https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a/items/S2B_29SNB_20250925_0_L2A)
- [Item Planetary Computer S2B_MSIL2A_20250925T112109_R037_T29SNB_20250925T151904](https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-2-l2a/items/S2B_MSIL2A_20250925T112109_R037_T29SNB_20250925T151904)
- [Microsoft Planetary Computer — baseline-change notebook](https://nbviewer.org/github/microsoft/PlanetaryComputerExamples/blob/main/datasets/sentinel-2-l2a/baseline-change.ipynb)
- [ESA — Sentinel-2 Level-2A Algorithms and Products (BOA_ADD_OFFSET)](https://sentinel.esa.int/web/sentinel/technical-guides/sentinel-2-msi/level-2a-algorithms-products)
