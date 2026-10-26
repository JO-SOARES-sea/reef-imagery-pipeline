# Índice de fontes locais (dados já descarregados)

Gerado por `scripts/index_local_sources.py` em 2026-10-26 a partir de: `/Volumes/4G_WD_BLACK/BACKUP`, `/Volumes/4G_WD_BLACK/dev/UnderwaterVisibility/reports`.

**Antes de pesquisar online, procure aqui.** O índice completo, uma linha por ficheiro com etiquetas, extensão WGS84, CRS e excerto do texto, está em `data/external/local_sources_index.csv`. O texto integral de todos os PDFs está em `data/external/local_sources_text/`. Exemplos:

```bash
grep -i greta data/external/local_sources_index.csv
grep -ril 'recifes artificiais' data/external/local_sources_text/ | head
grep -n -i 'B-24\|B24' data/external/local_sources_text/*Arqueologia*.txt
```

Total: 3220 ficheiros, 45.0 GB.

## Ficheiros por tema

| Tema | Ficheiros |
|---|---|
| `quarteira_vilamoura` | 1629 |
| `garrao_ancao_quinta_lago` | 1542 |
| `aia3587_quarteira_garrao` | 1461 |
| `sonar_lateral_sss` | 1385 |
| `valado_albufeira` | 1096 |
| `multifeixe_mbes` | 958 |
| `batimetria_emodnet_ih` | 366 |
| `ccmar_ipma` | 235 |
| `satelite` | 222 |
| `litologia_rocha` | 215 |
| `patrimonio_naufragios` | 156 |
| `praia_faro_ilha_faro` | 109 |
| `psoem_dgrm` | 68 |
| `greta` | 29 |
| `recifes_artificiais` | 25 |
| `b24_aviao` | 8 |

## Pastas

### `BACKUP/(raiz)`

12 ficheiros, 69.9 MB. Tipos: .png (7), .pdf (2), .mov (1), .jpeg (1), .zip (1). Temas: multifeixe_mbes (2), litologia_rocha (2), batimetria_emodnet_ih (2), satelite (2), greta (1), praia_faro_ilha_faro (1).

- `greta_recife.pdf`: Pedra da Greta deteção por satélite e coordenadas GPS UnderwaterVisibility · 21/10/2026 · Sentinel-2 L2A, órbita R037, tile 29SNA · marcas d
- `valado_completo_alta.pdf`: Parque Natural Marinho do Recife do Algarve Pedra do Valado · rocha RENSUB + limites oficiais + verdade de terreno CCMAR/AMPIC + mergulhos A

### `BACKUP/AIA3587_Quarteira_Garrao`

1356 ficheiros, 351.7 MB. Tipos: .png (1226), .jpg (45), .ppm (20), .pdf (19), .zip (6), .dbf (4). Temas: aia3587_quarteira_garrao (1356), quarteira_vilamoura (1356), garrao_ancao_quinta_lago (1356), sonar_lateral_sss (1299), patrimonio_naufragios (20), batimetria_emodnet_ih (14). Levantamentos COSMO: LTHBAIF1, LTHQUGA1.

- `AIA3587_Quarteira_Garrao/anexoD/N Editavel/22031-GER-001-0.pdf`: 001 | N SIMBOLOGIA: Área de Estudo Zona de Recarga de Areias Biologia - Áreas a evitar para instalação de estruturas de apoio à obra e de ac
- `AIA3587_Quarteira_Garrao/anexoD/N Editavel/22031-GER-002-0.pdf`: 002 | N 2 33 40 30 31 35 41 46 12 32 5 43 3 15 42 6 45 7 9 16;34;22 SIMBOLOGIA: 47 Área de Estudo Zona de Recarga de Areias 10 Mancha de emp
- `AIA3587_Quarteira_Garrao/anexoD/N Editavel/22031-GER-003-0.pdf`: 003 | F 33 N 40 ED C A 30 31 B 35 41 46 12 5 43 3 15 45 SIMBOLOGIA: Área de Estudo Zona de Recarga de Areias Mancha de empréstimo - Buffer d
- `AIA3587_Quarteira_Garrao/anexoD/N Editavel/22031-GER-004-0.pdf`: 004 | N SIMBOLOGIA: Área de Estudo Zona de Recarga de Areias F Mancha de empréstimo - Buffer de 500mt Mancha de empréstimo 33 Ocorrências Pa
- `AIA3587_Quarteira_Garrao/anexoD/N Editavel/22031-GER-005-0.pdf`: 005 | N 24 SIMBOLOGIA: Área de Estudo Zona de Recarga de Areias Mancha de empréstimo - Buffer de 500mt Mancha de empréstimo Ocorrências Patr
- `AIA3587_Quarteira_Garrao/anexoD/N Editavel/22031-GER-006-0.pdf`: 006 | N SIMBOLOGIA: Área de Estudo Zona de Recarga de Areias Mancha de empréstimo - Buffer de 500mt Mancha de empréstimo Ocorrências Patrimo
- `AIA3587_Quarteira_Garrao/anexoD/N Editavel/22031-GER-007-0.pdf`: 007 | N SIMBOLOGIA: Área de Estudo Zona de Recarga de Areias Mancha de empréstimo - Buffer de 500mt Mancha de empréstimo Ocorrências Patrimo
- `AIA3587_Quarteira_Garrao/anexoD/N Editavel/22031-GER-008-0.pdf`: 008 | N 15 45 7 SIMBOLOGIA: Área de Estudo Zona de Recarga de Areias Mancha de empréstimo - Buffer de 500mt Mancha de empréstimo Ocorrências
- `AIA3587_Quarteira_Garrao/extraido/README.md`: AIA3587 — Alimentação artificial do troço costeiro Quarteira–Garrão
- `AIA3587_Quarteira_Garrao/raw/AIA3587_01_DIA.pdf`: FICHA F | Declaração de Impacte Ambiental (Anexo ao TUA) Designação do projeto Alimentação artificial do troço Costeiro Quarteira-Garrão Fas
- `AIA3587_Quarteira_Garrao/raw/AIA3587_03_Consulta_Publica.pdf`: RELATÓRIO DE CONSULTA PÚBLICA Avaliação de Impacte Ambiental n.º 3587 Projeto “Alimentação Artificial do Troço Costeiro Quarteira-Garrão” Ju
- `AIA3587_Quarteira_Garrao/raw/AIA3587_04_RNT.pdf`: Alimentação Artificial do Troço Costeiro Quarteira-Garrão ESTUDO DE IMPACTE AMBIENTAL Volume 2 - Resumo Não Técnico Nº Trabalho: 21031 1/03/

- 🗺 `AIA3587_Quarteira_Garrao/extraido/areas_projeto.geojson`: bbox -8.12603,36.99508,-8.04148,37.07308; EPSG:4326; 5 entidades; campos: nome, area_ha, vertices_pttm06
- 🗺 `AIA3587_Quarteira_Garrao/extraido/ocorrencias_patrimoniais.geojson`: bbox -8.23062,36.97091,-8.00767,37.07678; EPSG:4326; 40 entidades; campos: id, nome, tipo, lat_dms, lon_dms, x_pttm06, y_pttm06, cns
- 🗺 `AIA3587_Quarteira_Garrao/extraido/sidescan_mosaic_PTTM06_2m.tif`: bbox -8.16384,36.94997,-7.99837,37.07457; EPSG:3763; 7357x6910 px, 1 banda(s), res 2.00
- 🗺 `AIA3587_Quarteira_Garrao/shp/shp/Area_Estudo_Total/Area_Estudo_Total.shp`: bbox -8.12603,36.99508,-8.04148,37.07308; EPSG:3763; 2 entidades; campos: Id
- 🗺 `AIA3587_Quarteira_Garrao/shp/shp/Localizacao_Proposta_Estaleiro/Localizacao_Proposta_Estaleiro.shp`: bbox -8.08979,37.06280,-8.08894,37.06312; EPSG:3763; 1 entidades; campos: Id
- 🗺 `AIA3587_Quarteira_Garrao/shp/shp/Mancha_de_Emprestimo/Mancha_de_Emprestimo.shp`: bbox -8.12041,36.99958,-8.09401,37.01977; EPSG:3763; 1 entidades; campos: Id
- 🗺 `AIA3587_Quarteira_Garrao/shp/shp/Zona_de_Recarga_Areias/Zona_de_Recarga_Areias.shp`: bbox -8.11174,37.03052,-8.04468,37.06898; EPSG:3763; 1 entidades; campos: Id
- 🗺 `AIA3587_Quarteira_Garrao/web/data/aia3587_unified.geojson`: bbox -8.23062,36.95009,-7.99856,37.07678; EPSG:4326; 167 entidades; campos: layer, color, id, descricao, nome, ficheiro, nota, prof_z

### `BACKUP/COSMO_LTHPEAL1_Albufeira`

9 ficheiros, 6.1 MB. Tipos: .png (2), .md (1), .csv (1), .geojson (1), .gpx (1), .pdf (1). Temas: valado_albufeira (9), multifeixe_mbes (9), litologia_rocha (6), ccmar_ipma (4), quarteira_vilamoura (2), satelite (2). Levantamentos COSMO: LTHPEAL1.

- `COSMO_LTHPEAL1_Albufeira/README.md`: Afloramentos rochosos no multifeixe COSMO de Albufeira (LTHPEAL1)
- `COSMO_LTHPEAL1_Albufeira/atlas_LTHPEAL1_Albufeira.pdf`: Albufeira (Peneco Alemães) · COSMO LTHPEAL1 2020 (multifeixe 0,3 m, APA) · = topo de cada pedra (ID e profundidade do topo, m ZH); pedras = 

- 🗺 `COSMO_LTHPEAL1_Albufeira/afloramentos.geojson`: bbox -8.25809,37.07444,-8.23288,37.08387; EPSG:4326; 61 entidades; campos: id, tipo, topo_lat, topo_lon, plotter, prof_topo_m, prof_f
- 🗺 `COSMO_LTHPEAL1_Albufeira/afloramentos.gpx`: bbox -8.25774,37.07444,-8.23334,37.08387; EPSG:4326; 55 entidades; campos: ele, time, magvar, geoidheight, name, cmt, desc, src

### `BACKUP/LTHBAIF1_Barreta_Farol`

19 ficheiros, 24.4 MB. Tipos: .png (13), .py (1), .log (1), .md (1), .geojson (1), .json (1). Temas: praia_faro_ilha_faro (19), multifeixe_mbes (19), quarteira_vilamoura (3), batimetria_emodnet_ih (3). Levantamentos COSMO: LTHBAIF1.

- `LTHBAIF1_Barreta_Farol/web/README.md`: LTHBAIF1 — Praia da Barreta / Praia da Ilha do Farol

- 🗺 `LTHBAIF1_Barreta_Farol/web/data/lthbaif1.geojson`: bbox -7.89042,36.94974,-7.84901,36.97821; EPSG:4326; 92 entidades; campos: layer, year, anchor, kind, level_zh, nma, pts, res_m

### `BACKUP/PNSACV_Costa_Vicentina_MARSW`

5 ficheiros, 26.7 MB. Tipos: .pdf (2), .txt (2), .md (1). Temas: ccmar_ipma (3), multifeixe_mbes (2), batimetria_emodnet_ih (2), valado_albufeira (1), sonar_lateral_sss (1), greta (1).

- `PNSACV_Costa_Vicentina_MARSW/LEIA-ME.md`: Parque Marinho do Sudoeste Alentejano e Costa Vicentina (PNSACV) — projeto MARSW
- `PNSACV_Costa_Vicentina_MARSW/brochura.pdf`: FICHA TÉCNICA Reservados todos os direitos de acordo com a legislação em vigor 2021, AUTORES E LIGA PARA A PROTECÇÃO DA NATUREZA AUTORES Coo
- `PNSACV_Costa_Vicentina_MARSW/h._Mapeamento.pdf`: Mapeamento de Habitats e de Espécies na Área Marinha do PNSACV Gonçalves, J.M.S., Oliveira, F., Monteiro, P., Bentes, L., Andrade, M., Belac

### `BACKUP/Parque_Pedra_do_Valado`

968 ficheiros, 2,994.2 MB. Tipos: .pdf (404), .txt (204), .json (95), .html (81), .geojson (48), .md (22). Temas: valado_albufeira (968), multifeixe_mbes (776), batimetria_emodnet_ih (167), ccmar_ipma (153), litologia_rocha (112), patrimonio_naufragios (80). Levantamentos COSMO: LTHNOCR1.

- `Parque_Pedra_do_Valado/README.md`: Parque Natural Marinho do Recife do Algarve – Pedra do Valado: batimetria e anomalias Sentinel-2
- `Parque_Pedra_do_Valado/RELATORIO-Mapeament-PNMRA_PV-CCMAR.pdf`: MAPEAMENTO FÍSICO DAS FUTURAS ÁREAS DE PROTEÇÃO DO PARQUE NATURAL MARINHO RECIFE DO ALGARVE – PEDRA DO VALADO RELATÓRIO TÉCNICO MAPEAMENTO F
- `Parque_Pedra_do_Valado/cosmo_LTHNOCR1/atlas_LTHNOCR1_Porches.pdf`: (sem camada de texto: digitalizado?)
- `Parque_Pedra_do_Valado/pesquisa_dados/01_AMPICCOM_ampiccomprojeto/AMPIC_Ficha-de-Projeto.pdf`: DESIGNAÇÃO DO PROJETO CODIGO DO PROJETO Comunicação e ciência para uma AMPIC (Área Marinha Protegida de Interesse Comunitário) Inovadora – M
- `Parque_Pedra_do_Valado/pesquisa_dados/01_AMPICCOM_ampiccomprojeto/ARTIGO-Co-desing-of-a-marine-protected-area-zoning-and-the-lessons-learned-from-it.pdf`: Co-design of a marine protected area zoning and the lessons learned from it | TYPE Original Research PUBLISHED 10 November 2022 DOI 10.3389/
- `Parque_Pedra_do_Valado/pesquisa_dados/01_AMPICCOM_ampiccomprojeto/RELATORIO-AMPIC-Corolario.pdf`: 1 | DADOS E NÚMEROS DE SUPORTE A UMA ÁREA MARINHA PROTEGIDA NO ALGARVE ↓ → ↓ → ↓ ↓ ↑ ↓ → 2 | DADOS E NÚMEROS DE SUPORTE A UMA ÁREA MARINHA P
- `Parque_Pedra_do_Valado/pesquisa_dados/01_AMPICCOM_ampiccomprojeto/RELATORIO-AMPIC-Final.pdf`: Comunicação e Ciência para uma AMPIC inovadora RELATÓRIO-FINAL DE PROJETO | MAIO 2024 Ficha Técnica Projeto: Comunicação e Ciência para uma 
- `Parque_Pedra_do_Valado/pesquisa_dados/01_AMPICCOM_ampiccomprojeto/RELATORIO-AMPIC-Proposta-de-Criacao-PNMRAPV.pdf`: Pedido de criação do PARQUE NATURAL MARINHO DO RECIFE DO ALGARVE PEDRA DO VALADO AMPIC – Área Marinha Protegida de Interesse Comunitário mar
- `Parque_Pedra_do_Valado/pesquisa_dados/01_AMPICCOM_ampiccomprojeto/RELATORIO-AMPIC-Value-final-report-2025-1.pdf`: Mapeamento e valoração das atividades suportadas pela costa de Albufeira, Lagoa e Silves com vista a informar o processo participativo para 
- `Parque_Pedra_do_Valado/pesquisa_dados/01_AMPICCOM_ampiccomprojeto/RELATORIO-AMPIC-Value-resumo-alargado-2025.pdf`: RESUMO ALARGADO Mapeamento e valoração das atividades suportadas pela costa de Albufeira, Lagoa e Silves com vista a informar o processo par
- `Parque_Pedra_do_Valado/pesquisa_dados/01_AMPICCOM_ampiccomprojeto/RELATORIO-AMPICCOM-Memoria-Descritiva.pdf`: MEMÓRIA DESCRITIVA PROJETO Comunicação e ciência para uma AMPIC Inovadora PROMOTOR Centro de Ciências do Mar do Algarve PO MAR 2020 – PROGRA
- `Parque_Pedra_do_Valado/pesquisa_dados/01_AMPICCOM_ampiccomprojeto/RELATORIO-Executive-Summary-EN.pdf`: BAÍA DE ARMAÇÃO DE PÊRA Baseline study on the natural values and human uses of the marine space FARO | March | 2018 Executive summary The Fu

- 🗺 `Parque_Pedra_do_Valado/EMODnet_batimetria_Pedra_do_Valado.tif`: bbox -8.50000,36.98021,-8.25000,37.11979; EPSG:4326; 240x134 px, 1 banda(s), res 0.00
- 🗺 `Parque_Pedra_do_Valado/cosmo_LTHNOCR1/afloramentos.geojson`: bbox -8.39061,37.08708,-8.37335,37.09803; EPSG:4326; 199 entidades; campos: id, tipo, topo_lat, topo_lon, plotter, prof_topo_m, prof_
- 🗺 `Parque_Pedra_do_Valado/cosmo_LTHNOCR1/afloramentos.gpx`: bbox -8.39053,37.08708,-8.37418,37.09760; EPSG:4326; 178 entidades; campos: ele, time, magvar, geoidheight, name, cmt, desc, src
- 🗺 `Parque_Pedra_do_Valado/pesquisa_dados/00_SINTESE_pontos_verdade_terreno/pontos_campo_substrato_PNMRA-PV_-8.50_-8.24.geojson`: bbox -8.49280,36.97660,-8.24020,37.09840; EPSG:4326; 397 entidades; campos: fonte, id, prof_m, substrato, classe, data, metodo, licen
- 🗺 `Parque_Pedra_do_Valado/pesquisa_dados/01_AMPICCOM_ampiccomprojeto/coordenadas_extraidas/AMPIC_TabA2.1_mergulhos_censos_visuais_2021-2022.geojson`: bbox -8.64051,37.03738,-8.28874,37.10508; EPSG:4326; 26 entidades; campos: id, zona, seccao, prof_media_m, prof_max_m, datas, obs, me
- 🗺 `Parque_Pedra_do_Valado/pesquisa_dados/01_AMPICCOM_ampiccomprojeto/coordenadas_extraidas/AMPIC_TabA2.7_camaras_iscadas_BRUV_substrato_2021-2022.geojson`: bbox -8.62732,37.02611,-8.28541,37.10730; EPSG:4326; 75 entidades; campos: id, local, prof_m, substrato, complexidade, obs, zona, sec
- 🗺 `Parque_Pedra_do_Valado/pesquisa_dados/03_RCM_1-2024_limites_zonamento/ICNF_RNAP_WFS_PNMRA-PV_limite_oficial_EPSG4326.geojson`: bbox -8.44254,36.99969,-8.26111,37.10169; EPSG:4326; 1 entidades; campos: id, nome_ap, classifica, publica1, publica2, jurisdicao, ar
- 🗺 `Parque_Pedra_do_Valado/pesquisa_dados/03_RCM_1-2024_limites_zonamento/RCM_1-2024_PNMRA-PV_Anexo_I_coordenadas.geojson`: bbox -8.44254,36.99969,-8.26125,37.08591; EPSG:4326; 10 entidades; campos: nome, ponto, descricao, fonte, datum, exato, nota

### `BACKUP/ROCHAS`

2 ficheiros, 1.2 MB. Tipos: .png (2). Temas: litologia_rocha (2), quarteira_vilamoura (1).

### `BACKUP/bulk_download_2026-09-28_04-21-05`

12 ficheiros, 4,437.8 MB. Tipos: .tif (6), .xml (6). Temas: valado_albufeira (12), multifeixe_mbes (12).

- 🗺 `bulk_download_2026-09-28_04-21-05/LTHPEAL1_20181020_20181024_DEM.tif`: bbox -8.25892,37.07363,-8.23286,37.08706; EPSG:3763; 7718x4962 px, 1 banda(s), res 0.30
- 🗺 `bulk_download_2026-09-28_04-21-05/LTHPEAL1_20181020_20181024_ORTHO.tif`: bbox -8.25905,37.08117,-8.23186,37.08752; EPSG:3763; 80563x23402 px, 3 banda(s), res 0.03
- 🗺 `bulk_download_2026-09-28_04-21-05/LTHPEAL1_20191020_20191021_DEM.tif`: bbox -8.25888,37.07363,-8.23283,37.08708; EPSG:3763; 7716x4967 px, 1 banda(s), res 0.30
- 🗺 `bulk_download_2026-09-28_04-21-05/LTHPEAL1_20191020_20191021_ORTHO.tif`: bbox -8.25905,37.08117,-8.23186,37.08752; EPSG:3763; 80563x23402 px, 3 banda(s), res 0.03
- 🗺 `bulk_download_2026-09-28_04-21-05/LTHPEAL1_20201113_20201117_DEM.tif`: bbox -8.25888,37.07363,-8.23283,37.08708; EPSG:3763; 7715x4967 px, 1 banda(s), res 0.30
- 🗺 `bulk_download_2026-09-28_04-21-05/LTHPEAL1_20201113_20201117_ORTHO.tif`: bbox -8.25905,37.08117,-8.23186,37.08752; EPSG:3763; 80563x23402 px, 3 banda(s), res 0.03

### `BACKUP/bulk_download_2026-09-28_04-23-11`

31 ficheiros, 7,420.3 MB. Tipos: .pdf (12), .tif (6), .xml (6), .zip (2), .kmz (2), .csv (2). Temas: aia3587_quarteira_garrao (16), quarteira_vilamoura (16), garrao_ancao_quinta_lago (14), multifeixe_mbes (13), patrimonio_naufragios (9), batimetria_emodnet_ih (8).

- `bulk_download_2026-09-28_04-23-11/AIA3587_01_DIA.pdf`: FICHA F | Declaração de Impacte Ambiental (Anexo ao TUA) Designação do projeto Alimentação artificial do troço Costeiro Quarteira-Garrão Fas
- `bulk_download_2026-09-28_04-23-11/AIA3587_03_Consulta_Publica.pdf`: RELATÓRIO DE CONSULTA PÚBLICA Avaliação de Impacte Ambiental n.º 3587 Projeto “Alimentação Artificial do Troço Costeiro Quarteira-Garrão” Ju
- `bulk_download_2026-09-28_04-23-11/AIA3587_04_RNT.pdf`: Alimentação Artificial do Troço Costeiro Quarteira-Garrão ESTUDO DE IMPACTE AMBIENTAL Volume 2 - Resumo Não Técnico Nº Trabalho: 21031 1/03/
- `bulk_download_2026-09-28_04-23-11/AIA3587_05_Aditamento.pdf`: Alimentação Artificial do Troço Costeiro Quarteira-Garrão ESTUDO DE IMPACTE AMBIENTAL Aditamento ao EIA Nº Trabalho: 21031 19/04/2023 Alimen
- `bulk_download_2026-09-28_04-23-11/AIA3587_06_RS_Vol1.pdf`: ALIMENTAÇÃO ARTIFICIAL DO TROÇO COSTEIRO QUARTEIRA- GARRÃO Estudo de Impacte Ambiental Volume 1 - Relatório Síntese Trabalho: T21.031 Data: 
- `bulk_download_2026-09-28_04-23-11/AIA3587_07_Desenhos_Vol4.pdf`: Alimentação Artificial do Troço Costeiro Quarteira-Garrão ESTUDO DE IMPACTE AMBIENTAL Volume 4- Peças Desenhadas Nº Trabalho: 21031 Data: 02
- `bulk_download_2026-09-28_04-23-11/AIA3587_08_Anexos_Tecnicos_Vol3.pdf`: Alimentação Artificial do Troço Costeiro Quarteira-Garrão ESTUDO DE IMPACTE AMBIENTAL Volume 3 - Anexos Técnicos Nº Trabalho: 21031 Data: 02
- `bulk_download_2026-09-28_04-23-11/AIA3587_09_AnexoC_Arqueologia.pdf`: Projecto Porto Novo - Santo Antão | RELATÓRIO DO ESTUDO DE IMPACTE AMBIENTAL DA ALIMENTAÇÃO ARTIFICIAL DO TROÇO QUARTEIRA – GARRÃO PATRIMÓNI
- `bulk_download_2026-09-28_04-23-11/AIA3587_11_PSS.pdf`: ALIMENTAÇÃO ARTIFICIAL DO TROÇO COSTEIRO QUARTEIRA- GARRÃO (LOULÉ) Projeto de Execução Plano de Segurança e Saúde Nº Trabalho: 21031 Data: 0
- `bulk_download_2026-09-28_04-23-11/AIA3587_12_Indice.pdf`: Alimentação Artificial do Troço Costeiro Quarteira-Garrão ESTUDO DE IMPACTE AMBIENTAL Volume 5 - Índice de ficheiros Nº Trabalho: 21031 Data
- `bulk_download_2026-09-28_04-23-11/AIA3587_13_PPGRCD.pdf`: ALIMENTAÇÃO ARTIFICIAL DO TROÇO COSTEIRO QUARTEIRA- GARRÃO Projeto de Execução Plano de Prevenção e Gestão de Resíduos da Construção e Demol
- `bulk_download_2026-09-28_04-23-11/n .pdf`: Untitled | PARECER DA COMISSÃO DE AVALIAÇÃO <Alimentação artificial do troço Costeiro Quarteira-Garrão= em fase de projeto de execução COMIS

- 🗺 `bulk_download_2026-09-28_04-23-11/LTHQUGA1_20181003_20181013_DEM.tif`: bbox -8.10066,37.02610,-8.04831,37.06283; EPSG:3763; 15514x13576 px, 1 banda(s), res 0.30
- 🗺 `bulk_download_2026-09-28_04-23-11/LTHQUGA1_20191017_20191019_ORTHO.tif`: bbox -8.09159,37.03825,-8.04822,37.06308; EPSG:3763; 128543x91757 px, 3 banda(s), res 0.03
- 🗺 `bulk_download_2026-09-28_04-23-11/LTHQUGA1_20200911_20201111_DEM.tif`: bbox -8.10066,37.02610,-8.04832,37.06283; EPSG:3763; 15512x13578 px, 1 banda(s), res 0.30
- 🗺 `bulk_download_2026-09-28_04-23-11/LTHQUGA1_20200911_20201111_ORTHO.tif`: bbox -8.09228,37.03817,-8.04818,37.06298; EPSG:3763; 130689x91720 px, 3 banda(s), res 0.03
- 🗺 `bulk_download_2026-09-28_04-23-11/LTHQUGA1_20250612_20250616_DEM.tif`: bbox -8.11647,37.02616,-8.04812,37.06903; EPSG:3763; 20258x15851 px, 1 banda(s), res 0.30
- 🗺 `bulk_download_2026-09-28_04-23-11/LTHQUGA1_20250612_20250616_ORTHO.tif`: bbox -8.11094,37.03791,-8.04727,37.06956; EPSG:3763; 113228x70197 px, 4 banda(s), res 0.05

### `BACKUP/bulk_download_2026-09-30_22-06-05`

10 ficheiros, 1,550.4 MB. Tipos: .tif (5), .xml (5). Temas: multifeixe_mbes (10).

- 🗺 `bulk_download_2026-09-30_22-06-05/LTHNOCR1_20181024_20181101_DEM.tif`: bbox -8.39123,37.08657,-8.37302,37.09900; EPSG:3763; 5387x4586 px, 1 banda(s), res 0.30
- 🗺 `bulk_download_2026-09-30_22-06-05/LTHNOCR1_20191021_20191022_DEM.tif`: bbox -8.39124,37.08656,-8.37302,37.09900; EPSG:3763; 5387x4586 px, 1 banda(s), res 0.30
- 🗺 `bulk_download_2026-09-30_22-06-05/LTHNOCR1_20191021_20191022_ORTHO.tif`: bbox -8.39160,37.09237,-8.37762,37.09967; EPSG:3763; 41361x26900 px, 3 banda(s), res 0.03
- 🗺 `bulk_download_2026-09-30_22-06-05/LTHNOCR1_20201113_20201118_DEM.tif`: bbox -8.39123,37.08657,-8.37302,37.09900; EPSG:3763; 5387x4585 px, 1 banda(s), res 0.30
- 🗺 `bulk_download_2026-09-30_22-06-05/LTHNOCR1_20201113_20201118_ORTHO.tif`: bbox -8.39160,37.09237,-8.37762,37.09967; EPSG:3763; 41361x26900 px, 3 banda(s), res 0.03

### `BACKUP/bulk_download_2026-09-30_22-14-35`

12 ficheiros, 4,437.8 MB. Tipos: .tif (6), .xml (6). Temas: valado_albufeira (12), multifeixe_mbes (12).

- 🗺 `bulk_download_2026-09-30_22-14-35/LTHPEAL1_20181020_20181024_DEM.tif`: bbox -8.25892,37.07363,-8.23286,37.08706; EPSG:3763; 7718x4962 px, 1 banda(s), res 0.30
- 🗺 `bulk_download_2026-09-30_22-14-35/LTHPEAL1_20181020_20181024_ORTHO.tif`: bbox -8.25905,37.08117,-8.23186,37.08752; EPSG:3763; 80563x23402 px, 3 banda(s), res 0.03
- 🗺 `bulk_download_2026-09-30_22-14-35/LTHPEAL1_20191020_20191021_DEM.tif`: bbox -8.25888,37.07363,-8.23283,37.08708; EPSG:3763; 7716x4967 px, 1 banda(s), res 0.30
- 🗺 `bulk_download_2026-09-30_22-14-35/LTHPEAL1_20191020_20191021_ORTHO.tif`: bbox -8.25905,37.08117,-8.23186,37.08752; EPSG:3763; 80563x23402 px, 3 banda(s), res 0.03
- 🗺 `bulk_download_2026-09-30_22-14-35/LTHPEAL1_20201113_20201117_DEM.tif`: bbox -8.25888,37.07363,-8.23283,37.08708; EPSG:3763; 7715x4967 px, 1 banda(s), res 0.30
- 🗺 `bulk_download_2026-09-30_22-14-35/LTHPEAL1_20201113_20201117_ORTHO.tif`: bbox -8.25905,37.08117,-8.23186,37.08752; EPSG:3763; 80563x23402 px, 3 banda(s), res 0.03

### `BACKUP/bulk_download_2026-10-02_02-35-48`

12 ficheiros, 13,362.9 MB. Tipos: .tif (6), .xml (6). Temas: praia_faro_ilha_faro (12), multifeixe_mbes (12).

- 🗺 `bulk_download_2026-10-02_02-35-48/LTHBAIF1_20181016_20181022_DEM.tif`: bbox -7.89024,36.94974,-7.84901,36.97809; EPSG:3763; 12211x10452 px, 1 banda(s), res 0.30
- 🗺 `bulk_download_2026-10-02_02-35-48/LTHBAIF1_20181016_20181022_ORTHO.tif`: bbox -7.89042,36.95927,-7.85796,36.97821; EPSG:3763; 96161x69803 px, 3 banda(s), res 0.03
- 🗺 `bulk_download_2026-10-02_02-35-48/LTHBAIF1_20191010_20191015_DEM.tif`: bbox -7.89024,36.94974,-7.84901,36.97809; EPSG:3763; 12211x10453 px, 1 banda(s), res 0.30
- 🗺 `bulk_download_2026-10-02_02-35-48/LTHBAIF1_20191010_20191015_ORTHO.tif`: bbox -7.89042,36.95927,-7.85796,36.97821; EPSG:3763; 96161x69803 px, 3 banda(s), res 0.03
- 🗺 `bulk_download_2026-10-02_02-35-48/LTHBAIF1_20201023_20201029_DEM.tif`: bbox -7.89024,36.94974,-7.84901,36.97809; EPSG:3763; 12211x10453 px, 1 banda(s), res 0.30
- 🗺 `bulk_download_2026-10-02_02-35-48/LTHBAIF1_20201023_20201029_ORTHO.tif`: bbox -7.89042,36.95927,-7.85796,36.97821; EPSG:3763; 96161x69803 px, 3 banda(s), res 0.03

### `BACKUP/bulk_download_2026-10-20_22-57-24`

14 ficheiros, 4,712.5 MB. Tipos: .tif (7), .xml (7). Temas: praia_faro_ilha_faro (14).

- 🗺 `bulk_download_2026-10-20_22-57-24/LTHFA1_20181010_20181014_DEM.tif`: bbox -8.00849,36.98884,-7.97860,37.01073; EPSG:3763; 8856x8083 px, 1 banda(s), res 0.30
- 🗺 `bulk_download_2026-10-20_22-57-24/LTHFA1_20181010_20181014_ORTHO.tif`: bbox -7.99995,36.99681,-7.97837,37.01081; EPSG:3763; 63946x51686 px, 3 banda(s), res 0.03
- 🗺 `bulk_download_2026-10-20_22-57-24/LTHFA1_20191016_20191016_ORTHO.tif`: bbox -7.99995,36.99681,-7.97837,37.01081; EPSG:3763; 63946x51686 px, 3 banda(s), res 0.03
- 🗺 `bulk_download_2026-10-20_22-57-24/LTHFA1_20201104_20201110_DEM.tif`: bbox -8.00850,36.98884,-7.97861,37.01073; EPSG:3763; 8856x8083 px, 1 banda(s), res 0.30
- 🗺 `bulk_download_2026-10-20_22-57-24/LTHFA1_20201104_20201110_ORTHO.tif`: bbox -8.00043,36.99665,-7.97835,37.01079; EPSG:3763; 65437x52215 px, 3 banda(s), res 0.03
- 🗺 `bulk_download_2026-10-20_22-57-24/LTHFA1_20250617_DEM.tif`: bbox -8.00855,36.98880,-7.97889,37.01052; EPSG:3763; 8788x8021 px, 1 banda(s), res 0.30
- 🗺 `bulk_download_2026-10-20_22-57-24/LTHFA1_20250617_ORTHO.tif`: bbox -8.00029,36.99659,-7.97818,37.01102; EPSG:3763; 39301x31960 px, 4 banda(s), res 0.05

### `BACKUP/bulk_download_2026-10-20_23-25-33`

14 ficheiros, 4,712.5 MB. Tipos: .tif (7), .xml (7). Temas: praia_faro_ilha_faro (14).

- 🗺 `bulk_download_2026-10-20_23-25-33/LTHFA1_20181010_20181014_DEM.tif`: bbox -8.00849,36.98884,-7.97860,37.01073; EPSG:3763; 8856x8083 px, 1 banda(s), res 0.30
- 🗺 `bulk_download_2026-10-20_23-25-33/LTHFA1_20181010_20181014_ORTHO.tif`: bbox -7.99995,36.99681,-7.97837,37.01081; EPSG:3763; 63946x51686 px, 3 banda(s), res 0.03
- 🗺 `bulk_download_2026-10-20_23-25-33/LTHFA1_20191016_20191016_ORTHO.tif`: bbox -7.99995,36.99681,-7.97837,37.01081; EPSG:3763; 63946x51686 px, 3 banda(s), res 0.03
- 🗺 `bulk_download_2026-10-20_23-25-33/LTHFA1_20201104_20201110_DEM.tif`: bbox -8.00850,36.98884,-7.97861,37.01073; EPSG:3763; 8856x8083 px, 1 banda(s), res 0.30
- 🗺 `bulk_download_2026-10-20_23-25-33/LTHFA1_20201104_20201110_ORTHO.tif`: bbox -8.00043,36.99665,-7.97835,37.01079; EPSG:3763; 65437x52215 px, 3 banda(s), res 0.03
- 🗺 `bulk_download_2026-10-20_23-25-33/LTHFA1_20250617_DEM.tif`: bbox -8.00855,36.98880,-7.97889,37.01052; EPSG:3763; 8788x8021 px, 1 banda(s), res 0.30
- 🗺 `bulk_download_2026-10-20_23-25-33/LTHFA1_20250617_ORTHO.tif`: bbox -8.00029,36.99659,-7.97818,37.01102; EPSG:3763; 39301x31960 px, 4 banda(s), res 0.05

### `BACKUP/data1`

9 ficheiros, 1.7 MB. Tipos: .json (9). Temas: batimetria_emodnet_ih (2), satelite (2), valado_albufeira (1), sonar_lateral_sss (1).

### `BACKUP/data2`

9 ficheiros, 24.6 MB. Tipos: .json (9). Temas: satelite (2), valado_albufeira (1), sonar_lateral_sss (1), batimetria_emodnet_ih (1).

### `BACKUP/data3`

7 ficheiros, 0.1 MB. Tipos: .py (5), .md (1), .sh (1). Temas: batimetria_emodnet_ih (4).

- `data3/README_projeto_mar.md`: Projecto Mar — Algarve Marine Data Stack

### `BACKUP/doc-q`

34 ficheiros, 208.7 MB. Tipos: .pdf (25), .png (6), .md (2), .txt (1). Temas: aia3587_quarteira_garrao (23), quarteira_vilamoura (20), garrao_ancao_quinta_lago (19), batimetria_emodnet_ih (19), patrimonio_naufragios (15), psoem_dgrm (14).

- `doc-q/AIA_3660_molhes/DIA_AIA3660.pdf`: FICHA F | Declaração de Impacte Ambiental (Anexo ao TUA) Designação do projeto Evolução da Linha de Costa – Reestruturação dos Molhes de Qua
- `doc-q/AIA_3660_molhes/RCP_AIA3660.pdf`: RELATÓRIO DE CONSULTA PÚBLICA Avaliação de Impacte Ambiental n.º 3660 Evolução da Linha de Costa - Reestruturação dos molhes de Quarteira 1 
- `doc-q/AIA_3660_molhes/Volume_III_Anexos.pdf`: ESTUDO DE IMPACTE AMBIENTAL Estudo Prévio da “Evolução da Linha de Costa – Projeto de Execução para a Reestruturação dos Molhes de Quarteira
- `doc-q/AIA_3660_molhes/Volume_I_RS.pdf`: ESTUDO DE IMPACTE AMBIENTAL Estudo da Evolução da Linha de Costa – Projeto de Execução para a Reestruturação dos Molhes de Quarteira VOLUME 
- `doc-q/DGRM_TUPEM/Desenho_1_V2.pdf`: (sem camada de texto: digitalizado?)
- `doc-q/DGRM_TUPEM/Desenho_2_Levantamento_Hidrografico_Portimao.pdf`: (sem camada de texto: digitalizado?)
- `doc-q/DGRM_TUPEM/Desenho_3_Arqueologia_Portimao.pdf`: (sem camada de texto: digitalizado?)
- `doc-q/DGRM_TUPEM/Edital_PT2026.pdf`: f_2984250_1.pdf | (sem camada de texto: digitalizado?)
- `doc-q/DGRM_TUPEM/Fixacao_estrutura_georeferencia.pdf`: Resposta ao pedido de esclarecimento i. i Deverá ser enviada informação relativa às estruturas previstas colocar em espaço marítimo nacional
- `doc-q/DGRM_TUPEM/Mapa_quantidades.pdf`: (Mapa de quantidades - Alimentação artificial Quarteira - Garrão_versão 21_11.xlsx) | Agência Portuguesa do Ambiente I.P. / Administração da
- `doc-q/DGRM_TUPEM/Memoria_Descritiva.pdf`: CONTRIBUTO PARA A DEFINIÇÃO DA COTA DA LINHA DA MÁXIMA PREIA-MAR DE ÁGUAS EQUINOCIAIS NAS PRAIAS DO LITORAL MERIDIONAL DO ALGARVE | Candidat
- `doc-q/DGRM_TUPEM/PAL_Quarteira_Garrao.pdf`: _PAL-XXI_Actualizacao_FINAL_actnov19.pdf | (sem camada de texto: digitalizado?)

### `BACKUP/hager` — sem relação com o projeto

5 ficheiros, 298.5 MB. Tipos: .dmg (2), .txa (1), .ezt (1), .txt (1).

### `BACKUP/re-cases` — sem relação com o projeto

6 ficheiros, 0.0 MB. Tipos: .txt (4), .md (2).

- `re-cases/diveroid-20260821/output/DIVEROID_SPEC.md`: DIVEROID — behavioural specification for a clean-room reimplementation
- `re-cases/diveroid-20260821/output/RISKS_REGISTER.md`: DIVEROID — risks register

### `BACKUP/recovery`

17 ficheiros, 0.1 MB. Tipos: .py (6), .md (4), .log (3), (sem) (1), .sh (1), .yml (1). Temas: sonar_lateral_sss (3), batimetria_emodnet_ih (2), patrimonio_naufragios (2).

- `recovery/furuno_gp30/FURUNO_GP30_TOOL.md`: Reverse Engineering: `furuno.EXE` (Furuno GP30 ↔ OziExplorer Transfer Tool)
- `recovery/furuno_gp30/REVERSE_ENGINEERING_REPORT.md`: SeaMarks — Relatório de Reverse Engineering
- `recovery/furuno_gp30/reverse_eng_extracted/reverse/REVERSE_ENGINEERING_REPORT.md`: SeaMarks — Relatório de Reverse Engineering
- `recovery/nsfw_scanner/README.md`: NSFW Video Scanner (macOS)

### `BACKUP/reef_rebuild_handoff_20261010`

9 ficheiros, 0.5 MB. Tipos: .md (5), (sem) (1), .txt (1), .json (1), .patch (1). Temas: satelite (5), batimetria_emodnet_ih (4), ccmar_ipma (4), patrimonio_naufragios (1).

- `reef_rebuild_handoff_20261010/AUDIT_TRAIL.md`: Audit trail: observations behind the rebuild (2026-10-10)
- `reef_rebuild_handoff_20261010/NETWORK_ACCESS_REQUEST.md`: Network access request: reef-detector rebuild (draft for review, not sent)
- `reef_rebuild_handoff_20261010/PREREGISTRATION.md`: Reef-detector rebuild: pre-registration (v1, 2026-10-10)
- `reef_rebuild_handoff_20261010/REEF_DETECTOR_REBUILD_VALIDATION.md`: Reef detector rebuild: validation report
- `reef_rebuild_handoff_20261010/RESUME.md`: Resuming the reef-detector rebuild

### `BACKUP/reef_rebuild_handoff_20261010_v2`

9 ficheiros, 10.5 MB. Tipos: .md (5), (sem) (1), .json (1), .geojson (1), .patch (1). Temas: satelite (5), batimetria_emodnet_ih (4), ccmar_ipma (4), patrimonio_naufragios (1), aia3587_quarteira_garrao (1).

- `reef_rebuild_handoff_20261010_v2/AUDIT_TRAIL.md`: Audit trail: observations behind the rebuild (2026-10-10)
- `reef_rebuild_handoff_20261010_v2/NETWORK_ACCESS_REQUEST.md`: Network access request: reef-detector rebuild (v2, draft for review, NOT sent)
- `reef_rebuild_handoff_20261010_v2/PREREGISTRATION.md`: Reef-detector rebuild: pre-registration (v1.1, 2026-10-10)
- `reef_rebuild_handoff_20261010_v2/REEF_DETECTOR_REBUILD_VALIDATION.md`: Reef detector rebuild: validation report
- `reef_rebuild_handoff_20261010_v2/RESUME.md`: Resuming the reef-detector rebuild

- 🗺 `reef_rebuild_handoff_20261010_v2/label_geoms.geojson`: bbox -8.67166,36.94560,-7.95350,37.12193; EPSG:32629; 3 entidades; campos: class

### `BACKUP/reef_rebuild_results`

52 ficheiros, 76.5 MB. Tipos: .json (41), .md (2), .npz (2), .py (2), .gz (2), .npy (1). Temas: batimetria_emodnet_ih (9), satelite (8), ccmar_ipma (5), valado_albufeira (1).

- `reef_rebuild_results/extract/data/calibration/sdb/reef_detector/REEF_DETECTOR_REBUILD_VALIDATION.md`: Reef detector rebuild: validation report
- `reef_rebuild_results/extract/data/calibration/sdb/reef_detector/rebuild/runs/20261014T185306Z/09_report/RESULTS.md`: Rebuild run 20261014T185306Z

### `reports/(raiz)`

2 ficheiros, 0.0 MB. Tipos: .md (2). Temas: quarteira_vilamoura (1), valado_albufeira (1).

- `baseline_v5.md`: Initial Baseline Validation (Pre-ML)
- `benchmark_all_sites_20260502_005853.md`: Model Validation Report

### `reports/2026-09-25_santa_eulalia_s2`

59 ficheiros, 21.7 MB. Tipos: .py (17), .tif (13), .json (11), .png (9), .pyc (4), .npy (3). Temas: batimetria_emodnet_ih (12), satelite (10), aia3587_quarteira_garrao (2), quarteira_vilamoura (1).

- `2026-09-25_santa_eulalia_s2/README.md`: Santa Eulália — recifes confirmados em terreno e assinatura Sentinel-2 persistente
- `2026-09-25_santa_eulalia_s2/extra/README.md`: Extras do estudo Santa Eulália (25/09/2026)

- 🗺 `2026-09-25_santa_eulalia_s2/se_2025-09-25.tif`: bbox -8.22183,37.05998,-8.19915,37.07816; EPSG:32629; 200x200 px, 5 banda(s), res 10.00
- 🗺 `2026-09-25_santa_eulalia_s2/se_2026-09-20.tif`: bbox -8.22183,37.05998,-8.19915,37.07816; EPSG:32629; 200x200 px, 5 banda(s), res 10.00
- 🗺 `2026-09-25_santa_eulalia_s2/se_2026-09-25.tif`: bbox -8.22183,37.05998,-8.19915,37.07816; EPSG:32629; 200x200 px, 5 banda(s), res 10.00
- 🗺 `2026-09-25_santa_eulalia_s2/work/deepA_37.025_2025-09-25.tif`: bbox -8.21567,37.02046,-8.20433,37.02954; EPSG:32629; 100x100 px, 16 banda(s), res 10.00
- 🗺 `2026-09-25_santa_eulalia_s2/work/deepA_37.025_2026-09-20.tif`: bbox -8.21567,37.02046,-8.20433,37.02954; EPSG:32629; 100x100 px, 16 banda(s), res 10.00
- 🗺 `2026-09-25_santa_eulalia_s2/work/deepA_37.025_2026-09-25.tif`: bbox -8.21567,37.02046,-8.20433,37.02954; EPSG:32629; 100x100 px, 16 banda(s), res 10.00
- 🗺 `2026-09-25_santa_eulalia_s2/work/deepB_36.985_2025-09-25.tif`: bbox -8.21566,36.98046,-8.20434,36.98954; EPSG:32629; 100x100 px, 16 banda(s), res 10.00
- 🗺 `2026-09-25_santa_eulalia_s2/work/deepB_36.985_2026-09-20.tif`: bbox -8.21566,36.98046,-8.20434,36.98954; EPSG:32629; 100x100 px, 16 banda(s), res 10.00

### `reports/2026-09-26_acolite_glint`

27 ficheiros, 0.1 MB. Tipos: .log (11), .txt (6), .py (3), .json (3), .sh (3), .md (1). Temas: satelite (14).

- `2026-09-26_acolite_glint/README.md`: ACOLITE na Santa Eulália: artefacto de glint B12 (26/09/2026)

### `reports/2026-09-26_quarteira_faro_candidates`

125 ficheiros, 69.5 MB. Tipos: .py (36), .json (31), .png (22), .log (14), .md (5), .csv (5). Temas: quarteira_vilamoura (125), satelite (40), batimetria_emodnet_ih (23), praia_faro_ilha_faro (4), litologia_rocha (4), aia3587_quarteira_garrao (2).

- `2026-09-26_quarteira_faro_candidates/README.md`: Quarteira – Ilha de Faro: anomalias persistentes do fundo (candidatas)
- `2026-09-26_quarteira_faro_candidates/README_candidates.md`: Quarteira – Ilha de Faro: candidatas a afloramentos rochosos submersos (anomalias persistentes do fundo)
- `2026-09-26_quarteira_faro_candidates/confirm_2018_2022/README.md`: Réplica congelada 2018–2022 (DL-021 / DL-022)
- `2026-09-26_quarteira_faro_candidates/confirm_2018_2022/summary_table.md`: | id | kind | lat | lon | pooled_dark_valid | pooled_p0 | pooled_p | R037_dark_valid | R037_p | R137_dark_valid | R137_p | median_zB08 | max
- `2026-09-26_quarteira_faro_candidates/views/README.md`: Vistas de candidatos (2 km, 4 km, 10 km)

- 🗺 `2026-09-26_quarteira_faro_candidates/candidates.geojson`: bbox -8.21079,36.94139,-7.84933,37.07420; EPSG:4326; 315 entidades; campos: cand_id, discovery_id, cluster, list, tier, lat, lon, emo
- 🗺 `2026-09-26_quarteira_faro_candidates/views/regen_2026-10-20/qf_t204_4km/qf_t204_2024-12-29.tif`: bbox -8.07583,37.00918,-8.03041,37.04559; EPSG:32629; 400x400 px, 5 banda(s), res 10.00
- 🗺 `2026-09-26_quarteira_faro_candidates/views/regen_2026-10-20/qf_t204_4km/qf_t204_2025-09-25.tif`: bbox -8.07583,37.00918,-8.03041,37.04559; EPSG:32629; 400x400 px, 5 banda(s), res 10.00
- 🗺 `2026-09-26_quarteira_faro_candidates/views/regen_2026-10-20/qf_t204_4km/qf_t204_2025-10-02.tif`: bbox -8.07583,37.00918,-8.03041,37.04559; EPSG:32629; 400x400 px, 5 banda(s), res 10.00
- 🗺 `2026-09-26_quarteira_faro_candidates/work/emodnet_aoi.tif`: bbox -8.18021,36.88021,-7.80000,37.11979; EPSG:4326; 365x230 px, 1 banda(s), res 0.00
- 🗺 `2026-09-26_quarteira_faro_candidates/work/emodnet_sebox.tif`: bbox -8.30000,36.95000,-8.11979,37.11979; EPSG:4326; 173x163 px, 1 banda(s), res 0.00

### `reports/2026-09-26_zimbral_clarity`

32 ficheiros, 9.7 MB. Tipos: .json (11), .py (10), .csv (4), .png (4), .md (1), .npz (1). Temas: quarteira_vilamoura (10), satelite (9), batimetria_emodnet_ih (2).

- `2026-09-26_zimbral_clarity/README.md`: Zimbral — pedra confirmada e modelo experimental de clareza

### `reports/2026-09-27_as_de_paus`

13 ficheiros, 2.2 MB. Tipos: .py (5), .json (5), .png (2), .md (1). Temas: satelite (7), batimetria_emodnet_ih (5), patrimonio_naufragios (3).

- `2026-09-27_as_de_paus/README.md`: Ás de Paus: o satélite mostra estrutura no fundo?

### `reports/2026-09-27_garrao_candidata2`

83 ficheiros, 39.5 MB. Tipos: .png (37), .py (16), .json (10), .pyc (6), .gpx (5), .csv (3). Temas: garrao_ancao_quinta_lago (83), multifeixe_mbes (37), aia3587_quarteira_garrao (31), quarteira_vilamoura (26), litologia_rocha (25), batimetria_emodnet_ih (17). Levantamentos COSMO: LTHQUGA1.

- `2026-09-27_garrao_candidata2/README.md`: Candidatas #2 e #5 frente ao Garrão, validadas com o EIA da alimentação artificial Quarteira–Garrão
- `2026-09-27_garrao_candidata2/cosmo_afloramentos/atlas_LTHQUGA1_Quarteira-Garrao.pdf`: Quarteira Vale do Lobo Garrão · COSMO LTHQUGA1 2025 (multifeixe 0,3 m, APA) · = topo de cada pedra (ID e profundidade do topo, m ZH); pedras

- 🗺 `2026-09-27_garrao_candidata2/coordenadas_aia3587.gpx`: bbox -8.23062,36.95986,-7.98183,37.07678; EPSG:4326; 82 entidades; campos: ele, time, magvar, geoidheight, name, cmt, desc, src
- 🗺 `2026-09-27_garrao_candidata2/cosmo/cosmo_topos.gpx`: bbox -8.08725,37.04233,-8.07042,37.04588; EPSG:4326; 3 entidades; campos: ele, time, magvar, geoidheight, name, cmt, desc, src
- 🗺 `2026-09-27_garrao_candidata2/cosmo_afloramentos/afloramentos.geojson`: bbox -8.11075,37.02967,-8.04947,37.06625; EPSG:4326; 53 entidades; campos: id, tipo, topo_lat, topo_lon, plotter, prof_topo_m, prof_f
- 🗺 `2026-09-27_garrao_candidata2/cosmo_afloramentos/afloramentos.gpx`: bbox -8.09020,37.02967,-8.05944,37.05610; EPSG:4326; 41 entidades; campos: ele, time, magvar, geoidheight, name, cmt, desc, src
- 🗺 `2026-09-27_garrao_candidata2/eia3587_fig20_layers.geojson`: bbox -8.17333,36.97224,-8.00671,37.08000; EPSG:4326; 325 entidades; campos: layer, area_ha, source
- 🗺 `2026-09-27_garrao_candidata2/eia3587_layers.geojson`: bbox -8.23062,36.95986,-7.98183,37.07678; EPSG:4326; 79 entidades; campos: layer, n, name, type, lat, lon, radius_m, hough
- 🗺 `2026-09-27_garrao_candidata2/naufragios_ih.gpx`: bbox -8.52693,36.96839,-7.93991,37.13478; EPSG:4326; 6 entidades; campos: ele, time, magvar, geoidheight, name, cmt, desc, src
- 🗺 `2026-09-27_garrao_candidata2/pedra_nova_quarteira.gpx`: bbox -8.08415,37.03072,-8.07950,37.03154; EPSG:4326; 3 entidades; campos: ele, time, magvar, geoidheight, name, cmt, desc, src

### `reports/2026-09-27_ilha_faro_13m`

108 ficheiros, 31.0 MB. Tipos: .json (29), .py (20), .log (18), .png (15), .pyc (8), .csv (6). Temas: garrao_ancao_quinta_lago (21), batimetria_emodnet_ih (21), satelite (21), praia_faro_ilha_faro (15), valado_albufeira (12), patrimonio_naufragios (10). Levantamentos COSMO: LTHCUAR2.

- `2026-09-27_ilha_faro_13m/README.md`: Fundo a partir dos 13 m nos dias mais limpos (B03): Ilha de Faro, Ancão e Santa Eulália
- `2026-09-27_ilha_faro_13m/relatorio_valeformosa_evolucao_v2.md`: Relatório Técnico: Evolução do Pipeline de Deteção & Análise do Naufrágio "Vale Formosa" (UKHO 60948)

- 🗺 `2026-09-27_ilha_faro_13m/brasinho_2026-09-24.tif`: bbox -7.79793,36.99098,-7.78655,37.00011; EPSG:32629; 100x100 px, 6 banda(s), res 10.00
- 🗺 `2026-09-27_ilha_faro_13m/brasinho_2026-09-25.tif`: bbox -7.79793,36.99098,-7.78655,37.00011; EPSG:32629; 100x100 px, 6 banda(s), res 10.00
- 🗺 `2026-09-27_ilha_faro_13m/euseamap_faro_fuzeta.geojson`: bbox -7.99000,36.88000,-7.64000,37.09580; EPSG:4326; 47 entidades; campos: substrate
- 🗺 `2026-09-27_ilha_faro_13m/euseamap_valado.geojson`: bbox -8.47000,36.99000,-8.24000,37.10000; EPSG:4326; 26 entidades; campos: substrate

### `reports/2026-09-27_qft204_quarteira`

5 ficheiros, 1.3 MB. Tipos: .json (2), .png (2), .md (1). Temas: quarteira_vilamoura (5), garrao_ancao_quinta_lago (3), praia_faro_ilha_faro (2), aia3587_quarteira_garrao (1), sonar_lateral_sss (1), litologia_rocha (1).

- `2026-09-27_qft204_quarteira/README.md`: Recifes entre a QF-T2-04 e Quarteira (Garrão – Vale do Lobo – Loulé Velho)

### `reports/2026-09-27_zimbral_cabo`

9 ficheiros, 1.5 MB. Tipos: .json (3), .py (3), .png (2), .md (1). Temas: praia_faro_ilha_faro (4), satelite (2).

- `2026-09-27_zimbral_cabo/README.md`: Recifes entre o Zimbral e o Cabo de Santa Maria (Ilha da Barreta / Deserta)

### `reports/2026-09-28_albufeira_cosmo`

11 ficheiros, 6.1 MB. Tipos: .py (2), .png (2), .md (1), .csv (1), .geojson (1), .gpx (1). Temas: valado_albufeira (11), litologia_rocha (8), multifeixe_mbes (5), ccmar_ipma (4), quarteira_vilamoura (2), satelite (2). Levantamentos COSMO: LTHNOCR1, LTHPEAL1.

- `2026-09-28_albufeira_cosmo/README.md`: Afloramentos rochosos no multifeixe COSMO de Albufeira (LTHPEAL1)
- `2026-09-28_albufeira_cosmo/atlas_LTHPEAL1_Albufeira.pdf`: Albufeira (Peneco Alemães) · COSMO LTHPEAL1 2020 (multifeixe 0,3 m, APA) · = topo de cada pedra (ID e profundidade do topo, m ZH); pedras = 

- 🗺 `2026-09-28_albufeira_cosmo/afloramentos.geojson`: bbox -8.25809,37.07444,-8.23288,37.08387; EPSG:4326; 61 entidades; campos: id, tipo, topo_lat, topo_lon, plotter, prof_topo_m, prof_f
- 🗺 `2026-09-28_albufeira_cosmo/afloramentos.gpx`: bbox -8.25774,37.07444,-8.23334,37.08387; EPSG:4326; 55 entidades; campos: ele, time, magvar, geoidheight, name, cmt, desc, src

### `reports/2026-09-30_pedra_do_valado`

47 ficheiros, 28.8 MB. Tipos: .py (8), .png (7), .md (6), .json (6), .gpx (5), .csv (4). Temas: valado_albufeira (47), ccmar_ipma (34), multifeixe_mbes (21), batimetria_emodnet_ih (19), litologia_rocha (17), satelite (15). Levantamentos COSMO: LTHNOCR1, LTHQUGA1.

- `2026-09-30_pedra_do_valado/README.md`: Parque Natural Marinho do Recife do Algarve – Pedra do Valado: batimetria e anomalias Sentinel-2
- `2026-09-30_pedra_do_valado/cosmo_lthnocr1/atlas_LTHNOCR1_Porches.pdf`: Praia Nova Cova Redonda (Porches) · Pedra do Valado costeira · COSMO LTHNOCR1 2020 (multifeixe 0,3 m, APA) · = topo de cada pedra (ID e prof
- `2026-09-30_pedra_do_valado/cross_ref_AMPIC_RENSUB.md`: Cruzamento AMPIC × RENSUB (PNMRA-PV)
- `2026-09-30_pedra_do_valado/cross_ref_LTHQUGA1.md`: Cruzamento das anomalias Sentinel-2 com o atlas COSMO LTHQUGA1
- `2026-09-30_pedra_do_valado/cross_ref_RENSUB_PNMRA-PV.md`: Cruzamento das 54 anomalias Sentinel-2 com o mapa RENSUB (PNMRA-PV)
- `2026-09-30_pedra_do_valado/cross_ref_anomalias_habitat_point.md`: Cruzamento das 54 anomalias S2 com EMODnet habitat_point (CCMAR/RenSub)
- `2026-09-30_pedra_do_valado/cross_ref_sediment_anomalies_investigation.md`: Investigação das 6 anomalias S2 em sedimento
- `2026-09-30_pedra_do_valado/valado_ccmar_multifeixe_alta.pdf`: Pedra do Valado · multifeixe CCMAR (Bentes & Gonçalves 2024, Fig. 1.1; 0,5 m no original, 11 × 14 m/píxel na figura publicada; posicionado p
- `2026-09-30_pedra_do_valado/valado_completo_alta.pdf`: Parque Natural Marinho do Recife do Algarve Pedra do Valado · rocha RENSUB + limites oficiais + verdade de terreno CCMAR/AMPIC + mergulhos A

- 🗺 `2026-09-30_pedra_do_valado/CCMAR_toponimia_Albufeira-Faro_latlon.gpx`: bbox -8.31184,36.83673,-7.95505,37.09436; EPSG:4326; 93 entidades; campos: ele, time, magvar, geoidheight, name, cmt, desc, src
- 🗺 `2026-09-30_pedra_do_valado/cosmo_lthnocr1/afloramentos.geojson`: bbox -8.39061,37.08708,-8.37335,37.09803; EPSG:4326; 199 entidades; campos: id, tipo, topo_lat, topo_lon, plotter, prof_topo_m, prof_
- 🗺 `2026-09-30_pedra_do_valado/cosmo_lthnocr1/afloramentos.gpx`: bbox -8.39053,37.08708,-8.37418,37.09760; EPSG:4326; 178 entidades; campos: ele, time, magvar, geoidheight, name, cmt, desc, src
- 🗺 `2026-09-30_pedra_do_valado/valado_anomalias.gpx`: bbox -8.44364,37.03621,-8.26564,37.08326; EPSG:4326; 54 entidades; campos: ele, time, magvar, geoidheight, name, cmt, desc, src
- 🗺 `2026-09-30_pedra_do_valado/valado_completo.gpx`: bbox -8.64051,36.97660,-8.25020,37.10508; EPSG:4326; 273 entidades; campos: ele, time, magvar, geoidheight, name, cmt, desc, src
- 🗺 `2026-09-30_pedra_do_valado/valado_cumes_ccmar.gpx`: bbox -8.41376,37.03516,-8.31582,37.07029; EPSG:4326; 36 entidades; campos: ele, time, magvar, geoidheight, name, cmt, desc, src

### `reports/2026-10-01_agentes_pesquisa`

1 ficheiros, 0.0 MB. Tipos: .md (1). Temas: aia3587_quarteira_garrao (1), quarteira_vilamoura (1), garrao_ancao_quinta_lago (1), valado_albufeira (1), recifes_artificiais (1), multifeixe_mbes (1).

- `2026-10-01_agentes_pesquisa/agente_p.md`: Relatório dos agentes de pesquisa (27/09 a 01/10/2026)

### `reports/2026-10-16_physics_coefficients_backtest`

7 ficheiros, 0.4 MB. Tipos: .txt (4), .json (2), .md (1). Temas: quarteira_vilamoura (3), batimetria_emodnet_ih (2), satelite (2).

- `2026-10-16_physics_coefficients_backtest/README.md`: Backtest dos coeficientes da física: por sítio vs universais

### `reports/2026-10-18_reef_pipeline_pr5_validation`

2 ficheiros, 0.0 MB. Tipos: .md (1), .json (1). Temas: ccmar_ipma (1), valado_albufeira (1).

- `2026-10-18_reef_pipeline_pr5_validation/README.md`: Validação do PR #5 do `reef-imagery-pipeline` contra observações de campo (2026-10-18)

### `reports/2026-10-20_cosmo_lthfa1_praia_faro`

13 ficheiros, 1.4 MB. Tipos: .png (5), .json (3), .py (2), .md (1), .csv (1), .gpx (1). Temas: praia_faro_ilha_faro (13), greta (5), litologia_rocha (5), multifeixe_mbes (3), batimetria_emodnet_ih (3), garrao_ancao_quinta_lago (3).

- `2026-10-20_cosmo_lthfa1_praia_faro/README.md`: COSMO LTHFA1 (Praia de Faro): multifeixe 2018, 2020 e 2025, Pedra da Greta e marcas GRT do barco

- 🗺 `2026-10-20_cosmo_lthfa1_praia_faro/afloramentos.gpx`: bbox -8.00587,36.98883,-7.98726,37.00368; EPSG:4326; 2 entidades; campos: ele, time, magvar, geoidheight, name, cmt, desc, src

### `reports/2026-10-20_gps_zimbral_pedra_do_alto`

3 ficheiros, 0.0 MB. Tipos: .md (1), .py (1), .json (1). Temas: batimetria_emodnet_ih (3), quarteira_vilamoura (1), praia_faro_ilha_faro (1), valado_albufeira (1), multifeixe_mbes (1), litologia_rocha (1). Levantamentos COSMO: LTHBAIF1, LTHPEAL1.

- `2026-10-20_gps_zimbral_pedra_do_alto/README.md`: Zimbral e Pedra do Alto: marcas GPS do dono, profundidade na marca e verificação com os mergulhos

### `reports/2026-10-20_persistentes_zimbral_quarteira`

7 ficheiros, 0.4 MB. Tipos: .py (3), .md (1), .txt (1), .png (1), .json (1). Temas: quarteira_vilamoura (7), garrao_ancao_quinta_lago (5), multifeixe_mbes (5), litologia_rocha (5), aia3587_quarteira_garrao (1), sonar_lateral_sss (1). Levantamentos COSMO: LTHQUGA1.

- `2026-10-20_persistentes_zimbral_quarteira/README.md`: Anomalias persistentes do fundo: Zimbral–Cabo e QF-T2-04–Quarteira. Coordenadas GPS, marcas do barco e COSMO

### `reports/2026-10-21_greta_recife`

12 ficheiros, 7.0 MB. Tipos: .png (3), .csv (2), .gpx (2), .pdf (2), .md (1), .json (1). Temas: greta (12), batimetria_emodnet_ih (7), multifeixe_mbes (5), litologia_rocha (5), satelite (3), praia_faro_ilha_faro (2).

- `2026-10-21_greta_recife/README.md`: Pedra da Greta: deteção de toda a crista por satélite e coordenadas GPS
- `2026-10-21_greta_recife/complexo_greta_lirios_lages_relatorio_nautico.pdf`: Complexo de Recifes: Pedra da Greta, Lírios e Lages Relatório Náutico Integrado · 22/10/2026 · Sentinel-2 Multi-Temporal & EMODnet · Marcas 
- `2026-10-21_greta_recife/greta_recife.pdf`: Pedra da Greta deteção por satélite e coordenadas GPS UnderwaterVisibility · 21/10/2026 · Sentinel-2 L2A, órbita R037, tile 29SNA · marcas d

- 🗺 `2026-10-21_greta_recife/greta_coordenadas.gpx`: bbox -8.00868,36.97273,-7.97825,37.00348; EPSG:4326; 14 entidades; campos: ele, time, magvar, geoidheight, name, cmt, desc, src

### `reports/2026-10-21_melhor_dia_satelite`

11 ficheiros, 6.7 MB. Tipos: .png (7), .py (3), .md (1). Temas: satelite (4), greta (3), valado_albufeira (2), batimetria_emodnet_ih (1).

- `2026-10-21_melhor_dia_satelite/README.md`: Melhor imagem de satélite: Santa Eulália, Zimbral e Pedra da Greta

### `reports/2026-10-21_santa_eulalia_pedra_do_alto`

7 ficheiros, 3.5 MB. Tipos: .py (2), .png (2), .md (1), .csv (1), .gpx (1). Temas: satelite (4), ccmar_ipma (4), litologia_rocha (3), batimetria_emodnet_ih (3), greta (1), valado_albufeira (1).

- `2026-10-21_santa_eulalia_pedra_do_alto/README.md`: Estudo de Deteção e Diagnóstico de Fundo Marinho: Pedra de Santa Eulália e Pedra do Alto

- 🗺 `2026-10-21_santa_eulalia_pedra_do_alto/eulalia_alto_pontos.gpx`: bbox -8.22453,37.05672,-8.20206,37.07420; EPSG:4326; 9 entidades; campos: ele, time, magvar, geoidheight, name, cmt, desc, src

### `reports/2026-10-24_pedra_do_valado_external_data`

3 ficheiros, 7.5 MB. Tipos: .md (2), .png (1). Temas: valado_albufeira (3), psoem_dgrm (2), batimetria_emodnet_ih (2), ccmar_ipma (1), multifeixe_mbes (1), sonar_lateral_sss (1).

- `2026-10-24_pedra_do_valado_external_data/relatorio_deep_research_valado.md`: Relatório de Investigação Profunda e Catálogo Mestre: Pedra do Valado
- `2026-10-24_pedra_do_valado_external_data/relatorio_investigacao_pedra_do_valado.md`: Relatório de Investigação Profunda e Integração de Dados Públicos: Pedra do Valado / Armação de Pêra
