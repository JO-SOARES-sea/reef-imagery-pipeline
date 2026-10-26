# Investigação: Pedra da Greta, B-24, recifes artificiais de Faro e troço Quarteira–Quinta do Lago

Investigação de 26/10/2026. Verifica as afirmações do Rui e os dados que estão no repositório, usando primeiro as fontes locais e só depois a web:
- `/Volumes/4G_WD_BLACK/BACKUP` (indexado em [LOCAL_SOURCES_INDEX.md](LOCAL_SOURCES_INDEX.md));
- o projeto `~/dev/UnderwaterVisibility`.

## Resumo

| Afirmação | Veredicto | Evidência principal |
|---|---|---|
| Na Pedra da Greta há multifeixe por causa do bombardeiro | **Falso como dito.** Há sonar lateral, não multifeixe | Ver §1 |
| Entre Quarteira e Quinta do Lago há multifeixe | **Verdadeiro**, até ~13 m | COSMO LTHQUGA1 (APA), 0,30 m, 2018/2020/2025. Para lá disso há sonar lateral do IH e o mapa de rocha de Teixeira |
| Nessa zona vêem-se recifes naturais por satélite | **Verdadeiro**, com a composição de dias límpidos (não com uma cena única) | Ver §3 |
| Há recifes artificiais perto da Pedra da Greta | **Verdadeiro** | Ver §2 |
| A Greta foi perfeitamente identificada por satélite | **Verdadeiro para a crista; falta sonda para saber se é rocha.** Corrigido a 26/10, ver §4 | Crista persistente de 2,1 km, reproduzida de forma independente |

## 1. B-24 e "multifeixe da Greta"

**Inventário da AIA 3587** (Anexo C – Arqueologia, Elemento 18, "Mar de Faro – B24 Liberator"; `BACKUP/doc-q/EIA_3587_aditamento/Anexo_C_Arqueologia.pdf`, p. 65):
- Posição: **36°59'25,80"N 8°00'27,60"W**.
- Estado: "estrutura do avião praticamente completa com as asas de 34 m, duas das hélices, um leme vertical e os motores".
- A ficha diz "1942". As outras fontes dizem 30/11/1943, por isso a ficha deve estar errada.
- A ficha **não indica profundidade**.

**Expedição de 2024** (Ships of Discovery/DPAA, com ECU, RECON Offshore, HistArc/ULisboa, CCMAR e CNANS; Barlavento, 16/09/2024):
- Posição: 36°59,424'N 8°00,456'W, a 10 m da posição da AIA.
- Profundidade: **18–21 m**, sobre **fundo de areia plano**, a ~2,1 km da Praia de Faro.
- Métodos: mergulho e escavação em 2024; **sonar lateral e magnetómetro** na campanha de 2022. Não há referência a multifeixe.

**Que levantamentos cobrem a zona:**
- **Sonar lateral do IH** (AIA 3587, Aditamento, Anexo D, Desenho 2; campanhas 2000–2017):
  - cobre o B-24 e as marcas GRT;
  - acaba a ~8,0 W, por isso **não cobre a crista do satélite**;
  - é um desenho digitalizado de ~4 m/píxel, só com intensidade e **sem alturas**.
- **Multifeixe COSMO LTHFA1 (Praia de Faro)** acaba a ~11,8 m ZH. A Greta está a 13,7–16 m, por isso **fica fora** (ver `UnderwaterVisibility/reports/2026-10-20_cosmo_lthfa1_praia_faro/README.md`).

**Dados sem suporte em `data/vector/greta_sonar_ih_b24_completo.csv`:**
- **Valores que nenhuma fonte suporta:**
  - as áreas dos "Afl-1" (4 356 m²) e "Afl-7" (2 824 m²);
  - os "relevos verticais" de 0,5–0,9 m em todos os pontos;
  - os 17,1 m do B-24, que vêm do EMODnet; a expedição mediu 18–21 m.
- **Porque não podem vir das fontes:** estão escritos à mão no CSV. Nenhum código os calcula. Nenhum dado disponível mede alturas nesses pontos: o sonar do IH é só intensidade e o COSMO não chega lá.
- **Origem provável:** as figuras "sonar IH B-24" foram geradas por uma sessão do Google Antigravity (Gemini) em 20–21/10 e **nunca foram revistas** (`UnderwaterVisibility-archive/2026-10-21_lthfa1_agente_extras_nao_revistos/`).
- **Onde estão os "Afl-1" e "Afl-7":** a 48–58 m da zona de recifes artificiais. O Afl-7 e o B-24 ficam mesmo **dentro** da zona tracejada de recifes artificiais da Fig. 20. Podem ser grupos de cubos de betão e não rocha natural.
- **Única exceção com dados:** a VILHA (relevo de 0,60 m, a 11 m de um afloramento) **está** medida no COSMO LTHFA1.
- `scripts/generate_b24_faro_probability_surface.py` também afirma usar "Multifeixe do IH (EIA AIA 3587)", **que não existe**.

## 2. Recifes artificiais de Faro/Ancão

**Polígono oficial** (PSOEM/DGRM, camada 13 "Áreas existentes de complexos recifais", fonte IPMA; descarregado para `data/external/psoem/lyr13_complexos_recifais_faro_quarteira.geojson`):

| Complexo | Área | Extensão | Centróide |
|---|---|---|---|
| **Faro/Ancão** | **15,8 km²** | 36,953–37,008 N, 8,042–7,971 W | 36,9777 N, 8,0098 W |
| Vilamoura/Quarteira | 6,4 km² | 37,018–37,042 N, 8,167–8,117 W | 37,0303 N, 8,1429 W |
| Oura/Albufeira | 5,7 km² | 37,031–37,050 N, 8,211–8,169 W | 37,0409 N, 8,1904 W |
| Tavira (?) | 0,8 km² | 37,005–37,013 N, 7,749–7,738 W | 37,0093 N, 7,7438 W |

**Literatura** (IPMA/IPIMAR; Moura et al. 2011, Braz. J. Oceanogr. 59:91–94):
- 52 grupos de cubos de betão de 1,4 m.
- 12,2 km² e 8,1 km de comprimento, paralelos à costa NW–SE ao longo da isóbata dos 20 m (com grupos também a 16 m).
- Ponto de amostragem publicado: 36°59,25'N 8°00,43'W.
- Ampliação em 2002–2003.
- **As coordenadas de cada grupo não são públicas.** Há que pedi-las ao IPMA/DGRM.

**Distância dos pontos da Greta ao polígono oficial:**
- marcas GRT: **560–850 m fora**;
- crista do satélite: **~310 m fora**, em todos os pontos e paralela ao limite NE do polígono;
- B-24: 159 m fora;
- "Afl-7": 58 m fora.

![Mapa](greta_b24_psoem_map.png)

## 3. Quarteira–Garrão–Quinta do Lago: o raciocínio da Valado aplicado

**Verdade de referência por área.** Em vez de pontos, usa-se o mapa do fundo de Teixeira (Projeto de Execução 2025, Fig. 20; base: sonar do IH 2000–2017, **383 mergulhos e 313 sondagens manuais**, confirmados no Anexo C, p. 129). Fica em `data/external/eia3587/fig20_teixeira2025_layers.geojson`.
- **Positivos:** rocha, a mais de 20 m da borda dos polígonos.
- **Negativos:** areia e lodos mapeados, à mesma profundidade (±1 m).
- **IC:** bootstrap por blocos de 500 m.
- **Script:** `scripts/evaluate_against_lithology.py`.

| Imagem | Área | AUC anomalia de escurecimento (IC 95%) | AUC só profundidade | Rocha vs lodos |
|---|---|---|---|---|
| Cena única (2025) | Quarteira | **0,42** (0,33–0,51) | 0,50 | 0,39 |
| **Composição 10 dias límpidos** | Quarteira (45 blocos de rocha) | **0,665 (0,60–0,73)** | 0,50 | 0,63 |
| **Composição 10 dias límpidos** | Quarteira–Garrão–Quinta do Lago (95 blocos) | **0,61 (0,54–0,68)** | 0,50 | **0,71** |

**Leitura:**
- A rocha natural é detetável por satélite neste troço, mas **só com a mediana de dias límpidos**. Com uma cena única a AUC fica abaixo do acaso.
- A AUC por píxel é moderada porque o mapa é uma síntese generalizada (~13 m/píxel).
- O estudo anterior do UnderwaterVisibility (`2026-09-27_garrao_candidata2`) mostrou o mesmo com outro método: as anomalias replicadas a ≥ 9 m caem **19/19 a ≤ 50 m de rocha mapeada**, contra 18 % ao acaso.
- O multifeixe COSMO LTHQUGA1 mede relevo de 0,88 m nessas anomalias, contra 0,06 m ao acaso.

**Ground truth inválido no repositório:**
- `data/vector/quarteira_coordenadas.csv`: 5 pontos de coordenadas redondas, **nenhum sobre rocha mapeada** (217–1012 m de distância).
- `data/vector/algarve_dive_sites.csv`: o "Prainha" está em terra (cota +41,5 m EMODnet), e os nomes e coordenadas não têm fonte.
- Os dois ficheiros devem ser substituídos pela litologia de Teixeira.

## 4. "A Greta foi perfeitamente identificada por satélite" (revisto)

A primeira versão desta secção dizia "não confirmado". **Estava errada**, por três motivos explicados abaixo.

**A deteção do UnderwaterVisibility reproduz-se.**
- **Método:** `reports/2026-10-21_greta_recife`, DL-024/DL-043. Passa-alto DoG(1–10 px) da B03 em **cada** passagem, com correção de glint pela B11 e normalização (z) pelo ruído de água funda; depois a mediana de 22 passagens R037 (2017–2026).
- **Perfil:** mínimo a ±50 m da linha, de 30 em 30 m. Controlos: linhas paralelas a 450–600 m, a 13–17 m de profundidade.
- **Repetido aqui sobre os mapas do próprio UV** (`data/raw/seabed/work/if13/if_maps.npz`):

| Mapa | Crista (2,1 km): mediana z | Crista mais escura que o p05 dos controlos | Prolongamento NW (para PEDRA0/GRT) |
|---|---|---|---|
| Mediana das 22 passagens | **−1,35** (controlos: p05 −0,86, mediana −0,40) | **86 %** | 42 % |
| Metade A (anos ímpares) | −1,23 (p05 −1,25) | 46 % | 18 % |
| Metade B (anos pares) | −1,57 (p05 −1,11) | 76 % | 44 % |
| Composição de 10 dias (este repositório), mesmo DoG | −0,72 (p05 −1,62) | **6 %** | 14 % |
| Cena única 2025, mesmo DoG | −0,92 (p05 −2,66) | 0 % | 0 % |

**O que isto significa:**
- Há uma **crista escura persistente de 2,1 km, paralela à costa, a 15–16 m**. Aparece nas duas metades do arquivo e é mais escura do que 95 % dos controlos à mesma profundidade.
- **A ponta NW fica a ~830 m da PEDRA0 e da GRT2.** O valor de "1,9–2,8 km" da DL-043 refere-se ao centro do DL-024, não à ponta.
- No prolongamento para NW, em direção às marcas GRT, cerca de 40 % dos pontos ainda passam o limiar.
- **Fica em aberto:**
  - se é rocha ou uma forma de areia persistente. A persistência de 2017 a 2026 favorece rocha, mas só a sonda confirma;
  - não há segunda órbita (a R137 não cobre a zona).
- **Recife artificial:** a crista fica **fora** do polígono oficial PSOEM/IPMA, ~300 m para terra. Como os módulos são colocados dentro da área licenciada, isto **pesa contra** a hipótese de serem cubos.
- **Atenção ao Lírios:** a marca de sonda **Lírios (17,6 m) está DENTRO** do polígono do recife artificial Faro/Ancão. A "continuidade Greta → Lírios" desenhada em `complexo_greta_lirios_lages_mapa.png` pode ligar a crista natural a módulos artificiais. O Lages (13,6 m) fica 1,2 km fora.

**Porque é que a primeira versão errou:**
1. **Instrumento errado para uma estrutura estreita.**
   - A "anomalia de escurecimento" deste repositório (`sdb_m − EMODnet`) mede o escurecimento absoluto face a um EMODnet de ~115 m.
   - Comparava-o com toda a água à mesma profundidade numa cena de 6 × 6 km.
   - Os gradientes de grande escala (coluna de água, erros do EMODnet) dominam, e uma crista de ~40 m de largura perde-se.
   - Para estruturas lineares estreitas a ~15 m, o passa-alto com controlos paralelos locais é o método certo.
2. **A composição apaga a crista.**
   - A composição deste repositório é a mediana da *reflectância* de 10 dias de inverno escolhidos pelo NIR baixo. O NIR baixo mede pouco brilho à superfície, não um fundo visível.
   - Ao juntar a reflectância de dias com colunas de água diferentes *antes* do passa-alto, o contraste fino perde-se (6 % contra 86 %).
   - O UV faz o passa-alto em cada passagem e depois a mediana dos z.
3. **Distância desatualizada.** Usei os "1,9–2,8 km" da DL-043 em vez da ponta NW medida no relatório de 21/10 (~830 m).
4. **Leitura do recife artificial:** tratei "paralela ao limite, a 300 m" como suspeito, quando estar fora do polígono licenciado é um argumento contra a hipótese artificial.

**Para fechar:** uma passagem com sonda ao longo da linha ponta SE → ponta NW → PEDRA0 → GRT2 → GRT0, a registar profundidade e hora (página 5 de `greta_recife.pdf`).

## 5. Verificação do `BACKUP/doc-q/INDICE_FONTES_QUARTEIRA.md` (compilado por MiniMax-M3)

| Afirmação | Verificação |
|---|---|
| 56 elementos patrimoniais | ✅ DIA AIA 3587 ("total de 56 ocorrências patrimoniais") |
| 383 mergulhos, 313 sondagens, IH 2000–2017 | ✅ Anexo C, p. 129 |
| Desenho 2 sem escala explícita | ✅ O Aditamento não a indica; a georreferenciação dá ~1:25 000 |
| Hidromar 71 (2002), sonar lateral + chirp, NRP Andrómeda, Quarteira–Ancão | ✅ "sistema combinado de sonar lateral e chirp", "Praia da Falésia (Vilamoura) e a barra do Ancão" |
| AIA 3660: 10 alvos | ✅ DIA AIA 3660 ("identificou ainda 10 alvos") |
| Multifeixe 3000 ha a 25–40 m (item 5.5) | ⚠️ Só numa imagem PNG; o PDF do projeto não tem texto. Mesmo que se confirme, não cobre a Greta (14–17 m) |
| COSMO "a registar e descarregar" | ⚠️ Desatualizado: o COSMO LTHQUGA1/LTHFA1 já foi descarregado (BACKUP/bulk_download_*) |

## Fontes

- Locais: `BACKUP/doc-q/` (AIA 3587, AIA 3660, Hidromar 71, PSOEM), `UnderwaterVisibility/reports/2026-09-27_garrao_candidata2`, `…/2026-09-27_ilha_faro_13m`, `…/2026-10-20_cosmo_lthfa1_praia_faro`.
- [DIA AIA 3587 (APA/SIAIA)](https://siaia.apambiente.pt/AIADOC/AIA3587/aia3587_dia(anexotua)2023823102114.pdf)
- [Barlavento, 16/09/2024 — expedição ao B-24](https://www.barlavento.pt/faro-militares-americanos-procuram-vestigios-dos-aviadores-desaparecidos-em-1943/)
- [Moura et al. 2011, Estimation of secondary production of the Faro/Ancão artificial reefs](https://revistas.usp.br/bjoce/article/view/39612)
- [PSOEM/DGRM, serviço de mapas](https://webgis.dgrm.mm.gov.pt/arcgis/rest/services/PSOEM_GEOPORTAL/PSOEM_geoportal_servicos_semPatrimonio/MapServer) (camadas 13 e 14)
- [Subnauta — Pedra da Greta](http://www.subnauta.pt/mergulhar-sotavento-algarve/pedra-greta.html), [Deep Blue](http://www.deepblue.pt/saidas-de-mergulho/faro/pedra-da-greta), [Portugal Dive](https://www.portugaldive.com/best-dives/spot/pedra-da-greta-182.html)
- [Pássaro de Ferro — O B-24 de Faro](http://www.passarodeferro.com/2013/11/o-b-24-de-faro-arqueologia-de-aviacao.html)

## 6. Sonar lateral do IH: o que dá para extrair (26/10)

O sonar lateral **deteta** rocha; o que não dá diretamente são cotas.
- **Métodos de reconstrução 3D:** sombras H = L·A/(R+L); shape-from-shading; Neural SfS (Bore & Folkesson, IEEE JOE 2023); normais por CNN (Xie et al. 2022). Todos precisam da geometria de cada ping (altitude, alcance, navegação), que o mosaico impresso do Desenho 2 (JPEG, 2,1 m/píxel) perdeu.
- **Ferramentas para dados brutos (XTF/JSF):**
  - SidescanTools (GEOMAR/sonoware; IHR, 2026);
  - SideScanSonarEditor (JOSS, 2025);
  - MB-System;
  - Open Sidescan (CIDCO);
  - RockMapper/PINGMapper (substrato a partir de mosaicos).

`scripts/sidescan_to_relief.py` calibra o mosaico com o multifeixe COSMO LTHQUGA1 na faixa comum. Recupera a geometria das passagens: rotação de ~30°, nadires a cada ~130 m. Validação por blocos espaciais de 250 m:
- **Rocha (relevo ≥ 0,3 m no COSMO): AUC 0,92**, contra 0,59 só com a intensidade. O mosaico mapeia rocha.
- **Altura: não recuperável.**
  - Erro igual ao de prever a mediana: 0,0174 m contra 0,0174 m.
  - Spearman de 0,01 a 2 m e de 0,11 em blocos de 10 m.
  - O script só grava a probabilidade de rocha (`outputs/sidescan/greta_b24/rock_probability.tif`, EPSG:3763).
- **Cautela:** calibrado até ~13 m, onde só 0,6 % é rocha. Na Greta (15–17 m) a probabilidade sai alta em grande parte da janela, por isso serve para localizar, não para medir área.
- **Para ter alturas tipo COSMO** são precisos os dados brutos do IH (pedido IH.CD.08, já redigido em `BACKUP/Parque_Pedra_do_Valado/pesquisa_dados/_pedidos_formais/`) ou um levantamento multifeixe.

## 7. Base de estruturas persistentes e classificador rocha vs areia (26/10)

`scripts/persistent_structures_ml.py` gera `data/external/persistent_structures/estruturas_persistentes.geojson` e `.csv`.
- **Âmbito:** 3 238 estruturas persistentes (Sentinel-2 R037, 2017–2026, escuras na mediana e nas duas metades do arquivo) em 6 zonas do Valado à Fuzeta.
- **Classificação:** 685 prováveis rochas, 2 132 prováveis areia/lodo, 421 incertas.

**Rótulos:**
- rocha: a Greta e o Zimbral (sonda do Rui), Santa Eulália (terreno), os cumes do CCMAR e a rocha de Teixeira;
- areia/lodo persistente: o mapa de Teixeira.

**Validação:**
- No troço de Teixeira, com blocos de 1 km:
  - **AUC 0,85 (0,77–0,92)**;
  - **0,82 à mesma profundidade (±1,5 m)**;
  - só o escurecimento dá 0,56;
  - só a profundidade dá 0,82. Por isso a profundidade foi **excluída** das feições: no mapa de Teixeira os lodos estão junto à costa.
- Transferência (treino só com Teixeira; percentil das rochas confirmadas entre as outras estruturas da mesma zona, à mesma profundidade):

| Rocha confirmada | Percentil |
|---|---|
| Zimbral | 99 |
| Santa Eulália | 100 |
| Valado | 86 |
| Greta | 67 |

**Limites:**
- Persistente não é sinónimo de rocha. O classificador ordena e prioriza; não confirma.
- Os únicos negativos verificados são os de Teixeira, numa só zona. Mais marcas de areia confirmadas noutras zonas melhorariam o modelo.
