# Protocolo E024 — verificação com a sonda (verdade prospetiva)

Estado: **aprovado pelo Rui em 2026-10-26** ("yes to all"). Entra no ledger como E024 (declaração), com o SHA-256 deste
ficheiro, e a partir daí não se altera. Os resultados entram como E025 em diante.

Base: `docs/HOLDOUT_PREREGISTO_v2.md` §2 (as verificações de sonda feitas depois de 2026-10-26 são verdade independente
nunca vista) e a lista `data/external/persistent_structures/verificacao_sonda.{csv,gpx}` (30 marcas).

## 1. Porque é preciso um protocolo antes de ir ao mar

A lista atual tem só **candidatas**: estruturas escolhidas pelo valor de informação (incerteza × tamanho), quase todas
"claras" (27 de 30). Só com candidatas mede-se a precisão, mas não a taxa de falsos negativos, nem a AUC. Sem
controlos, "70 % eram pedra" não distingue um bom detetor de uma costa onde 70 % do fundo é pedra. Por isso:

- juntam-se **controlos cegos**: fundo onde os detetores não veem nada, à mesma profundidade;
- o observador **não sabe** qual marca é candidata e qual é controlo.

## 2. Lista de marcas (a regenerar depois da aprovação)

1. **Candidatas:** as 30 atuais. Uma é tratada à parte (§6):
   - **VVA004** está dentro da caixa do holdout v2 (LTHNOCR1, Porches). A lista foi gerada antes do registo do v2.
2. **Controlos:** 1 por candidata (30), na mesma zona.
   - Profundidade EMODnet igual ± 0,5 m.
   - A ≥ 300 m de qualquer estrutura da base (incluindo as zonas extensas) e de qualquer outro controlo.
   - Fora das caixas dos holdouts v1 e v2 e dos polígonos de recife artificial (PSOEM + Fig. 20).
     Também o controlo da VVA004 fica fora da caixa v2.
   - H4 congelada e |mediana z| abaixo da mediana do fundo da zona.
   - Sorteio com seed 42, sem olhar para imagens.
   - Uma candidata sem píxel que cumpra tudo fica sem controlo, e isso reporta-se.
3. **GPX cego:** nomes neutros de 8 caracteres (`SND001`…`SND060`), em ordem aleatória (seed 42).
   - Para o barco: `verificacao_sonda_cega.{gpx,csv}`. Só tem nome, posição, posição no formato do plotter, profundidade
     EMODnet e as colunas a preencher.
   - A correspondência candidata/controlo fica em `verificacao_sonda_chave.csv`, que não vai para o barco.

## 3. Registo no mar (por marca)

| Campo | Valores |
|---|---|
| `nome_gpx` | o nome do GPX |
| `hora_utc` | hora da passagem sobre a marca |
| `profundidade_sonda_m` | leitura da sonda |
| `pedra_ou_areia` | `pedra`, `areia`, `lodo`, `misto`, `nao_visto` |
| `notas` | livre (ex.: algas, peixe, relevo estimado) |

**Gravar o registo SL3 de cada saída** (como os de 2024). O registo dá uma segunda verdade, objetiva, que não depende do
observador.

Ficheiro devolvido: `data/external/own_soundings/verificacao_resultados_AAAA-MM-DD.csv`, com os `.sl3` na mesma pasta.

## 4. Correspondência e rótulos (fixados agora)

- **Marca visitada:** o trilho (SL3 ou plotter) passa a ≤ 25 m da marca. Os 25 m somam o GPS (~5 m), o píxel de
  10 m e a largura típica das estruturas. As restantes ficam "não visitadas" e não entram.
- **Rótulo principal (observador):**
  - `pedra` e `misto` = 1;
  - `areia` e `lodo` = 0;
  - `nao_visto` = excluída.
- **Rótulo secundário (SL3, método E001/E016):**
  - usa-se a rugosidade máxima dos pontos do trilho a ≤ 25 m da marca, com ≥ 2 pontos (agregação da E016);
  - ≥ 0,15 m = 1;
  - ≤ 0,04 m = 0;
  - entre os dois, ou menos de 2 pontos = indeterminado.
- **Concordância:** reporta-se o κ de Cohen entre os dois rótulos. Se κ < 0,4, os resultados reportam-se com os dois
  rótulos lado a lado e nenhum decide sozinho.

## 5. Métricas e regra de decisão (fixadas agora)

**Score de cada método na marca:** o máximo numa vizinhança 3×3 píxeis.

**Métodos avaliados:**
- H4 congelada (`models/frozen/h4_2026-10-26.joblib`);
- |mediana z| e mediana z da pilha `_dev`;
- o candidato ao holdout v2, se existir nessa data.

**Métricas:**
- AUC (candidatas + controlos visitados), com IC por bootstrap estratificado por zona (1 000 réplicas, seed 42);
- precisão nas candidatas (fração pedra), com IC de Wilson;
- taxa de pedra nos controlos;
- tudo à parte por `dominio_validado` (sim / parcial / não).

**Mínimo para decidir:** ≥ 20 candidatas e ≥ 20 controlos visitados, com rótulo principal. Abaixo disso, o relatório é
só descritivo.

**Regra principal:**
- a H4 congelada é **confirmada como ferramenta de descoberta** nas zonas visitadas se:
  - AUC ≥ 0,70 com limite inferior > 0,5;
  - E a precisão nas candidatas exceder a taxa de pedra nos controlos (Wilson sem sobreposição);
- caso contrário, regista-se como **não confirmada**;
- a mesma regra aplica-se à |mediana z|, como referência sem treino.

## 6. Higiene dos holdouts

- **VVA004 (dentro do v2):** o resultado fica selado e só entra no teste único do holdout v2. Nunca entra em E024/E025
  nem em treino.
- **VIF002 (dentro da caixa do v1, já gasto):** reporta-se à parte, só como diagnóstico.
- **Uso dos rótulos de sonda:** ficam fora de qualquer treino até E025 estar registado. Depois disso podem entrar em
  treino, mas deixam de servir de teste.
- **Sem afinação nos resultados:** nenhum limiar, raio ou método se ajusta depois de ver os resultados. Mudanças exigem
  uma nova declaração.

## 7. O que falta fazer depois da aprovação

1. Regenerar a lista com os controlos e o GPX cego (`scripts/build_structures_db.py`, opção nova). O GPX com 60 marcas
   é para o Rui levar para o mar.
2. Escrever `scripts/ingest_sounder_checks.py`: correspondência, rótulos, métricas e regra. Testes offline com dados
   sintéticos, antes de haver resultados reais.

## Decisões do Rui (2026-10-26)

- 60 marcas (30 candidatas + 30 controlos): aceite.
- Raio de 25 m e `misto` = pedra: aceites.
- SL3 gravado em todas as saídas: confirmado.

## Emenda E024a (2026-10-27, decisão do Rui, antes de qualquer dado de campo)

Ao gerar a lista, só 17 das 30 candidatas tiveram controlo com 300 m. A costa tem estruturas tão densas que a regra de
"≥ 20 controlos visitados" nunca se cumpriria e o E024 ficaria só descritivo. Diagnóstico, só com a geração e sem
rótulos: com 150 m há ~25 controlos.

**Única alteração:** a distância mínima dos controlos às estruturas da base, às candidatas e entre controlos passa de
300 m para **150 m** (`CONTROL_MIN_DIST_M`). São 15 píxeis, muito acima do raio de visita (25 m) e da janela 3×3.
Tudo o resto do protocolo fica igual.
