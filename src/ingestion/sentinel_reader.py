"""
src/ingestion/sentinel_reader.py — Ingestão e Pré-Processamento de Bandas Sentinel-2 L2A.

Fornece funções para:
1. Carregar bandas espectrais B02 (Azul), B03 (Verde), B04 (Vermelho) e opcionalmente B08 (NIR)
   a partir de ficheiros GeoTIFF usando Rasterio.
2. Converter Digital Numbers (DN) em Reflectância de Fundo de Atmosfera (BOA).
3. Calcular o NDWI (Normalized Difference Water Index) de McFeeters (1996) para isolar
   a coluna de água e mascarar a linha de costa/terra do Algarve.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

logger = logging.getLogger("reef_pipeline.ingestion")

# Fator de escala padrão do Sentinel-2 L2A (ESA / Copernicus) para converter DN em reflectância
S2_DN_TO_REFLECTANCE_SCALE = 0.0001
S2_NODATA_VALUE = 0


@dataclass
class SentinelScene:
    """Representação estruturada de uma cena/recorte Sentinel-2."""

    blue: np.ndarray        # Banda B02 (490 nm) - Penetração profunda na água
    green: np.ndarray       # Banda B03 (560 nm) - Pico de reflectância marinha / batimetria
    red: np.ndarray         # Banda B04 (665 nm) - Atenuação rápida na água / contraste costeiro
    nir: np.ndarray | None  # Banda B08 (842 nm) - Absorção total pela água, alta reflectância em terra
    profile: dict[str, Any] # Metadados do raster (CRS, Transform, Dimensões, etc.)
    crs: CRS
    transform: Affine
    water_mask: np.ndarray | None = None  # Máscara booleana (True = Água / Oceano)
    ndwi: np.ndarray | None = None        # Matriz contínua de valores NDWI

    @property
    def shape(self) -> tuple[int, int]:
        return self.blue.shape


def read_sentinel_bands(
    source: str | Path | dict[str, str | Path],
    band_indices: dict[str, int] | None = None,
    normalize_reflectance: bool = True,
) -> SentinelScene:
    """
    Carrega as bandas B02 (Azul), B03 (Verde), B04 (Vermelho) e opcionalmente B08 (NIR).

    Suporta:
    1. Um ficheiro GeoTIFF multi-banda (ex: RGB ou RGB+NIR composto).
    2. Um dicionário mapeando o nome da banda para o caminho do ficheiro GeoTIFF correspondente:
       {'B02': 'caminho/b02.tif', 'B03': '...', 'B04': '...', 'B08': '...'}

    Args:
        source: Caminho para ficheiro GeoTIFF composto ou dicionário de bandas.
        band_indices: Mapeamento de banda -> índice de canal (1-based para Rasterio).
                      Padrão para ficheiro de 3 bandas: {'B02': 1, 'B03': 2, 'B04': 3}
                      Padrão para ficheiro de 4 bandas: {'B02': 1, 'B03': 2, 'B04': 3, 'B08': 4}
        normalize_reflectance: Se True e os dados forem inteiros (uint16), aplica a escala
                               0.0001 para converter em reflectância de superfície [0.0, 1.0].

    Returns:
        SentinelScene com as bandas em formato NumPy (float32).
    """
    if isinstance(source, (str, Path)) and Path(source).is_file():
        file_path = Path(source)
        logger.info(f"Lendo cena Sentinel-2 multi-banda a partir de: {file_path}")

        with rasterio.open(file_path) as src:
            profile = src.profile.copy()
            crs = src.crs
            transform = src.transform
            num_bands = src.count

            # Configura mapeamento padrão conforme o número de canais disponíveis
            if band_indices is None:
                if num_bands >= 4:
                    band_indices = {"B02": 1, "B03": 2, "B04": 3, "B08": 4}
                elif num_bands == 3:
                    band_indices = {"B02": 1, "B03": 2, "B04": 3}
                else:
                    raise ValueError(
                        f"O ficheiro GeoTIFF tem apenas {num_bands} banda(s). "
                        "São necessárias pelo menos 3 bandas (B02, B03, B04)."
                    )

            def _read_band(idx: int) -> np.ndarray:
                arr = src.read(idx).astype(np.float32)
                nodata = src.nodatavals[idx - 1]
                if nodata is not None:
                    arr[arr == nodata] = np.nan
                if normalize_reflectance and np.nanmax(arr) > 2.0:
                    arr = arr * S2_DN_TO_REFLECTANCE_SCALE
                return arr

            blue = _read_band(band_indices["B02"])
            green = _read_band(band_indices["B03"])
            red = _read_band(band_indices["B04"])
            nir = _read_band(band_indices["B08"]) if "B08" in band_indices and band_indices["B08"] <= num_bands else None

    elif isinstance(source, dict):
        logger.info("Lendo bandas Sentinel-2 a partir de dicionário de ficheiros individuais...")
        required_bands = ["B02", "B03", "B04"]
        for b in required_bands:
            if b not in source:
                raise KeyError(f"Banda obrigatória '{b}' não fornecida no dicionário de fontes.")

        def _read_single_band_file(p: str | Path) -> tuple[np.ndarray, dict, CRS, Affine]:
            with rasterio.open(p) as src:
                arr = src.read(1).astype(np.float32)
                nodata = src.nodata
                if nodata is not None:
                    arr[arr == nodata] = np.nan
                if normalize_reflectance and np.nanmax(arr) > 2.0:
                    arr = arr * S2_DN_TO_REFLECTANCE_SCALE
                return arr, src.profile.copy(), src.crs, src.transform

        blue, profile, crs, transform = _read_single_band_file(source["B02"])
        green, _, _, _ = _read_single_band_file(source["B03"])
        red, _, _, _ = _read_single_band_file(source["B04"])
        nir = None
        if "B08" in source and Path(source["B08"]).exists():
            nir, _, _, _ = _read_single_band_file(source["B08"])
    else:
        raise ValueError(f"Fonte de dados inválida: {source}. Esperado ficheiro GeoTIFF ou dicionário de bandas.")

    return SentinelScene(
        blue=blue,
        green=green,
        red=red,
        nir=nir,
        profile=profile,
        crs=crs,
        transform=transform,
    )


def calculate_ndwi(
    green: np.ndarray,
    nir: np.ndarray | None = None,
    red: np.ndarray | None = None,
    threshold: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcula o Normalized Difference Water Index (NDWI) e gera uma máscara de água.

    Fórmula Primária (McFeeters, 1996 - Green e NIR):
        NDWI = (B03 - B08) / (B03 + B08)
        - Água: NDWI > threshold (habitualmente > 0.0 devido à absorção forte do NIR pela água)
        - Terra / Vegetação / Rochas terrestres: NDWI <= threshold (reflectância NIR muito superior à verde)

    Fallback (caso a banda NIR não esteja disponível):
        Utiliza o contraste espectral Verde vs Vermelho (B03 - B04) / (B03 + B04),
        aproveitando a rápida atenuação do vermelho na água marinha.

    Args:
        green: Matriz NumPy da banda B03 (Verde).
        nir: Matriz NumPy da banda B08 (NIR). Opcional, mas altamente recomendado.
        red: Matriz NumPy da banda B04 (Vermelho). Utilizado como fallback se nir for None.
        threshold: Limiar de corte para classificar como água (padrão 0.0).

    Returns:
        tuple[ndwi, water_mask]:
            ndwi: Matriz contínua float32 com valores de -1.0 a +1.0.
            water_mask: Matriz booleana onde True = Água marinha e False = Terra/Nuvem/NoData.
    """
    if nir is not None:
        numerator = green - nir
        denominator = green + nir
        index_name = "McFeeters NDWI (B03-B08)"
    elif red is not None:
        logger.warning(
            "Banda NIR (B08) ausente; calculando índice de água baseado em Verde-Vermelho (B03-B04)."
        )
        numerator = green - red
        denominator = green + red
        index_name = "RGB Water Index (B03-B04)"
    else:
        raise ValueError("É necessário fornecer a banda NIR (B08) ou a banda Vermelha (B04) para calcular o NDWI.")

    with np.errstate(divide="ignore", invalid="ignore"):
        ndwi = np.where(denominator != 0, numerator / denominator, -1.0)
        ndwi = np.nan_to_num(ndwi, nan=-1.0, posinf=1.0, neginf=-1.0).astype(np.float32)

    water_mask = ndwi > threshold

    total_pixels = water_mask.size
    water_pixels = int(np.sum(water_mask))
    water_pct = (water_pixels / total_pixels) * 100

    logger.info(
        f"{index_name} calculado com sucesso | Limiar: {threshold} | "
        f"Água: {water_pct:.2f}% dos pixels ({water_pixels}/{total_pixels})"
    )

    return ndwi, water_mask


def build_glint_mask(
    green: np.ndarray,
    nir: np.ndarray | None = None,
    window: int = 5,
    var_thresh: float = 0.15,
    nir_thresh: float | None = None,
    dilate_anomaly_px: int = 1,
) -> np.ndarray:
    """
    Mascara esteiras de embarcações, espuma de ondas e brilho solar (sunglint) na superfície marinha com a banda B08 (NIR).

    Fundamento Físico (Hedley et al., 2005 / Lyzenga, 1981):
    A água pura ou costeira limpa absorve quase 100% da radiação no infravermelho próximo (B08, 842 nm).
    Qualquer pixel na água com reflectância B08 significativamente superior ao baseline límpido
    corresponde a anomalia de superfície (esteira de barco, espuma de rebentação, glint solar ou cirro/nuvem fina).

    Args:
        green: Matriz NumPy da banda B03 em reflectância [0, 1].
        nir: Matriz NumPy da banda B08 (NIR) em reflectância [0, 1].
        window: Dimensão da janela focal para cálculo da variância local (padrão 5x5 px = 50 m).
        var_thresh: Limiar do coeficiente de variação para ruído de textura de superfície.
        nir_thresh: Limiar estrito de reflectância NIR. Se None, é calculado adaptativamente
                    a partir do baseline de água límpida da cena (Hedley baseline + margem).
        dilate_anomaly_px: Raio de dilatação morfológica para limpar franjas de espuma e esteiras.

    Returns:
        Matriz booleana onde True = Água calma com fundo visível (sem glint nem esteiras).
    """
    from scipy.ndimage import binary_dilation, uniform_filter

    g = np.nan_to_num(green, nan=0.0).astype(np.float32)
    mean = uniform_filter(g, size=window, mode="reflect")
    sqmean = uniform_filter(g ** 2, size=window, mode="reflect")
    var = np.maximum(sqmean - mean ** 2, 0.0)
    cv = np.where(mean > 1e-6, np.sqrt(var) / np.maximum(mean, 1e-6), 0.0)

    if nir is not None:
        n = np.nan_to_num(nir, nan=0.0).astype(np.float32)
        
        # Determina o limiar adaptativo de NIR para água limpa caso nir_thresh seja None
        if nir_thresh is None:
            rough_water = (g > n) & (n < 0.15) & (g > 0.01)
            if np.sum(rough_water) > 100:
                p5 = float(np.percentile(n[rough_water], 5))
                eff_nir_thresh = max(0.035, min(0.065, p5 + 0.022))
                var_cutoff_nir = p5 + 0.012
            else:
                eff_nir_thresh = 0.035
                var_cutoff_nir = 0.025
        else:
            eff_nir_thresh = nir_thresh
            var_cutoff_nir = max(0.020, eff_nir_thresh - 0.010)

        # Anomalias: reflectância NIR anómala ou combinação de NIR elevado com alta turbilhonância (esteiras)
        is_anomaly = (n > eff_nir_thresh) | ((n > var_cutoff_nir) & (cv > var_thresh))
        
        # Dilatação morfológica para englobar a franja difusa de espuma e esteiras de embarcações
        if dilate_anomaly_px > 0:
            footprint = np.ones((2 * dilate_anomaly_px + 1, 2 * dilate_anomaly_px + 1), dtype=bool)
            is_anomaly = binary_dilation(is_anomaly, structure=footprint)
            
        clean_mask = ~is_anomaly
    else:
        clean_mask = cv < var_thresh
        eff_nir_thresh = None

    total_pixels = clean_mask.size
    clean_count = int(np.sum(clean_mask))
    thresh_str = f"{eff_nir_thresh:.4f} (adaptativo)" if eff_nir_thresh is not None else "N/A"
    logger.info(
        f"Máscara B08 (Glint/Esteiras/Espuma) aplicada | Limiar NIR: {thresh_str} | "
        f"Pixels válidos com fundo visível: {clean_count}/{total_pixels} ({(clean_count/total_pixels)*100:.1f}%)"
    )
    return clean_mask


def build_clean_water_mask(
    scene: SentinelScene,
    ndwi_threshold: float = 0.0,
    apply_glint_filter: bool = True,
    nir_thresh: float | None = None,
) -> np.ndarray:
    """
    Gera uma máscara de coluna de água estrita e limpa:
    1. NDWI isola o oceano da terra firme e falésias do Algarve.
    2. Filtro de glint/espuma remove zonas de arrebentação na praia, esteiras de barcos e reflexos solares.
    """
    ndwi, water_mask = calculate_ndwi(scene.green, scene.nir, red=scene.red, threshold=ndwi_threshold)
    scene.ndwi = ndwi

    if apply_glint_filter:
        glint_mask = build_glint_mask(scene.green, scene.nir, nir_thresh=nir_thresh)
        combined_mask = water_mask & glint_mask
    else:
        combined_mask = water_mask

    return combined_mask


def apply_water_mask(
    scene: SentinelScene,
    water_mask: np.ndarray,
    fill_value: float = np.nan,
) -> SentinelScene:
    """
    Aplica a máscara de água sobre todas as bandas espectrais, isolando o oceano
    e anulando os pixels terrestres ou de terra firme do Algarve.

    Args:
        scene: Instância de SentinelScene.
        water_mask: Matriz booleana onde True = Água.
        fill_value: Valor a atribuir a pixels de terra (padrão np.nan).

    Returns:
        Nova instância de SentinelScene apenas com pixels marinhos.
    """
    masked_blue = np.where(water_mask, scene.blue, fill_value)
    masked_green = np.where(water_mask, scene.green, fill_value)
    masked_red = np.where(water_mask, scene.red, fill_value)
    masked_nir = np.where(water_mask, scene.nir, fill_value) if scene.nir is not None else None

    scene.water_mask = water_mask
    return SentinelScene(
        blue=masked_blue,
        green=masked_green,
        red=masked_red,
        nir=masked_nir,
        profile=scene.profile,
        crs=scene.crs,
        transform=scene.transform,
        water_mask=water_mask,
        ndwi=scene.ndwi,
    )


def save_geotiff(
    output_path: str | Path,
    data: np.ndarray,
    profile: dict[str, Any],
    nodata: float = -9999.0,
) -> Path:
    """
    Exporta uma matriz 2D ou 3D como ficheiro GeoTIFF geo-referenciado.

    Args:
        output_path: Caminho do ficheiro de saída (.tif).
        data: Matriz NumPy 2D (H, W) ou 3D (C, H, W).
        profile: Perfil Rasterio contendo CRS e transform.
        nodata: Valor representativo de NoData.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    out_profile = profile.copy()
    if data.ndim == 2:
        count = 1
        h, w = data.shape
        data_to_write = data[np.newaxis, :, :]
    elif data.ndim == 3:
        count, h, w = data.shape
        data_to_write = data
    else:
        raise ValueError(f"Dimensão de matriz não suportada: {data.ndim}")

    out_profile.update(
        driver="GTiff",
        height=h,
        width=w,
        count=count,
        dtype=rasterio.float32,
        nodata=nodata,
    )

    clean_data = np.nan_to_num(data_to_write, nan=nodata)

    with rasterio.open(out_file, "w", **out_profile) as dst:
        dst.write(clean_data.astype(np.float32))

    logger.info(f"GeoTIFF exportado com sucesso: {out_file}")
    return out_file


# =============================================================================
# AUTO-TESTE E DEMONSTRAÇÃO SINTÉTICA (Algarve / Pedra do Valado)
# =============================================================================
if __name__ == "__main__":
    import tempfile

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print("\n" + "=" * 70)
    print(" TESTE DE SANIDADE: SENTINEL READER & NDWI WATER MASK")
    print("=" * 70)

    # 1. Criação de um GeoTIFF sintético 4-bandas representando a costa de Armação de Pêra
    # Coordenadas aproximadas: UTM Zone 29N (EPSG:32629)
    width, height = 256, 256
    mock_transform = Affine(10.0, 0.0, 560000.0, 0.0, -10.0, 4104000.0)
    mock_crs = CRS.from_epsg(32629)

    # Simulação de linha de costa (metade norte = terra firme, metade sul = oceano)
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    is_land = y_coords < (height // 2)

    # Reflectâncias típicas:
    # Terra: B02 baixo, B03 médio, B04 moderado, B08 (NIR) MUITO ALTO (vegetação/solo)
    # Mar: B02 alto (água limpa), B03 alto (substrato costeiro/recife), B04 baixo, B08 ~ 0 (absorção NIR)
    b02 = np.where(is_land, 0.04, 0.18).astype(np.float32)  # Azul
    b03 = np.where(is_land, 0.07, 0.15).astype(np.float32)  # Verde
    b04 = np.where(is_land, 0.09, 0.03).astype(np.float32)  # Vermelho
    b08 = np.where(is_land, 0.35, 0.01).astype(np.float32)  # NIR

    # Anomalia de recife submerso no oceano (aumento ligeiro de reflectância verde/azul)
    reef_spot = (~is_land) & ((x_coords - 128) ** 2 + (y_coords - 190) ** 2 < 20**2)
    b03[reef_spot] += 0.05
    b02[reef_spot] += 0.03

    stack = np.stack([b02, b03, b04, b08], axis=0)

    with tempfile.TemporaryDirectory() as tmpdir:
        test_tif = Path(tmpdir) / "sentinel2_algarve_sample.tif"
        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 4,
            "dtype": rasterio.float32,
            "crs": mock_crs,
            "transform": mock_transform,
        }
        with rasterio.open(test_tif, "w", **profile) as dst:
            dst.write(stack)

        print(f"\n1. Ficheiro sintético GeoTIFF criado em: {test_tif.name}")

        # 2. Teste de Leitura
        scene = read_sentinel_bands(test_tif, normalize_reflectance=False)
        print(f"2. Cena lida com sucesso: {scene.shape[0]}x{scene.shape[1]} px | CRS: {scene.crs}")

        # 3. Teste de NDWI
        ndwi, mask = calculate_ndwi(scene.green, scene.nir, threshold=0.0)
        scene.ndwi = ndwi
        print(f"3. NDWI Intervalo: [{np.nanmin(ndwi):.3f}, {np.nanmax(ndwi):.3f}]")

        # 4. Aplicação da Máscara
        masked_scene = apply_water_mask(scene, mask)
        print(f"4. Máscara aplicada: {np.sum(mask)} pixels oceânicos isolados.")

        # Validação: Terra mascarada com NaN
        assert np.isnan(masked_scene.blue[0, 0]), "A terra não foi mascarada como NaN!"
        assert not np.isnan(masked_scene.blue[height - 1, width - 1]), "O oceano foi incorretamente mascarado!"

    print("\n" + "=" * 70)
    print(" TODOS OS TESTES DO SENTINEL_READER PASSARAM COM DISTINÇÃO! ")
    print("=" * 70)
