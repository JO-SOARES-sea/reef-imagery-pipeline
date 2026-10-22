#!/usr/bin/env python3
"""
Pipeline de Deteção de Recifes Marinhos Naturais (Costa do Algarve)
Ponto de entrada: Verificação de conectividade e validação com TypeSafe Jev.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

from src.analysis.gemini_analyzer import JEV_DECISION_LABELS, interpret_reef_noul

# Configuração de Logging Estruturado
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("reef_pipeline.validation")

# 1. Carregamento de variáveis de ambiente
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    logger.info("Ficheiro .env carregado com sucesso.")
else:
    load_dotenv()
    logger.warning("Ficheiro .env local não encontrado no caminho padrão; recorrendo ao ambiente global.")

TYPESAFE_API_KEY = os.getenv("TYPESAFE_API_KEY")

if not TYPESAFE_API_KEY:
    logger.error("A variável de ambiente 'TYPESAFE_API_KEY' não está configurada no .env.")
    sys.exit(1)

# 2. Definição do cenário de teste para a costa do Algarve
SAMPLE_STATE = (
    "Relatório de Deteção Sentinel-2 na costa de Armação de Pêra / Pedra do Valado (Algarve): "
    "Coordenadas [37.0725° N, -8.3241° W]. Análise de reflectância multiespectral nas bandas B02 (Azul), "
    "B03 (Verde) e B04 (Vermelho) corrigidas para atenuação da coluna de água a 14 metros de profundidade. "
    "Identificada formação rochosa contínua com rugosidade batimétrica elevada, associada a bio-construções "
    "e colónias de gorgónias/algas calcárias. Ausência de traços de geo-engenharia artificial ou quebra-mares."
)

SAMPLE_QUESTIONS = {
    "is_natural_reef": {
        "type": "noul",
        "instructions": (
            "Com base nas evidências de reflectância e morfologia subaquática, "
            "o padrão indica a presença de um recife marinho natural?"
        ),
    }
}


def test_via_sdk(api_key: str, state: str, questions: dict) -> dict:
    """Execução através da biblioteca oficial typesafe-sdk."""
    from typesafe_sdk import Noul, TypeSafeClient

    sdk_questions = {
        q_id: Noul(instructions=q_meta["instructions"])
        for q_id, q_meta in questions.items()
        if q_meta.get("type") == "noul"
    }

    t0 = time.perf_counter()
    with TypeSafeClient(api_key=api_key) as client:
        result = client.system_one(state=state, questions=sdk_questions, model="jev-latest")
    latency_ms = (time.perf_counter() - t0) * 1000

    return {
        "client": "typesafe-sdk",
        "latency_ms": latency_ms,
        **interpret_reef_noul(result.nouls["is_natural_reef"].noul),
    }


def test_via_rest(api_key: str, state: str, questions: dict) -> dict:
    """Fallback direto via REST HTTP nativo sem dependências externas."""
    import urllib.request

    endpoint = "https://api.typesafe.ai/v1/systemone"
    payload = {
        "model": "jev-latest",
        "state": state,
        "questions": questions,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "AlgarveReefPipeline/1.0",
    }

    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=10) as response:
        latency_ms = (time.perf_counter() - t0) * 1000
        raw_body = response.read().decode("utf-8")
        data = json.loads(raw_body)

    # Resposta sem o Noul → None ("não executado"), nunca 0.0 (que seria um "não" falso)
    noul_val = data.get("answers", {}).get("is_natural_reef", {}).get("noul")
    return {
        "client": "rest-api",
        "latency_ms": latency_ms,
        **interpret_reef_noul(noul_val),
        "raw": data,
    }


def main():
    logger.info("Iniciando verificação de conectividade com TypeSafe Jev...")

    try:
        try:
            import typesafe_sdk  # noqa: F401
            res = test_via_sdk(TYPESAFE_API_KEY, SAMPLE_STATE, SAMPLE_QUESTIONS)
        except ImportError:
            logger.info("typesafe-sdk não importado; executando chamada direta via REST API...")
            res = test_via_rest(TYPESAFE_API_KEY, SAMPLE_STATE, SAMPLE_QUESTIONS)

        print("\n" + "=" * 65)
        print(" TESTE DE CONEXÃO E VALIDAÇÃO: TYPESAFE JEV")
        print("=" * 65)
        print(f" Cliente Utilizado      : {res['client']}")
        print(f" Latência de Execução   : {res['latency_ms']:.1f} ms")
        print(f" Decisão                : {JEV_DECISION_LABELS[res['decision']]}")
        if res["p_reef"] is not None:
            print(f" P(recife natural)      : {res['p_reef'] * 100:.2f}%")
            print(f" Confiança |2p−1|       : {res['confidence'] * 100:.2f}%")
        print("=" * 65)
        logger.info("Conectividade e validação concluídas com sucesso.")

    except Exception as exc:
        logger.error(f"Falha na comunicação com a API TypeSafe: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
