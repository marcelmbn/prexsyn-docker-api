from __future__ import annotations

import pathlib
import time
from functools import lru_cache
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from rdkit import Chem
from rdkit.Chem import rdChemReactions

from prexsyn.applications.analog import generate_analogs
from prexsyn.factories import load_model
from prexsyn.samplers.basic import BasicSampler
from prexsyn_engine.fingerprints import tanimoto_similarity
from prexsyn_engine.synthesis import Synthesis

app = FastAPI(title="PrexSyn Service", version="1.0")

DEFAULT_MODEL_PATH = pathlib.Path("data/trained_models/v1_converted.yaml")
MAX_SMILES = 256


class PredictRequest(BaseModel):
    smiles: str | list[str] = Field(
        ...,
        description="Single SMILES, list of SMILES, or newline/comma-separated string.",
    )
    top: int = Field(default=10, ge=1, le=100)
    num_samples: int = Field(default=64, ge=1, le=512)
    max_length: int = Field(default=16, ge=1, le=128)


def _parse_smiles(smiles_input: str | list[str]) -> list[str]:
    if isinstance(smiles_input, list):
        smiles_list = [s.strip() for s in smiles_input if s.strip()]
        if not smiles_list:
            raise ValueError("No SMILES strings provided.")
        return smiles_list

    chunks: list[str] = []
    for line in smiles_input.replace("\r", "\n").split("\n"):
        for part in line.split(","):
            token = part.strip()
            if token:
                chunks.append(token)

    if not chunks:
        raise ValueError("No SMILES strings were detected.")
    return chunks


def _indent_lines(text: str, level: int, indent: str = "  ") -> str:
    return "\n".join(indent * level + line for line in text.splitlines())


def _synthesis_to_string(synthesis: Synthesis) -> str:
    replay = Synthesis()
    pfn_list = synthesis.get_postfix_notation().to_list()
    text_stack: list[str] = []

    for item in pfn_list:
        if isinstance(item, Chem.Mol):
            replay.push_mol(item)
            smi = Chem.MolToSmiles(item, canonical=True)
            idx = item.GetProp("building_block_index")
            text = f"- SMILES: {smi}\n"
            text += f"  Building Block Index: {idx}\n"
            if item.HasProp("id"):
                text += f"  ID: {item.GetProp('id')}\n"
            text_stack.append(text.strip())
        elif isinstance(item, rdChemReactions.ChemicalReaction):
            replay.push_reaction(item)
            prod_list = replay.top().to_list()
            prod_smi_set = sorted({Chem.MolToSmiles(mol, canonical=True) for mol in prod_list})
            num_reactants = item.GetNumReactantTemplates()

            idx = item.GetProp("reaction_index")
            text = f"- Reaction Index: {idx}\n"
            text += "  Possible Products:\n"
            product_text = "\n".join(f"- {smi}" for smi in prod_smi_set)
            text += _indent_lines(product_text, 1) + "\n"
            text += "  Reactants:\n"
            reactant_text = "\n".join(text_stack[-num_reactants:])
            text += _indent_lines(reactant_text, 1) + "\n"

            text_stack = text_stack[:-num_reactants]
            text_stack.append(text.strip())

    return "\n".join(text_stack)


@lru_cache(maxsize=1)
def _load_prexsyn() -> tuple[Any, Any, str]:
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    facade, model = load_model(DEFAULT_MODEL_PATH, train=False)
    model = model.to(device)
    return facade, model, device


def _run_for_smiles(
    facade: Any,
    model: Any,
    smiles: str,
    top: int,
    num_samples: int,
    max_length: int,
) -> dict[str, Any]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    canonical_smi = Chem.MolToSmiles(mol, canonical=True)

    sampler = BasicSampler(
        model,
        token_def=facade.tokenization.token_def,
        num_samples=num_samples,
        max_length=max_length,
    )

    result = generate_analogs(
        facade=facade,
        model=model,
        sampler=sampler,
        fp_property=facade.property_set["ecfp4"],
        mol=mol,
    )

    visited: set[str] = set()
    result_list: list[tuple[str, float, str]] = []
    for synthesis in result["synthesis"]:
        if synthesis.stack_size() != 1:
            continue
        for prod in synthesis.top().to_list():
            prod_smi = Chem.MolToSmiles(prod, canonical=True)
            if prod_smi in visited:
                continue
            visited.add(prod_smi)
            sim = tanimoto_similarity(prod, mol, fp_type="ecfp4")
            synthesis_text = _synthesis_to_string(synthesis)
            result_list.append((prod_smi, sim, synthesis_text))

    result_list.sort(key=lambda x: x[1], reverse=True)
    return {
        "input_smiles": smiles,
        "target_canonical_smiles": canonical_smi,
        "results": [
            {
                "smiles": prod_smi,
                "similarity": round(sim, 6),
                "synthesis": synthesis_text,
            }
            for prod_smi, sim, synthesis_text in result_list[:top]
        ],
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: PredictRequest) -> dict[str, Any]:
    try:
        smiles_list = _parse_smiles(payload.smiles)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if len(smiles_list) > MAX_SMILES:
        raise HTTPException(
            status_code=413,
            detail=f"Too many SMILES (max {MAX_SMILES}).",
        )

    if not DEFAULT_MODEL_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Model file not found at {DEFAULT_MODEL_PATH}",
        )

    try:
        facade, model, device = _load_prexsyn()
        start = time.perf_counter()
        outputs = [
            _run_for_smiles(
                facade=facade,
                model=model,
                smiles=smi,
                top=payload.top,
                num_samples=payload.num_samples,
                max_length=payload.max_length,
            )
            for smi in smiles_list
        ]
        runtime = time.perf_counter() - start
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    return {
        "molecule_count": len(smiles_list),
        "top": payload.top,
        "num_samples": payload.num_samples,
        "max_length": payload.max_length,
        "runtime_seconds": round(runtime, 3),
        "device": device,
        "predictions": outputs,
    }
