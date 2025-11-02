# -*- coding: utf-8 -*-
"""Constants for EspressoPro."""

from __future__ import annotations

# Reference families used to build class maps
_REF_SIMPLIFIED = ("Averaged.Simplified",)
_REF_DETAILED   = ("Averaged.Detailed",)

# Detailed labels (suffixes must match predscore columns)
_DETAILED_LABELS = [
    "CD14_Mono", "CD16_Mono", "cDC1", "cDC2", "ILC", "Macrophage", "MkP",
    "HSC_MPP", "MEP", "ErP", "GMP", "Pre-Pro-B", "EoBaMaP", "Plasma",
    "B_Naive", "B_Memory", "Immature_B", "CD4_T_Naive", "CD4_T_Memory", "CD4_CTL",
    "Treg", "CD8_T_Naive", "CD8_T_Memory", "MAIT", "NK_CD56_dim", "NK_CD56_bright",
    "Erythroblast", "Stroma", "pDC", "gdT", "Myeloid_progenitor",
    "Pre-B", "Pro-B",
]

# Simplified class → sources to vote over
SIMPLIFIED_CLASSES = {
    lbl: [f"{ref}.{lbl}.predscore" for ref in _REF_SIMPLIFIED]
    for lbl in (
        "NK", "HSPC", "Erythroid", "pDC", "Monocyte", "Myeloid",
        "CD4_T", "CD8_T", "B", "cDC", "Other_T", "Plasma",
    )
}

# Detailed class → sources to vote over
DETAILED_CLASSES = {
    lbl: [f"{ref}.{lbl}.predscore" for ref in _REF_DETAILED]
    for lbl in _DETAILED_LABELS
}

# Broad/Simplified parent → allowed Simplified preds
SIMPLIFIED_PARENT_MAP = {
    "Immature": ["Averaged.Simplified.HSPC.predscore"],
    "Mature":   [f"Averaged.Simplified.{l}.predscore" for l in SIMPLIFIED_CLASSES if l != "HSPC"],
}

# Simplified parent → allowed Detailed preds
DETAILED_PARENT_MAP = {
    "HSPC": [
        "Averaged.Detailed.HSC_MPP.predscore",
        "Averaged.Detailed.GMP.predscore",
        "Averaged.Detailed.Pre-Pro-B.predscore",
        "Averaged.Detailed.MEP.predscore",
        "Averaged.Detailed.Pro-B.predscore",
    ],
    "Erythroid": [
        "Averaged.Detailed.HSC_MPP.predscore",
        "Averaged.Detailed.MEP.predscore",
        "Averaged.Detailed.ErP.predscore",
        "Averaged.Detailed.Erythroblast.predscore",
    ],
    "pDC": ["Averaged.Detailed.pDC.predscore"],
    "Monocyte": [
        "Averaged.Detailed.CD14_Mono.predscore",
        "Averaged.Detailed.CD16_Mono.predscore",
    ],
    "Myeloid": ["Averaged.Detailed.Myeloid_progenitor.predscore"],
    "cDC": [
        "Averaged.Detailed.cDC1.predscore",
        "Averaged.Detailed.cDC2.predscore",
    ],
    "Other_T": [
        "Averaged.Detailed.gdT.predscore",
    ],
    "NK": [
        "Averaged.Detailed.NK_CD56_dim.predscore",
        "Averaged.Detailed.NK_CD56_bright.predscore",
    ],
    "CD4_T": [
        "Averaged.Detailed.CD4_T_Naive.predscore",
        "Averaged.Detailed.CD4_T_Memory.predscore",
        "Averaged.Detailed.CD4_CTL.predscore",
        "Averaged.Detailed.Treg.predscore",
    ],
    "CD8_T": [
        "Averaged.Detailed.CD8_T_Naive.predscore",
        "Averaged.Detailed.CD8_T_Memory.predscore",
        "Averaged.Detailed.MAIT.predscore",
    ],
    "B": [
        "Averaged.Detailed.Plasma.predscore",
        "Averaged.Detailed.B_Naive.predscore",
        "Averaged.Detailed.B_Memory.predscore",
        "Averaged.Detailed.Immature_B.predscore",
        "Averaged.Detailed.Pre-B.predscore",
        "Averaged.Detailed.Pro-B.predscore",
    ],
}

# Mast cell signatures
MAST_POS = ['FcεRIα', 'CD117', 'CD62L']
MAST_NEG = [
    'CD303', 'CD304', 'CD123', 'CD34', 'CD8', 'CD4',
    'CD138', 'CD7', 'CD10', 'CD11b', 'CD5', 'CD141', 'CD1c',
]
