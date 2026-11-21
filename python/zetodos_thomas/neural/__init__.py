# SPDX-License-Identifier: Apache-2.0
"""From-scratch offline neural Coptic->Koine NMT, Koine LM & noisy-channel reranking.

Strictly isolated from P.Oxy. fragment witnesses and Dilley Thomas during training,
tokenizer fitting, back-translation, and dev-set tuning.
"""

from __future__ import annotations
