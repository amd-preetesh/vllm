# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionBackend,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.rocm_aiter_fa import (
    AiterFlashAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.ultraquant_spec_compat_attn import (
    UltraQuantSpecCompatMetadataBuilder,
)


@pytest.mark.cpu_test
def test_gdn_supports_adaptive_query_lens_mismatch():
    assert GDNAttentionBackend.supports_device_cpu_query_lens_mismatch()


@pytest.mark.cpu_test
def test_gdn_and_uq_spec_compat_adaptive_builders_support_varlen_full_graphs():
    adaptive_config = SimpleNamespace(
        speculative_config=SimpleNamespace(enable_adaptive_verification=True)
    )
    fixed_config = SimpleNamespace(
        speculative_config=SimpleNamespace(enable_adaptive_verification=False)
    )

    for builder_cls in (
        GDNAttentionMetadataBuilder,
        UltraQuantSpecCompatMetadataBuilder,
        AiterFlashAttentionMetadataBuilder,
    ):
        assert (
            builder_cls.get_cudagraph_support(adaptive_config, SimpleNamespace())
            == AttentionCGSupport.ALWAYS
        )
        assert (
            builder_cls.get_cudagraph_support(fixed_config, SimpleNamespace())
            == AttentionCGSupport.UNIFORM_BATCH
        )
