import itertools

import pytest
from workflow_helpers import canonical_request, canonical_result

from recommendation_agent.selection.selector import DeterministicResultSelector


@pytest.mark.asyncio
async def test_non_tied_input_permutations_keep_candidate_and_type_semantics() -> None:
    request = canonical_request()
    result = canonical_result()
    expected = await DeterministicResultSelector().select(request, result)
    for permutation in itertools.islice(itertools.permutations(result.candidates), 12):
        permuted = type(result)(
            model_version=result.model_version,
            model_package_version=result.model_package_version,
            feature_schema_version=result.feature_schema_version,
            inference_contract_version=result.inference_contract_version,
            model_key_version=result.model_key_version,
            candidates=permutation,
        )
        actual = await DeterministicResultSelector().select(request, permuted)
        assert [(item.candidate_id, item.recommendation_type) for item in actual] == [
            (item.candidate_id, item.recommendation_type) for item in expected
        ]
