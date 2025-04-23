import pytest
from huggingface_hub.constants import _staging_mode


@pytest.fixture
def staging():
    """A pytest fixture only available in huggingface_hub staging mode

    If the huggingface_hub is not operating in staging mode, tests using
    that fixture are automatically skipped.

    Returns:
        a Dict containing a valid staging user and token.
    """
    if not _staging_mode:
        pytest.skip("requires huggingface_hub staging mode")
    return {
        "user": "__DUMMY_TRANSFORMERS_USER__",
        # Not critical, only usable on the sandboxed CI instance.
        "token": "hf_94wBhPGp6KrrTH3KDchhKpRxZwd6dmHWLL",
    }


@pytest.fixture(autouse=True)
def skip_if_staging(request):
    if _staging_mode:
        if "staging" not in request.fixturenames:
            pytest.skip("requires huggingface_hub standard mode")
