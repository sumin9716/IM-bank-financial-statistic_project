import pandas as pd
import pytest

from tests.synthetic import make_raw_frame


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    return make_raw_frame()
