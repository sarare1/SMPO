"""Shared test set-up.

WHAT THIS FILE DOES (plain English)
-----------------------------------
pytest (the test runner) reads this file automatically. It prepares the things
many tests need - the loaded prediction models and a toolbox pointed at a known
factory moment - so each test does not have to set them up itself.

If the data has not been downloaded and the models not trained yet, the tests
that need them are skipped with a message explaining what to run first.
"""
# --- Imports: tools this file needs -----------------------------------------
import pytest  # the test framework

from src import config  # project folders, to check that trained models exist


@pytest.fixture(scope="session")
def artifacts_ready():
    """Skip model-based tests if the data and trained models are not there yet."""
    needed = [
        config.MODELS_DIR / "ai4i_models.joblib",
        config.MODELS_DIR / "azure_models.joblib",
        config.DATA_PROCESSED / "azure_features.parquet",
    ]
    if not all(p.exists() for p in needed):
        pytest.skip("Run `python -m src.data.download`, `src.data.features` and `src.models.train` first.")


@pytest.fixture(scope="session")
def process_model(artifacts_ready):
    """The machining-line model, loaded once for the whole test run."""
    from src.models.predictor import get_process_model
    return get_process_model()


@pytest.fixture(scope="session")
def fleet_model(artifacts_ready):
    """The fleet model, loaded once for the whole test run."""
    from src.models.predictor import get_fleet_model
    return get_fleet_model()


@pytest.fixture
def toolbox(fleet_model, process_model):
    """A toolbox at a fixed factory moment, running a known heat-dissipation failure cycle."""
    from src.agents.tools import ToolBox
    from src.simulation.stream import FactoryState
    tc = process_model.test_cycles
    return ToolBox(FactoryState(ts_index=200, cycle_index=int(tc.index[tc["HDF"] == 1][0])))
