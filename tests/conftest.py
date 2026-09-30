import pytest

from src import config


@pytest.fixture(scope="session")
def artifacts_ready():
    needed = [
        config.MODELS_DIR / "ai4i_models.joblib",
        config.MODELS_DIR / "azure_models.joblib",
        config.DATA_PROCESSED / "azure_features.parquet",
    ]
    if not all(p.exists() for p in needed):
        pytest.skip("Run `python -m src.data.download`, `src.data.features` and `src.models.train` first.")


@pytest.fixture(scope="session")
def process_model(artifacts_ready):
    from src.models.predictor import get_process_model
    return get_process_model()


@pytest.fixture(scope="session")
def fleet_model(artifacts_ready):
    from src.models.predictor import get_fleet_model
    return get_fleet_model()


@pytest.fixture
def toolbox(fleet_model, process_model):
    from src.agents.tools import ToolBox
    from src.simulation.stream import FactoryState
    tc = process_model.test_cycles
    return ToolBox(FactoryState(ts_index=200, cycle_index=int(tc.index[tc["HDF"] == 1][0])))
