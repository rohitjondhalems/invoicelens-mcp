from pathlib import Path

import pytest

from invoicelens.config import Settings
from invoicelens.service import InvoiceLensService

SAMPLES = Path(__file__).parent.parent / "samples"


def sample(name: str) -> bytes:
    return (SAMPLES / name).read_bytes()


ACME, NORTHWIND = "acme_supplies_2026-03.pdf", "northwind_logistics_2026-04.pdf"
GLOBEX, INITECH = "globex_consulting_2026-05.pdf", "initech_software_2026-06.pdf"
ALL = [ACME, NORTHWIND, GLOBEX, INITECH]


@pytest.fixture
def settings(tmp_path) -> Settings:
    # Offline providers: no keys, deterministic
    return Settings(data_dir=tmp_path / "data", embedding_provider="local", extraction_provider="local",
                    _env_file=None)


@pytest.fixture
def service(settings) -> InvoiceLensService:
    return InvoiceLensService(settings)


@pytest.fixture
def loaded(service) -> InvoiceLensService:
    for name in ALL:
        service.ingest(sample(name), name)
    return service
