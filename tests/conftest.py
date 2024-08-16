import shutil
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def test_cleaner():
    funcs = []

    def add_func(func):
        funcs.append(func)

    yield add_func
    for func in funcs:
        func()


@pytest.fixture(scope="session")
def test_directory() -> Path:
    folder_path = Path(__file__).parent
    assert folder_path.exists()
    return folder_path


@pytest.fixture(scope="session")
def tmp_files_folder(test_cleaner, test_directory) -> Path:
    folder_path = test_directory / "tmp_folder_for_tests"
    if folder_path.exists():
        shutil.rmtree(folder_path)
    folder_path.mkdir()
    test_cleaner(lambda: shutil.rmtree(folder_path))
    return folder_path


@pytest.fixture(scope="session")
def dataset_examples_folder(test_directory) -> Path:
    folder_path = test_directory.parent / "dataset_examples"
    assert folder_path.exists()
    return folder_path
