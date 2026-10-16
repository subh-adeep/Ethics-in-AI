"""Publish a private Kaggle draft without allocating another GPU session."""
import argparse
import json
import base64
import zlib
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiSaveKernelRequest
from kagglesdk.kernels.types.kernels_enums import KernelExecutionType


def publish(directory):
    directory = Path(directory)
    meta = json.loads((directory / "kernel-metadata.json").read_text(encoding="utf-8"))
    api = KaggleApi()
    api.authenticate()
    request = ApiSaveKernelRequest()
    source = (directory / meta["code_file"]).read_text(encoding="utf-8")
    encoded = base64.b64encode(zlib.compress(source.encode(), 9)).decode()
    # Kaggle QUICK_SAVE is a notebook path. A cell writes and executes the
    # unchanged standalone runner, preserving __file__ and source fingerprints.
    cell = ("import base64, zlib, runpy, sys\nfrom pathlib import Path\n"
            f"runner_source = zlib.decompress(base64.b64decode({encoded!r})).decode()\n"
            "runner_path = Path('/kaggle/working/tools_v2_runner.py')\n"
            "runner_path.write_text(runner_source, encoding='utf-8')\n"
            "previous_argv = sys.argv\nsys.argv = [str(runner_path)]\n"
            "try:\n    runpy.run_path(str(runner_path), run_name='__main__')\n"
            "finally:\n    sys.argv = previous_argv\n")
    notebook = {"cells": [{"id": "intro", "cell_type": "markdown", "metadata": {}, "source": [
                    "# Main-case tool benchmark\n",
                    "Run the cell below on a GPU with Internet enabled. It executes the frozen, tested runner.\n",
                    "T0/T1/T2 use the same 421 verified questions. The separate progressive-flip experiment is excluded.\n"]},
                {"id": "runner", "cell_type": "code", "metadata": {}, "source": cell.splitlines(keepends=True),
                 "execution_count": None, "outputs": []}],
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
                "nbformat": 4, "nbformat_minor": 5}
    (directory / "notebook.ipynb").write_text(json.dumps(notebook), encoding="utf-8")
    request.slug = meta["id"] + "-notebook"
    request.new_title = meta["title"] + " Notebook"
    request.text = json.dumps(notebook)
    request.language = "python"
    request.kernel_type = "notebook"
    request.is_private = True
    request.enable_internet = True
    request.machine_shape = "NvidiaTeslaT4"
    request.kernel_execution_type = KernelExecutionType.QUICK_SAVE
    with api.build_kaggle_client() as client:
        response = client.kernels.kernels_api_client.save_kernel(request)
    if response.error:
        raise RuntimeError(response.error)
    record = {"ref": response.ref, "url": response.url, "version": response.version_number,
              "mode": "private_quick_save"}
    (directory / "publication.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    publish(parser.parse_args().directory)
