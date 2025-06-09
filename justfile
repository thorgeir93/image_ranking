run:
    uv run dvc repro

prepare:
    uv run dvc repro prepare
    # uv run dvc repro prepare-lower-body-good

init-dvc-localremote:
    mkdir -p ~/dvc-remote
    uv run dvc remote add -d localremote ~/dvc-remote
