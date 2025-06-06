run:
    uv run dvc repro

prepare:
    uv run dvc repro prepare

init-dvc-localremote:
    mkdir -p ~/dvc-remote
    uv run dvc remote add -d localremote ~/dvc-remote
