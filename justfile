run:
    uv run dvc repro

prepare:
    uv run dvc repro prepare
    # uv run dvc repro prepare-lower-body-good

init-dvc-localremote:
    mkdir -p ~/dvc-remote
    uv run dvc remote add -d localremote ~/dvc-remote


app name:
    mkdir -p apps
    (cd apps && uv init --build-backend hatch --lib {{name}})
    touch apps/{{name}}/src/{{snakecase(name)}}/__main__.py

fix:
    uv run ruff format .
