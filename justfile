run:
    uv run dvc repro

prepare:
    uv run dvc repro prepare
    # uv run dvc repro prepare-lower-body-good

init-dvc-localremote:
    mkdir -p ~/dvc-remote
    uv run dvc remote add -d localremote ~/dvc-remote

# Example usage: $ just generate-justfile apps/prepare
generate-justfile project_path:
    touch {{project_path}}/justfile
    echo "fix:" >> {{project_path}}/justfile
    echo "    uv run ruff format ." >> {{project_path}}/justfile
    echo "check:" >> {{project_path}}/justfile
    echo "    uv run ruff format . --check" >> {{project_path}}/justfile
    echo "    uv run mypy ." >> {{project_path}}/justfile


app name:
    mkdir -p apps
    (cd apps && uv init --build-backend hatch --lib {{name}})
    touch apps/{{name}}/src/{{snakecase(name)}}/__main__.py
    just generate-justfile apps/{{name}}

lib name:
    mkdir -p libs
    (cd libs && uv init --build-backend hatch --lib {{name}})
    just generate-justfile libs/{{name}}

fix:
    uv run ruff format .
