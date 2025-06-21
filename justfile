run:
    # Run the hole training process.
    uv run dvc repro

prepare:
    # Run only the prepare stage.
    uv run dvc repro prepare

init-dvc-localremote:
    # Initialize a local DVC remote for storing data.
    # Only run this once per project.
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


api:
    (cd apps/api && uv run python -m src.api.run)

app name:
    mkdir -p apps
    (cd apps && uv init --build-backend hatch --lib {{name}})
    touch apps/{{name}}/src/{{snakecase(name)}}/__main__.py
    just generate-justfile apps/{{name}}

lib name:
    mkdir -p libs
    (cd libs && uv init --build-backend hatch --lib {{name}})
    just generate-justfile libs/{{name}}

setup:
    uv sync --all-packages

fix:
    uv run ruff format .
