# Running style training model

## Managing data/*

### Example - adding new image to data/raw

```
# Adding the new image
cp my_new_image.jpg data/raw/lower_body/bad/

# Re-run pipeline to generate dvc.lock (see changes)
uv run dvc repro

# Stage and commit updated version
git add dvc.lock
git commit -m "Add new image to data/raw"
uv run dvc push
```
