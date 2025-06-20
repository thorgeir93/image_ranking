# Running style training model

## Background
This project was created to make photography post-processing faster and easier. Editing photos manually takes a lot of time and effort. By training a model to score images based on running style, we can speed up this process.

## About
This project uses training data where a running person have been cropped out of image and split into lower and upper body (prepare stage).
The output model should be able to determine from running legs whether it is a good style or a bad style, scoring from 0.0 to 1.0.

## Version control
This project uses `dvc` to version control the dataset, models, parameters and the code.

## Visulize the DVC workflow

```
$ uv run dvc dag
                                                           +--------------------+         
                                                           | data/raw/train.dvc |         
                                                           +--------------------+         
                                                               ***         ***            
                                                              *               *           
                                                            **                 **         
      +-----------------------+                     +-----------+           +-----------+ 
      | data/raw/evaluate.dvc |                     | prepare@0 |           | prepare@1 | 
      +-----------------------+                     +-----------+           +-----------+ 
           ***         ***                                     ***         ***            
          *               *                                       *       *               
        **                 **                                      **   **                
+-----------+           +-----------+                         +-------------+             
| prepare@2 |           | prepare@3 |                         | featurize@0 |             
+-----------+           +-----------+                         +-------------+             
           ***         ***                                            *                   
              *       *                                               *                   
               **   **                                                *                   
           +-------------+                                      +---------+               
           | featurize@1 |                                   ***| train@0 |               
           +-------------+                         **********   +---------+               
                  *                      **********                   *                   
                  *            **********                             *                   
                  *       *****                                       *                   
            +------------+                                      +----------+              
            | evaluate@0 |                                      | export@0 |              
            +------------+                                      +----------+              
```

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

# TODO
- [ ] Collect images into a dataset where people have hands above their head. If they do, the image should get extra points.
