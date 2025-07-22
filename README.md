# Soft Local Completeness: Rethinking Completeness in Model Explainability

## Setting up environment:

See `batch/cenv.sh` (SLURM script to setup conda env)

## Creating Explanations:

```bash
python src/create_explanations.py <yaml-config-path>
```

See `config/config.yaml` example (fill missing values)

```yaml
images:
  base_path: 
  dir_ptrn: "*"     # images would be looked for at <base_path>/<dir_ptrn>
  selection:        # filename at <base_path> containing list of image filenames (one per line)
model_name:         # model_name: for example resnet50, vgg16, vit_base_patch16_224, etc...
creator_name: sloc      
creator_args:
  sloc:              # sloc parameters
results_path:       # results (explanations) base path. would be saved at <results_path>/<model_name>/saliency/<explanation_name>/<image_name>
```

### YAML Configuration for Images

If the images are located at `<images_base_path>/*/*.JPEG`, the YAML configuration should be structured as follows:

- **`images/base_path`**: Specifies the `<images_base_path>`, which is the base directory where the images are stored.
- **`images/selection`**: Specifies the filename within `<images_base_path>` that contains the list of images (file names) for which explanations would be created


The **images list** is a simple text file containing the names of the images (not their full paths). For example, the content of the selection file might look like this:

```
ILSVRC2012_val_00018871.JPEG
ILSVRC2012_val_00042983.JPEG
ILSVRC2012_val_00040154.JPEG
```

(A list of all images can be generated with `find . -name '*.JPEG' | xargs -l basename `)


