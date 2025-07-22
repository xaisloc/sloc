import os, sys
import logging
import torch
import yaml
import models
import dataset
import sloc
import argparse

def save_saliency(results_path, obj, model_name, variant, image_name):
    path = os.path.join(results_path, model_name, "saliency", variant, image_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(obj, path)


def create_explanations(me, algo, all_images, results_path, skip_img_error=True):

    for itr, info in enumerate(all_images):    
        
        image_name = info.name
        image_path = info.path 
        target = info.target

        pidx = image_name.find(".")
        if pidx > 0:
            image_name = image_name[0:pidx]

        try:
            img, inp = me.get_image_ext(image_path)
        except:
            logging.exception("Failed getting image")
            if skip_img_error:
                logging.info("Skipping")
                continue

        logits = me.model(inp).cpu()
        topidx = int(torch.argmax(logits))        
        logging.info(f"creating sal {itr} {image_path} {image_name} {topidx} {info.desc}")
       
        sal_dict = algo(me, inp, topidx)

        logging.info("done, saving")
        for variant, sal  in sal_dict.items():
            if variant.startswith("_"):
                continue
            save_saliency(results_path, sal, me.arch, variant, image_name)
        
def validate_cfg(cfg, path=[]):
    missing = []
    for name, value in cfg.items():
        next_path = path + [name]
        if type(value) == dict:
            missing += validate_cfg(value, next_path)
        elif value is None:
            missing.append(".".join(next_path))
    if not path:
        print("Missing config:")
        for x in missing:
            print(f' - {x}')
    return missing

### creators 

def get_sloc_explanation_creator(**kwargs):
    return sloc.AutoProbSlocExplanationCreator(**kwargs)


########################



def parse_arguments():
    parser = argparse.ArgumentParser(description="Process a config file and optionally specify a creator.")
    
    # Mandatory positional parameter
    parser.add_argument(
        "yaml_path",
        type=str,
        help="Path to the configuration file"
    )
    
    # Optional parameter with default value
    parser.add_argument(
        "--creator",
        type=str,
        default=None,
        help="Name of the creator (default: None)"
    )
    
    return parser.parse_args()


if __name__ == '__main__':
        
    logging.basicConfig(format='[%(asctime)-15s  %(filename)s:%(lineno)d - %(process)d] %(message)s', level=logging.DEBUG)
    logging.info("start")  

    args = parse_arguments()    
    with open(args.yaml_path, "rt") as yfile:
        cfg = yaml.safe_load(yfile)
        validate_cfg(cfg)

    if args.creator is not None:
        cfg['creator_name'] = args.creator
        
    isrc = dataset.ImagenetSource(base_path=cfg['images']['base_path'],
                                  image_dir_ptrn=cfg['images']['dir_ptrn'],
                                  selection_filename=cfg['images']['selection'])    
    all_images = list(isrc.get_all_images().values())
    logging.info(f"found {len(all_images)} images")

    me = models.ModelEnv(cfg['model_name'])

    creator_name = cfg['creator_name']
    creator_func_name = f'get_{creator_name}_explanation_creator'
    creator_func =  globals().get(creator_func_name)
    if creator_func is None:
        raise Exception(f"Missing creator: {creator_name} / {creator_func_name}")
    creator_args = cfg['creator_args'].get(creator_name, {})
    creator = creator_func(**creator_args)
    create_explanations(me, creator, all_images, cfg['results_path'])