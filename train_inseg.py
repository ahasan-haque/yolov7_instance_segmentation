import os
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch
)
from detectron2.data import build_detection_train_loader
from detectron2.modeling import build_model

from yolov7.config import add_yolo_config
from yolov7.data.dataset_mapper import MyDatasetMapper, MyDatasetMapper2
from yolov7.evaluation.coco_evaluation import COCOMaskEvaluator
from detectron2.data.datasets.coco import register_coco_instances

"""
Script used for training instance segmentation, i.e. SparseInst.
"""

class Trainer(DefaultTrainer):

    custom_mapper = None

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOMaskEvaluator(dataset_name, output_dir=output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        cls.custom_mapper = MyDatasetMapper(cfg, True)
        return build_detection_train_loader(cfg, mapper=cls.custom_mapper)

    @classmethod
    def build_model(cls, cfg):
        model = build_model(cfg)
        return model

def register_custom_datasets():
    # facemask dataset
    DATASET_ROOT = "./datasets/test_data"
    ANN_ROOT = os.path.join(DATASET_ROOT, "annotations")
    TRAIN_PATH = os.path.join(DATASET_ROOT, "train")
    VAL_PATH = os.path.join(DATASET_ROOT, "val")
    TRAIN_JSON = os.path.join(ANN_ROOT, "train.json")
    VAL_JSON = os.path.join(ANN_ROOT, "val.json")
    register_coco_instances("ahsan_train", {}, TRAIN_JSON, TRAIN_PATH)
    register_coco_instances("ahsan_val", {}, VAL_JSON, VAL_PATH)


def setup(args):
    cfg = get_cfg()
    add_yolo_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    register_custom_datasets()
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
