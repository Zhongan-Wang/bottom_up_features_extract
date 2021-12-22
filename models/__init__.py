from .bua import add_bottom_up_attention_config
from detectron2.config import CfgNode as CN
def add_config(args, cfg):
    cfg.MODEL.SWINT = CN()
    cfg.MODEL.SWINT.EMBED_DIM = 96
    cfg.MODEL.SWINT.OUT_FEATURES = ["stage2", "stage3", "stage4", "stage5"]
    cfg.MODEL.SWINT.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWINT.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWINT.WINDOW_SIZE = 7
    cfg.MODEL.SWINT.MLP_RATIO = 4
    cfg.MODEL.SWINT.DROP_PATH_RATE = 0.2
    cfg.MODEL.SWINT.APE = False
    cfg.MODEL.BACKBONE.FREEZE_AT = -1

    # addation
    cfg.MODEL.FPN.TOP_LEVELS = 2
    cfg.SOLVER.OPTIMIZER = "AdamW"
    if args.mode == "caffe":
        add_bottom_up_attention_config(cfg, True)
    elif args.mode == "detectron2":
        add_bottom_up_attention_config(cfg)
    else:
        raise Exception("detection model not supported: {}".format(args.model))
