import argparse
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description='Export yolo11 model trained on coco.')
    parser.add_argument('--config', default='./config/plugin_config.yml', help='Plugin config file used in end2end')
    parser.add_argument('--input_h', type=int, default=720, help='Model\'s image input height')
    parser.add_argument('--input_w', type=int, default=1280, help='Model\'s image input width')
    parser.add_argument('--net_input_h', type=int, default=384, help='Model\'s network input height, make sure it\'s divided by 32')
    parser.add_argument('--net_input_w', type=int, default=640, help='Model\'s network input width, make sure it\'s divided by 32')
    parser.add_argument('--mean', type=list, default=[0.0, 0.0, 0.0], help='Preprocess mean configuration')
    parser.add_argument('--norm_inv', type=list, default=[0.003921569, 0.003921569, 0.003921569], help='Preprocess norm configuration')
    return parser.parse_args()

class Crop:
    def __init__(self, l, t, w, h):
        self.l = l
        self.t = t
        self.w = w
        self.h = h
    
    def get(self):
        return [self.l, self.t, self.w, self.h]

if __name__ == "__main__":
    args = parse_args()
    assert(args.net_input_h % 32 == 0)
    assert(args.net_input_w % 32 == 0)
    src_w = args.input_w
    src_h = args.input_h
    dst_w = args.net_input_w
    dst_h = args.net_input_h

    with open(args.config, "r") as file:
        plugin_config = yaml.load(file, Loader=yaml.SafeLoader)
    
    src_crop = Crop(0, 0, src_w, src_h)
    dst_crop = Crop(0, 0, dst_w, dst_h)
    src_ratio = src_w / src_h
    dst_ratio = dst_w / dst_h
    if (src_ratio > dst_ratio):
        dst_crop.w = dst_w
        dst_crop.h = int(dst_w / src_ratio)
        dst_crop.l = 0
        dst_crop.t = int((dst_h - dst_crop.h) / 2)
    else:
        # Use dst's height as base
        dst_crop.h = dst_h
        dst_crop.w = int(dst_h * src_ratio)
        dst_crop.t = 0
        dst_crop.l = int((dst_w - dst_crop.w) / 2)
    scale_inv_x = (src_crop.w) / dst_crop.w
    scale_inv_y = (src_crop.h) / dst_crop.h
    plugin_config['precess_plugin']['src_crop'] = src_crop.get()
    plugin_config['precess_plugin']['dst_crop'] = dst_crop.get()
    plugin_config['precess_plugin']['src_size'] = [src_h, src_w]
    plugin_config['precess_plugin']['dst_size'] = [dst_h, dst_w]
    plugin_config['precess_plugin']['scale_inv'] = [scale_inv_x, scale_inv_y]
    plugin_config['precess_plugin']['mean'] = args.mean
    plugin_config['precess_plugin']['norm_inv'] = args.norm_inv
    plugin_config['precess_plugin']['src_step'] = [src_crop.w]
    plugin_config['precess_plugin']['dst_step'] = [dst_crop.w]
    with open(args.config, "w") as file:
        yaml.dump(plugin_config, file, sort_keys=False)