import src.utils as utils
import src.config as config
import src.cyclegan as cyclegan

from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description='CycleGAN PyTorch')
    parser.add_argument('--epochs', type=int, default=160)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=.0002)
    parser.add_argument('--training', type=bool, default=False)
    parser.add_argument('--testing', type=bool, default=False)
    parser.add_argument('--results_dir', type=str, default='/home/root/results/images')
    parser.add_argument('--stats_dir', type=str, default='/home/root/results/stats')
    parser.add_argument('--photo_path', type=str, default='/home/root/images/photo_jpg')
    parser.add_argument('--monet_path', type=str, default='/home/root/images/monet_jpg')
    parser.add_argument('--checkpoint_dir', type=str, default='/home/root/checkpoints/cyclegan')
    parser.add_argument('--checkpoint_name', type=str, default='latest.ckpt')
    parser.add_argument('--gen_model', type=str, default='unet')
    parser.add_argument('--dis_model', type=str, default='simple')
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    cnfg = config.NetConfig(args.gen_model, args.dis_model)
    if args.training:
        print("Training")
        model = cyclegan.CycleGAN(args, cnfg)
        model.train(args, cnfg)
        utils.save_loss_history(model.gen_loss_photo_history, model.gen_loss_monet_history, \
            model.id_loss_photo_history, model.id_loss_monet_history, model.cyc_loss_history, \
            model.photo_dis_loss_history, model.monet_dis_loss_history, args.checkpoint_name, args.stats_dir)
    if args.testing:
        print("Testing")
        model = cyclegan.CycleGAN(args, cnfg)
        model.test(args)

if __name__ == '__main__':
    main()