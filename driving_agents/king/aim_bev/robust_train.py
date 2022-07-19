import argparse
import sys
import json
import os
import time
from tqdm import tqdm
import webdataset as wds
from pathlib import Path
import signal
import math
from datetime import datetime
import random
import getpass
import subprocess
sys.path.append(os.getcwd())  # make imports from king root work (e.g. driving_agents/*)

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
torch.multiprocessing.set_sharing_strategy('file_system')

from driving_agents.king.aim_bev.data import CARLA_Data
from driving_agents.king.aim_bev.training_utils import augment_all, augment_names
from driving_agents.king.common.robust_training_engine import RobustTrainingEngine
from driving_agents.king.aim_bev.model import AimBev
from driving_agents.king.aim_bev.aim_bev_agent import AimBEVAgent
from driving_agents.king.expert.expert_agent import BEVDataAgent


USER = getpass.getuser()


class Trainer(object):
    """
    Engine that orthestrates robust training, including interfacing with KING
    for data collection and evaluation.
    """
    def __init__(self, conf_log, model, optimizer, dataloader_train=None, king_dataloader_train=None, cur_epoch=0, cur_iter=0):
        self.model = model
        self.optimizer = optimizer
        self.reg_dataloader_train = dataloader_train
        self.king_dataloader_train = king_dataloader_train
        self.king_dataset_train = None
        self.cur_epoch = cur_epoch
        self.cur_iter = cur_iter
        self.bestval_epoch = cur_epoch
        self.train_loss = []
        self.val_loss = []
        self.bestval = -1e5
        self.conf_log = conf_log

    def train(self):
        self.model.train()
        loss_epoch = 0.
        wp_epoch = 0.
        num_batches = 0

        if self.king_dataloader_train is not None:
            data_loader = self.king_dataloader_train
        else:
            data_loader = range(len(self.king_dataset_train) // args.robust_train_batch_size)

        for king_data in tqdm(data_loader):
            # efficiently zero gradients
            for p in self.model.parameters():
                p.grad = None

            # map regular data onto gpu
            if self.reg_dataloader_train:
                # if we went through the dataset, we reset
                try:
                    data_reg = self.reg_dataloader_train_iter.next()
                except StopIteration:
                    self.reg_dataloader_train_iter = iter(self.reg_dataloader_train)
                    data_reg = self.reg_dataloader_train_iter.next()

                # map regular data onto gpu
                bev_in_reg = [data_reg['bev']]
                target_point_reg = torch.stack(data_reg['target_point'], dim=1).to(args.device, dtype=torch.float32)
                light_hazard_reg = torch.stack([data_reg['light']], dim=1).to(args.device, dtype=torch.float32)
                gt_waypoints_reg = [torch.stack(data_reg['waypoints'][i], dim=1).to(args.device, dtype=torch.float32) for i in range(args.pred_len)]
                gt_waypoints_reg = torch.stack(gt_waypoints_reg, dim=1).to(args.device, dtype=torch.float32)

            # map king data onto gpu
            if self.king_dataloader_train:
                bev_in_king = [king_data['bev']]
                target_point_king = torch.stack(king_data['target_point'], dim=1).to(args.device, dtype=torch.float32)
                light_hazard_king = torch.stack(king_data['light'], dim=1).to(args.device, dtype=torch.float32)
                gt_waypoints_king = [torch.stack(king_data['waypoints'][i], dim=1).to(args.device, dtype=torch.float32) for i in range(args.pred_len)]
                gt_waypoints_king = torch.stack(gt_waypoints_king, dim=1).to(args.device, dtype=torch.float32)

            # concat regular and king data to form mixed batches
            if self.reg_dataloader_train and self.king_dataloader_train:
                concat_bev_in = []
                for bev_in, king_bev in zip(bev_in_reg, bev_in_king):
                    concat_bev_in.append(
                        torch.cat([bev_in, king_bev])
                    )
                bev_in = concat_bev_in
                target_point = torch.cat([target_point_reg, target_point_king])
                light_hazard = torch.cat([light_hazard_reg, light_hazard_king])
                gt_waypoints = torch.cat([gt_waypoints_reg, gt_waypoints_king])
            elif self.reg_dataloader_train and not self.king_dataloader_train:
                bev_in = bev_in_reg
                target_point = target_point_reg
                light_hazard = light_hazard_reg
                gt_waypoints = gt_waypoints_reg
            elif self.king_dataloader_train and not self.reg_dataloader_train:
                bev_in = bev_in_king
                target_point = target_point_king
                light_hazard = light_hazard_king
                gt_waypoints = gt_waypoints_king

            bev = []

            for i in range(args.seq_len):
                bev.append(bev_in[i].to(args.device, dtype=torch.float32))
            # inference
            encoding = [self.model.image_encoder(bev)]

            pred_wp = self.model(encoding, target_point, light_hazard=light_hazard)

            loss_wp = F.l1_loss(pred_wp, gt_waypoints, reduction='none')
            loss = loss_wp

            loss.mean().backward()
            wp_epoch += loss.mean().item()

            num_batches += 1
            self.optimizer.step()

            self.cur_iter += 1

        loss_epoch = wp_epoch / num_batches

        self.train_loss.append(loss_epoch)
        self.cur_epoch += 1

    def save_best(self):
        save_best = False
        if self.val_loss[-1] < self.bestval:
            self.bestval = self.val_loss[-1]
            self.bestval_epoch = self.cur_epoch
            save_best = True

        # Create a dictionary of all data to save
        log_table = {
            'epoch': self.cur_epoch,
            'iter': self.cur_iter,
            'bestval': self.bestval,
            'bestval_epoch': self.bestval_epoch,
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
        }

        if save_best:
            torch.save(self.model.state_dict(), os.path.join(args.logdir, 'best_model.pth'))
            torch.save(self.optimizer.state_dict(), os.path.join(args.logdir, 'best_optim.pth'))
            print('====== Overwrote best model ======>')
            with open(os.path.join(args.logdir, 'best.log'), 'w') as f:
                f.write(json.dumps(log_table))

        torch.save(self.model.state_dict(), os.path.join(args.logdir, f'model_epoch{self.cur_epoch}.pth'))
        torch.save(self.optimizer.state_dict(), os.path.join(args.logdir, f'optim_epoch{self.cur_epoch}.pth'))
        print('====== Save model by epoch ======>')
        with open(os.path.join(args.logdir, f'epoch{self.cur_epoch}.log'), 'w') as f:
            f.write(json.dumps(log_table))

    def save_recent(self):
        # Create a dictionary of all data to save
        log_table = {
            'epoch': self.cur_epoch,
            'iter': self.cur_iter,
            'bestval': self.bestval,
            'bestval_epoch': self.bestval_epoch,
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
        }

        torch.save(self.model.state_dict(), os.path.join(args.logdir, 'model.pth'))
        torch.save(self.optimizer.state_dict(), os.path.join(args.logdir, 'recent_optim.pth'))

        # Log other data corresponding to the recent model
        with open(os.path.join(args.logdir, 'recent.log'), 'w') as f:
            f.write(json.dumps(log_table))

        print('====== Saved recent model ======>')

    def load_checkpoint(self):
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print ('Total trainable parameters: ', params)

        # Create logdir
        if not os.path.isfile(os.path.join(args.logdir, 'recent.log')):
            # Load checkpoint
            print('Loading checkpoint from \'driving_agents/king/aim_bev/model_checkpoints/regular\'')
            self.model.load_state_dict(torch.load(os.path.join(
                f'driving_agents/king/aim_bev/model_checkpoints/regular',
                'model.pth'
            )))
        elif os.path.isfile(os.path.join(args.logdir, 'recent.log')):
            if self.cur_epoch == 0:
                print('Loading checkpoint from \'driving_agents/king/aim_bev/model_checkpoints/regular\'')
                self.model.load_state_dict(torch.load(os.path.join(
                    f'driving_agents/king/aim_bev/model_checkpoints/regular',
                    'model.pth'
                )))
            else:
                print('Loading checkpoint from ' + args.logdir)
                with open(os.path.join(args.logdir, 'recent.log'), 'r') as f:
                    log_table = json.load(f)
                # Load variables
                self.cur_epoch = log_table['epoch']
                if 'iter' in log_table: self.cur_iter = log_table['iter']
                self.bestval = log_table['bestval']
                self.train_loss = log_table['train_loss']
                self.val_loss = log_table['val_loss']
                self.model.load_state_dict(torch.load(os.path.join(args.logdir, 'model.pth')))
                self.optimizer.load_state_dict(torch.load(os.path.join(args.logdir, 'recent_optim.pth')))

    def run_king_eval(self, args):
        print(
            datetime.now().strftime("%m/%d/%Y - %H:%M:%S"),
            "|",
            f"Epoch {self.cur_epoch} - Evaluating on KING Scenarios..."
        )
        self.model.eval()
        args.scenario_summary_dir = args.scenario_summary_dir_eval
        args.routes_file = args.routes_file_eval

        # Launch Carla
        modified_env = os.environ.copy()
        modified_env["DISPLAY"] = ""
        carla_root = os.environ["CARLA_ROOT"]
        print("Relauching CARLA...")
        carla_process = subprocess.Popen(
            [f"{carla_root}/CarlaUE4.sh", f"-port={args.port}", "-opengl"],
            env=modified_env,
            preexec_fn=self._switch_to_carla_user
        )
        time.sleep(5)

        # Build Agent
        ego_agent = AimBEVAgent(
            args,
            device=args.device,
            path_to_conf_file=args.ego_agent_ckpt
        )
        if os.path.isfile(os.path.join(args.logdir, 'recent.log')):
            ego_agent.net.load_state_dict(torch.load(os.path.join(args.logdir, 'model.pth')))
        else:
            ego_agent.net.load_state_dict(torch.load(os.path.join(
                    f'driving_agents/king/aim_bev/model_checkpoints/regular',
                    'model.pth')
            ))

        # Build interface to proxy simulator for robust training
        king_hybrid_evaluator = RobustTrainingEngine(
            args,
            ego_agent=ego_agent,
        )
        collision_rate = king_hybrid_evaluator.run_king_eval(epoch=self.cur_epoch)

        print(
            datetime.now().strftime("%m/%d/%Y - %H:%M:%S"),
            "|",
            f"Epoch {self.cur_epoch} - Collision Rate: {collision_rate}"
        )

        # Clean up carla
        print("Cleaning up CARLA Server...")
        os.killpg(os.getpgid(carla_process.pid), signal.SIGKILL)
        time.sleep(1)

        self.val_loss.append(collision_rate)

        # Store results in json files
        rob_results_path = os.path.join(args.logdir, "robustness_results.json")
        if os.path.exists(rob_results_path) and self.cur_epoch == 0:
            rob_results_dict = {} # overwrite existing file if starting from scratch
        elif os.path.exists(rob_results_path) and self.cur_epoch != 0:
            with open(rob_results_path, "r") as f:
                rob_results_dict = json.load(f)
        else:
            rob_results_dict = {}
        rob_results_dict.update({f"Epoch_{self.cur_epoch}": {"CR": collision_rate}})

        with open(rob_results_path, "w") as f:
                json.dump(rob_results_dict, f, indent=2)

        # Cleanup
        del ego_agent
        del king_hybrid_evaluator

        torch.cuda.empty_cache()

    def collect_king_data(self, args):
        print(
            datetime.now().strftime("%m/%d/%Y - %H:%M:%S"),
            "|",
            f"Collecting KING data..."
        )
        self.model.eval()
        args.scenario_summary_dir = args.scenario_summary_dir_train
        args.routes_file = args.routes_file_train

        # LAUNCH CARLA
        modified_env = os.environ.copy()
        modified_env["DISPLAY"] = ""
        carla_root = os.environ["CARLA_ROOT"]
        print("Relauching CARLA...")
        carla_process = subprocess.Popen(
            [f"{carla_root}/CarlaUE4.sh", f"-port={args.port}", "-opengl"],
            env=modified_env,
            preexec_fn=self._switch_to_carla_user
        )
        time.sleep(5)

        save_path = os.path.join(args.logdir, f"king_data", f"epoch_{self.cur_epoch}")

        ego_expert = BEVDataAgent(
            args,
            device=args.device,
            save_path=save_path
        )

        king_hybrid_evaluator = RobustTrainingEngine(
            args,
            ego_expert=ego_expert,
        )

        king_hybrid_evaluator.collect_data()

        # CLEAN UP CARLA
        print("Cleaning up CARLA Server...")
        os.killpg(os.getpgid(carla_process.pid), signal.SIGKILL)
        time.sleep(1)

        # CLEANUP
        del king_hybrid_evaluator
        torch.cuda.empty_cache()

    def _switch_to_carla_user(self):
        """
        For use with docker, where we usually are root. Carla does not like that
        so we temporily switch to a dedicated carla user.
        """
        # this make sure the process is independent of parent
        os.setsid()

        # no need to do anything else if we are not root
        if not getpass.getuser() == "root":
            return

        # this switches the user - set desired UID here
        os.setgid(1000)
        os.setuid(1000)


def get_dataloaders(args):
    batch_size_reg = math.floor(args.robust_train_batch_size * (1 - args.mixed_batch_ratio))
    batch_size_king = math.floor(args.robust_train_batch_size * args.mixed_batch_ratio)

    # regular data
    totalDir = 0
    for base, dirs, files in os.walk(args.train_dataset_path):
        totalDir += len(files)-1
    Shard_path = f"{args.train_dataset_path}/shard-" + "{000000.." + f"{totalDir:06}" + "}.tar"
    shardlist =  wds.PytorchShardList(Shard_path, shuffle=True)
    dataset = (
        wds.WebDataset(shardlist)
        .decode(wds.imagehandler("torchrgb"))
        .map(augment_names)
        .map(augment_all)
        .shuffle(500)
    )
    if batch_size_reg != 0:
        dataloader_train = wds.WebLoader(dataset, batch_size=batch_size_reg, num_workers=args.num_workers)
    else:
        dataloader_train = None

    # KING data
    king_train_subdirs = []
    king_train_root_dir = args.train_king_dataset_path
    for subdir in os.listdir(king_train_root_dir):
        king_train_subdirs.append(os.path.join(king_train_root_dir, subdir))

    king_dataset_train = CARLA_Data(
        args,
        king_train_subdirs,
        aug_max_rotation=args.aug_max_rotation,
        create_webdataset=True
    )
    if batch_size_king != 0:
        king_dataloader_train = DataLoader(king_dataset_train, batch_size=batch_size_king, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    else:
        king_dataloader_train = None

    return dataloader_train, king_dataloader_train, king_dataset_train


def main(args):
    model = AimBev(args, 'cuda', args.pred_len).to(args.device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    trainer = Trainer(
        vars(args),
        model,
        optimizer,
    )
    trainer.load_checkpoint()

    # Log args
    with open(os.path.join(args.logdir, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    trainer.run_king_eval(args)
    trainer.collect_king_data(args)

    dataloader_train, king_dataloader_train, king_dataset_train = get_dataloaders(args)
    if dataloader_train is not None:
        trainer.reg_dataloader_train = dataloader_train
        trainer.reg_dataloader_train_iter = iter(dataloader_train)
    trainer.king_dataset_train = king_dataset_train
    trainer.king_dataloader_train = king_dataloader_train

    for epoch in range(trainer.cur_epoch, args.epochs):
        print('Training')
        trainer.train()
        trainer.save_recent()
        if epoch % args.val_every == 0 and not epoch == 0:
            trainer.run_king_eval(args)
            trainer.save_best()


if __name__ == "__main__":
    # Robust train arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--id',
        type=str,
        default='robust_aim_bev',
        help='Unique experiment identifier.'
    )
    parser.add_argument(
        '--train_dataset_path',
        type=str,
        help='Root directory of regular data.'
    )
    parser.add_argument(
        '--king_data_fps',
        type=int,
        default=2,
        help='Temporal subsampling for data collection.'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Compute target.'
    )
    parser.add_argument(
        '--num_workers',
        type=int, default=2,
        help='Number of dataloader workers.'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=4,
        help='Number of train epochs.'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Learning rate.'
    )
    parser.add_argument(
        '--val_every',
        type=int,
        default=3,
        help='Evaluation frequency (epochs).'
    )
    parser.add_argument(
        '--robust_train_batch_size',
        type=int,
        default=512,
        help='Batch size'
    )
    parser.add_argument(
        '--mixed_batch_ratio',
        type=float,
        default=0.4,
        help='Proportion of regular/KING data. Higher number means more KING data.'
    )
    parser.add_argument(
        '--target_im_size',
        default=(192,192),
        help='Input crop size'
    )
    parser.add_argument(
        '--seq_len',
        type=int,
        default=1,
        help='Input sequence length (factor of 10).'
    )
    parser.add_argument(
        '--pred_len',
        type=int,
        default=4,
        help='number of timesteps to predict.'
    )
    parser.add_argument(
        '--do_augmentation',
        type=int,
        default=1,
        help='Wether or not to apply augmentation.'
    )
    parser.add_argument(
        '--aug_max_rotation',
        type=int,
        default=20,
        help='Max rotation angle [degree] for augmentation. 0.0 equals to no agmentation.'
    )
    parser.add_argument(
        '--logdir',
        type=str,
        default='fine_tuning_results',
        help='Directory to log data to.'
    )

    # KING arguments, as expected by proxy simulator
    parser.add_argument(
        "--seed",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The number of parallel simulations."
    )
    parser.add_argument(
        "--num_agents",
        type=int,
        default=4,
        help="The number of other agents in the simulation."
    )
    parser.add_argument(
        "--sim_tickrate",
        type=int,
        default=4,
        help="Inverse of the delta_t between subsequent timesteps of the simulation."
    )
    parser.add_argument(
        "--sim_horizon",
        type=int,
        default=80,
        help="The number of timesteps to run the simulation for."
    )
    parser.add_argument(
        "--renderer_class",
        type=str,
        default='STN',
        choices=['STN', "CARLA"],
    )
    parser.add_argument(
        "--port",
        type=int,
        default=2000,
        help="Carla port."
    )
    parser.add_argument(
        "--scenario_summary_dir_train",
        type=str,
        default="driving_agents/king/aim_bev/king_scenarios/robust_training/train",
        help="The directory containing the scenario records and results files for each of the optimized scenarios/routes.",
    )
    parser.add_argument(
        "--scenario_summary_dir_eval",
        type=str,
        default="driving_agents/king/aim_bev/king_scenarios/robust_training/eval",
        help="The directory containing the scenario records and results files for each of the optimized scenarios/routes."
    )
    parser.add_argument(
        "--routes_file_train",
        type=str,
        default="leaderboard/data/routes/train_robustness.xml",
        help="Path to the .xml file describing the routes to be driven."
    )
    parser.add_argument(
        "--routes_file_eval",
        type=str,
        default="leaderboard/data/routes/eval_robustness.xml",
        help="Path to the .xml file describing the routes to be driven."
    )
    parser.add_argument(
        "--routes_file_adv",
        type=str,
        default="leaderboard/data/routes/adv_all.xml",
        help="Path to the .xml file describing the routes to be driven."
    )
    parser.add_argument(
        "--init_root",
        type=str,
        default="driving_agents/king/aim_bev/king_initializations/initializations_all",
        help="Path to the scenario initalization files for this agent."
    )
    parser.add_argument(
        "--ego_agent",
        type=str,
        default='aim-bev',
        choices=['aim-bev', 'transfuser'],
        help="The agent under test."
    )
    parser.add_argument(
        "--ego_agent_ckpt",
        type=str,
        default="driving_agents/king/aim_bev/model_checkpoints/regular",
        help="Path to the model checkpoint for the agent under test."
    )
    parser.add_argument(
        "--gradient_clip",
        type=float,
        default=0.0
    )

    args = parser.parse_args()

    args.train_king_dataset_path = os.path.join(args.logdir, args.id, "king_data")
    args.eval_king_dataset_path = os.path.join(args.logdir, args.id, "king_data")

    # reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    args.logdir = os.path.join(args.logdir, args.id)
    Path(f'{args.logdir}/results').mkdir(parents=True, exist_ok=True)
    Path(f'{args.logdir}/king_data/').mkdir(parents=True, exist_ok=True)

    os.environ["aug_max_rotation"] = str(args.aug_max_rotation)

    main(args)