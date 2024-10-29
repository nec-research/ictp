"""
       ICTP: Irreducible Cartesian Tensor Potentials
	  
  File:     trainer.py
  Authors:  Viktor Zaverkin (viktor.zaverkin@neclab.eu)
            Francesco Alesiani (francesco.alesiani@neclab.eu)
            Takashi Maruyama (takashi.maruyama@neclab.eu)
            Federico Errica (federico.errica@neclab.eu)
            Henrik Christiansen (henrik.christiansen@neclab.eu)
            Makoto Takamoto (makoto.takamoto@neclab.eu)
            Nicolas Weber (nicolas.weber@neclab.eu)
            Mathias Niepert (mathias.niepert@ki.uni-stuttgart.de)

NEC Laboratories Europe GmbH, Copyright (c) 2024, All rights reserved.  

       THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
 
       PROPRIETARY INFORMATION ---  

SOFTWARE LICENSE AGREEMENT

ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY

BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS
LICENSE AGREEMENT.  IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR
DOWNLOAD THE SOFTWARE.

This is a license agreement ("Agreement") between your academic institution
or non-profit organization or self (called "Licensee" or "You" in this
Agreement) and NEC Laboratories Europe GmbH (called "Licensor" in this
Agreement).  All rights not specifically granted to you in this Agreement
are reserved for Licensor. 

RESERVATION OF OWNERSHIP AND GRANT OF LICENSE: Licensor retains exclusive
ownership of any copy of the Software (as defined below) licensed under this
Agreement and hereby grants to Licensee a personal, non-exclusive,
non-transferable license to use the Software for noncommercial research
purposes, without the right to sublicense, pursuant to the terms and
conditions of this Agreement. NO EXPRESS OR IMPLIED LICENSES TO ANY OF
LICENSOR'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. As used in this
Agreement, the term "Software" means (i) the actual copy of all or any
portion of code for program routines made accessible to Licensee by Licensor
pursuant to this Agreement, inclusive of backups, updates, and/or merged
copies permitted hereunder or subsequently supplied by Licensor,  including
all or any file structures, programming instructions, user interfaces and
screen formats and sequences as well as any and all documentation and
instructions related to it, and (ii) all or any derivatives and/or
modifications created or made by You to any of the items specified in (i).

CONFIDENTIALITY/PUBLICATIONS: Licensee acknowledges that the Software is
proprietary to Licensor, and as such, Licensee agrees to receive all such
materials and to use the Software only in accordance with the terms of this
Agreement.  Licensee agrees to use reasonable effort to protect the Software
from unauthorized use, reproduction, distribution, or publication. All
publication materials mentioning features or use of this software must
explicitly include an acknowledgement the software was developed by NEC
Laboratories Europe GmbH.

COPYRIGHT: The Software is owned by Licensor.  

PERMITTED USES:  The Software may be used for your own noncommercial
internal research purposes. You understand and agree that Licensor is not
obligated to implement any suggestions and/or feedback you might provide
regarding the Software, but to the extent Licensor does so, you are not
entitled to any compensation related thereto.

DERIVATIVES: You may create derivatives of or make modifications to the
Software, however, You agree that all and any such derivatives and
modifications will be owned by Licensor and become a part of the Software
licensed to You under this Agreement.  You may only use such derivatives and
modifications for your own noncommercial internal research purposes, and you
may not otherwise use, distribute or copy such derivatives and modifications
in violation of this Agreement.

BACKUPS:  If Licensee is an organization, it may make that number of copies
of the Software necessary for internal noncommercial use at a single site
within its organization provided that all information appearing in or on the
original labels, including the copyright and trademark notices are copied
onto the labels of the copies.

USES NOT PERMITTED:  You may not distribute, copy or use the Software except
as explicitly permitted herein. Licensee has not been granted any trademark
license as part of this Agreement.  Neither the name of NEC Laboratories
Europe GmbH nor the names of its contributors may be used to endorse or
promote products derived from this Software without specific prior written
permission.

You may not sell, rent, lease, sublicense, lend, time-share or transfer, in
whole or in part, or provide third parties access to prior or present
versions (or any parts thereof) of the Software.

ASSIGNMENT: You may not assign this Agreement or your rights hereunder
without the prior written consent of Licensor. Any attempted assignment
without such consent shall be null and void.

TERM: The term of the license granted by this Agreement is from Licensee's
acceptance of this Agreement by downloading the Software or by using the
Software until terminated as provided below.  

The Agreement automatically terminates without notice if you fail to comply
with any provision of this Agreement.  Licensee may terminate this Agreement
by ceasing using the Software.  Upon any termination of this Agreement,
Licensee will delete any and all copies of the Software. You agree that all
provisions which operate to protect the proprietary rights of Licensor shall
remain in force should breach occur and that the obligation of
confidentiality described in this Agreement is binding in perpetuity and, as
such, survives the term of the Agreement.

FEE: Provided Licensee abides completely by the terms and conditions of this
Agreement, there is no fee due to Licensor for Licensee's use of the
Software in accordance with this Agreement.

DISCLAIMER OF WARRANTIES:  THE SOFTWARE IS PROVIDED "AS-IS" WITHOUT WARRANTY
OF ANY KIND INCLUDING ANY WARRANTIES OF PERFORMANCE OR MERCHANTABILITY OR
FITNESS FOR A PARTICULAR USE OR PURPOSE OR OF NON- INFRINGEMENT.  LICENSEE
BEARS ALL RISK RELATING TO QUALITY AND PERFORMANCE OF THE SOFTWARE AND
RELATED MATERIALS.

SUPPORT AND MAINTENANCE: No Software support or training by the Licensor is
provided as part of this Agreement.  

EXCLUSIVE REMEDY AND LIMITATION OF LIABILITY: To the maximum extent
permitted under applicable law, Licensor shall not be liable for direct,
indirect, special, incidental, or consequential damages or lost profits
related to Licensee's use of and/or inability to use the Software, even if
Licensor is advised of the possibility of such damage.

EXPORT REGULATION: Licensee agrees to comply with any and all applicable
export control laws, regulations, and/or other laws related to embargoes and
sanction programs administered by law.

SEVERABILITY: If any provision(s) of this Agreement shall be held to be
invalid, illegal, or unenforceable by a court or other tribunal of competent
jurisdiction, the validity, legality and enforceability of the remaining
provisions shall not in any way be affected or impaired thereby.

NO IMPLIED WAIVERS: No failure or delay by Licensor in enforcing any right
or remedy under this Agreement shall be construed as a waiver of any future
or other exercise of such right or remedy by Licensor.

GOVERNING LAW: This Agreement shall be construed and enforced in accordance
with the laws of Germany without reference to conflict of laws principles.
You consent to the personal jurisdiction of the courts of this country and
waive their rights to venue outside of Germany.

ENTIRE AGREEMENT AND AMENDMENTS: This Agreement constitutes the sole and
entire agreement between Licensee and Licensor as to the matter set forth
herein and supersedes any previous agreements, understandings, and
arrangements between the parties relating hereto.

       THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
"""
import os
import shutil
import time
from pathlib import Path
from typing import Dict, Union, List, Optional, Any


import numpy as np

import torch

from torch_ema import ExponentialMovingAverage

from ictp.data.data import AtomicData

from ictp.model.forward import ForwardAtomisticNetwork
from ictp.model.calculators import StructurePropertyCalculator

from ictp.training.callbacks import TrainingCallback
from ictp.training.loss_fns import LossFunction, TotalLossTracker

from ictp.utils.torch_geometric import Data
from ictp.utils.torch_geometric.dataloader import DataLoader
from ictp.utils.misc import load_object, save_object, get_default_device


def eval_metrics(calc: StructurePropertyCalculator,
                 dl: DataLoader,
                 eval_loss_fns: Dict[str, LossFunction],
                 eval_output_variables: List[str],
                 device: str = 'cuda:0',
                 early_stopping_loss_fn: Optional[LossFunction] = None) -> Dict[str, Any]:
    """Evaluates error metrics using the provided data set.

    Args:
        calc (StructurePropertyCalculator): Torch calculator for the atomistic model, see `calculators.py`.
        dl (DataLoader): Atomic data loader.
        eval_loss_fns (Dict[str, LossFunction]): Loss functions defined for evaluating model's performance.
        eval_output_variables (List[str]): Output variables: energy, forces, etc.
        device (str, optional): Available device (e.g., 'cuda:0' or 'cpu'). Defaults to 'cuda:0'.
        early_stopping_loss_fn (Optional[LossFunction], optional): Optional early stopping loss (used, e.g., 
                                                                   during training). Defaults to None.

    Returns:
        Dict[str, Any]: Dictionary with evaluation metrics provided by the loss function.
    """
    metrics = {}

    loss_trackers = {name: TotalLossTracker(loss_fn, requires_grad=False)
                     for name, loss_fn in eval_loss_fns.items()}

    if early_stopping_loss_fn is not None:
        early_stopping_loss_tracker = TotalLossTracker(early_stopping_loss_fn, requires_grad=False)
    else:
        early_stopping_loss_tracker = None

    n_structures_total = 0
    n_atoms_total = 0

    for _, batch in enumerate(dl):
        n_structures_total += len(batch.n_atoms)
        n_atoms_total += batch.n_atoms.sum().item()

        results = calc(batch.to(device), 
                       forces='forces' in eval_output_variables,
                       stress='stress' in eval_output_variables,
                       virials='virials' in eval_output_variables,
                       create_graph=True)

        if early_stopping_loss_fn is not None:
            early_stopping_loss_tracker.append_batch(results, batch)

        for loss_tracker in loss_trackers.values():
            loss_tracker.append_batch(results, batch)

    metrics['eval_losses'] = {name: loss_tracker.compute_final_result(n_structures_total, n_atoms_total).item() for name, loss_tracker in loss_trackers.items()}

    if early_stopping_loss_fn is not None:
        metrics['early_stopping'] = early_stopping_loss_tracker.compute_final_result(n_structures_total, n_atoms_total).item()

    return metrics


class Trainer:
    """Trains an atomistic model using the provided training data set. It uses early stopping to prevent 
    overfitting.

    Args:
        model (ForwardAtomisticNetwork): Atomistic model.
        lrs (float): Learning rate.
        lr_factor (float): Factor by which learning rate is reduced.
        scheduler_patience (int): Frequency for applying `lr_factor'.
        model_path (str): Path to the model.
        train_loss (LossFunction): Train loss function.
        eval_losses (Dict[str, LossFunction]): Evaluation loss function.
        early_stopping_loss (LossFunction): Early stopping loss function.
        device (Optional[str], optional): Available device (e.g., 'cuda:0' or 'cpu'). Defaults to None.
        max_epoch (int, optional): Maximal training epoch. Defaults to 1000.
        save_epoch (int, optional): Frequency for storing models for restarting. Defaults to 100.
        validate_epoch (int, optional): Frequency for evaluating models on validation data set and storing 
                                        best models, if requested.  Defaults to 1.
        train_batch_size (int, optional): Training mini-batch size. Defaults to 32.
        valid_batch_size (int, optional): Validation mini-batch size. Defaults to 100.
        callbacks (Optional[List[TrainingCallback]], optional): Callbacks to track training process. 
                                                                Defaults to None.
        opt_class (optional): Optimizer class. Defaults to torch.optim.Adam.
        amsgrad (bool, optional): If True, use amsgrad variant of adam. Defaults to False.
        max_grad_norm (float, optional): Gradient clipping value. Defaults to None.
        weight_decay (float, optional): Weight decay for the parameters of product basis.
        ema (bool, optional): It True, use exponential moving average.
        ema_decay (float, optional): Decay parameter for the exponential moving average.
    """
    def __init__(self,
                 model: ForwardAtomisticNetwork,
                 lr: float,
                 lr_factor: float,
                 scheduler_patience: int,
                 model_path: str,
                 train_loss: LossFunction,
                 eval_losses: Dict[str, LossFunction],
                 early_stopping_loss: LossFunction,
                 device: Optional[str] = None,
                 max_epoch: int = 1000,
                 save_epoch: int = 100,
                 validate_epoch: int = 1,
                 train_batch_size: int = 32,
                 valid_batch_size: int = 100,
                 callbacks: Optional[List[TrainingCallback]] = None,
                 opt_class=torch.optim.Adam,
                 amsgrad: bool = False,
                 max_grad_norm: Optional[float] = None,
                 weight_decay: float = 5e-7,
                 ema: bool = False,
                 ema_decay: float = 0.99):
        self.model = model
        self.device = device or get_default_device()
        self.calc = StructurePropertyCalculator(self.model, training=True).to(self.device)
        self.train_loss = train_loss
        self.eval_loss_fns = eval_losses
        self.early_stopping_loss_fn = early_stopping_loss
        self.train_output_variables = self.train_loss.get_output_variables()
        self.eval_output_variables = list(set(sum([l.get_output_variables() for l in self.eval_loss_fns.values()], [])))
        self.early_stopping_output_variables = self.early_stopping_loss_fn.get_output_variables()
        
        decay_interactions = {}
        no_decay_interactions = {}
        for name, param in self.model.representation.interactions.named_parameters():
            if "linear_second.weight" in name:
                decay_interactions[name] = param
            else:
                no_decay_interactions[name] = param
        
        parameter_ops = dict(
            params=[
                {
                    'name': 'embedding', 
                    'params': self.model.representation.node_embedding.parameters(), 
                    'weight_decay': 0.0,
                },
                {
                    'name': 'interactions_decay',
                    'params': list(decay_interactions.values()),
                    'weight_decay': weight_decay,
                },
                {
                    'name': 'interactions_no_decay',
                    'params': list(no_decay_interactions.values()),
                    'weight_decay': 0.0,
                },
                {
                    'name': 'products',
                    'params': self.model.representation.products.parameters(),
                    'weight_decay': weight_decay,
                },
                {
                    'name': 'readouts',
                    'params': self.model.readouts.parameters(),
                    'weight_decay': 0.0,
                }],
            lr=lr,
            amsgrad=amsgrad
            )
        
        self.optimizer = opt_class(**parameter_ops)
        self.lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer, factor=lr_factor, patience=scheduler_patience)
        
        if ema:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
        else: 
            self.ema = None
        
        self.callbacks = callbacks
        self.model_path = model_path
        self.max_epoch = max_epoch
        self.save_epoch = save_epoch
        self.validate_epoch = validate_epoch
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.max_grad_norm = max_grad_norm

        self.epoch = 0
        self.best_es_metric = np.Inf
        self.best_epoch = 0
        self.best_eval_metrics = None

        # create best and log directories to save/restore training progress
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
            
        self.log_dir = os.path.join(self.model_path, 'logs')
        self.best_dir = os.path.join(self.model_path, 'best')
        
        for dir in [self.log_dir] + [self.best_dir]:
            if not os.path.exists(dir):
                os.makedirs(dir)

    def save(self,
             path: Union[Path, str]):
        """Saves the model to the folder.

        Args:
            path (Union[Path, str]): Path to the model.
        """
        to_save = {'opt': self.optimizer.state_dict(),
                   'lr_sched': self.lr_sched.state_dict(),
                   'ema': self.ema.state_dict() if self.ema is not None else None,
                   'best_es_metric': self.best_es_metric, 
                   'best_epoch': self.best_epoch,
                   'epoch': self.epoch,
                   'best_eval_metrics': self.best_eval_metrics}

        old_folders = list(Path(path).iterdir())

        new_folder = Path(path) / f'ckpt_{self.epoch}'
        os.makedirs(new_folder)

        if self.ema is not None:
            with self.ema.average_parameters():
                self.model.save(new_folder)
        else:
            self.model.save(new_folder)
        save_object(new_folder / f'training_state.pkl', to_save)
        # delete older checkpoints after the new one has been saved
        for folder in old_folders:
            if any([p.is_dir() for p in folder.iterdir()]):
                # folder contains another folder, this shouldn't occur, we don't want to delete anything important
                raise RuntimeError(f'Model saving folder {folder} contains another folder, will not be deleted')
            else:
                shutil.rmtree(folder)

    def try_load(self,
                 path: Union[Path, str]):
        """Loads the model from the folder.

        Args:
            path (Union[Path, str]): Path to the model.
        """
        # if no checkpoint exists, just don't load
        folders = list(Path(path).iterdir())
        if len(folders) == 0:
            return  # no checkpoint exists
        if len(folders) >= 2:
            folders = [f for f in folders if f.name.startswith('ckpt_')]
            file_epoch_numbers = [int(f.name[5:]) for f in folders]
            newest_file_idx = np.argmax(np.asarray(file_epoch_numbers))
            folder = folders[newest_file_idx]
        else:
            folder = folders[0]

        self.model.load_params(folder / 'params.pkl')

        state_dict = load_object(folder / 'training_state.pkl')
        self.optimizer.load_state_dict(state_dict['opt'])
        self.lr_sched.load_state_dict(state_dict['lr_sched'])
        if self.ema is not None and state_dict['ema'] is not None:
            self.ema.load_state_dict(state_dict['ema'])
        else:
            self.ema = None
        self.best_es_metric = state_dict['best_es_metric']
        self.best_eval_metrics = state_dict['best_eval_metrics']
        self.best_epoch = state_dict['best_epoch']
        self.epoch = state_dict['epoch']

    def _train_step(self,
                    batch: Data,
                    train_loss_trackers: Dict[str, TotalLossTracker]):
        """Performs a training step using the provided batch.

        Args:
            batch (Data): Atomic data graph.
            train_loss_trackers (Dict[str, TotalLossTracker]): Dictionary of loss trackers using during training; 
                                                               see `loss_fns.py`.
        """
        self.optimizer.zero_grad(set_to_none=True)
        results = self.calc(batch, 
                            forces='forces' in self.train_output_variables,
                            stress='stress' in self.train_output_variables,
                            virials='virials' in self.train_output_variables,
                            create_graph=True)

        # compute sum of train losses for model
        tracker = TotalLossTracker(self.train_loss, requires_grad=True)
        tracker.append_batch(results, batch)
        loss = tracker.compute_final_result(n_atoms_total=batch.n_atoms.sum(), n_structures_total=batch.n_atoms.shape[0])
        loss.backward()
        
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
        
        # optimizer update
        self.optimizer.step()
        
        if self.ema is not None:
            self.ema.update()

        with torch.no_grad():
            for loss_tracker in train_loss_trackers.values():
                loss_tracker.append_batch(results, batch)

    def fit(self,
            train_ds: List[AtomicData],
            valid_ds: List[AtomicData]):
        """Trains atomistic models using provided training structures. Validation data is used for early stopping.

        Args:
            train_ds (List[AtomicData]): Training data.
            valid_ds (List[AtomicData]): Validation data.
        """
        # todo: put model in train() mode in the beginning and in eval() mode (or the mode they had before) at the end?
        # reset in case this fit() is called multiple times and try_load() doesn't find a checkpoint
        self.epoch = 0
        self.best_es_metric = np.Inf
        self.best_epoch = 0
        self.best_eval_metrics = None

        self.try_load(self.log_dir)
        
        # start timing
        start_session = time.time()

        # generate data queues for efficient training
        use_gpu = self.device.startswith('cuda')
        train_dl = DataLoader(train_ds, batch_size=self.train_batch_size, shuffle=True, drop_last=True,
                              pin_memory=use_gpu, pin_memory_device=self.device if use_gpu else '')
        valid_dl = DataLoader(valid_ds, batch_size=self.valid_batch_size, shuffle=False, drop_last=False,
                              pin_memory=use_gpu, pin_memory_device=self.device if use_gpu else '')

        for callback in self.callbacks:
            callback.before_fit(self)

        while self.epoch < self.max_epoch:
            start_epoch = time.time()
            
            self.epoch += 1

            train_loss_trackers = {name: TotalLossTracker(loss_fn, requires_grad=False) 
                                   for name, loss_fn in self.eval_loss_fns.items()}

            n_structures_total = 0
            n_atoms_total = 0

            for batch in train_dl:
                n_structures_total += len(batch.n_atoms)
                n_atoms_total += batch.n_atoms.sum().item()

                self._train_step(batch.to(self.device), train_loss_trackers)

            train_metrics = {name: loss_tracker.compute_final_result(n_structures_total, n_atoms_total).item()
                             for name, loss_tracker in train_loss_trackers.items()}

            if self.epoch % self.save_epoch == 0:
                # save progress for restoring
                self.save(self.log_dir)

            if self.epoch % self.validate_epoch == 0 or self.epoch == self.max_epoch:
                # check performance on validation step
                if self.ema is not None:
                    with self.ema.average_parameters():
                        valid_metrics = eval_metrics(calc=self.calc, dl=valid_dl, eval_loss_fns=self.eval_loss_fns,
                                                     eval_output_variables=self.eval_output_variables,
                                                     early_stopping_loss_fn=self.early_stopping_loss_fn,
                                                     device=self.device)
                else:
                    valid_metrics = eval_metrics(calc=self.calc, dl=valid_dl, eval_loss_fns=self.eval_loss_fns,
                                                 eval_output_variables=self.eval_output_variables,
                                                 early_stopping_loss_fn=self.early_stopping_loss_fn,
                                                 device=self.device)

                # update best metric based on early stopping score
                es_metric = valid_metrics['early_stopping']
                if es_metric < self.best_es_metric:
                    self.best_es_metric = es_metric
                    self.best_eval_metrics = valid_metrics['eval_losses']
                    self.best_epoch = self.epoch
                    self.save(self.best_dir)

                self.lr_sched.step(metrics=valid_metrics['early_stopping'])
                
                end_epoch = time.time()

                for callback in self.callbacks:
                    callback.after_epoch(self, train_metrics, valid_metrics['eval_losses'], end_epoch - start_epoch)

        end_session = time.time()

        for callback in self.callbacks:
            callback.after_fit(self, end_session - start_session)
