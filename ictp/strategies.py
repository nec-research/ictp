"""
       ICTP: Irreducible Cartesian Tensor Potentials
	  
  File:     strategies.py
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
import time
from pathlib import Path
from typing import Dict, Optional, Union, Any

import numpy as np

import torch

import ase

from ictp.data.data import AtomicStructures, AtomicTypeConverter

from ictp.model.calculators import StructurePropertyCalculator
from ictp.model.forward import ForwardAtomisticNetwork, build_model, load_model_from_folder

from ictp.training.callbacks import FileLoggingCallback
from ictp.training.loss_fns import config_to_loss
from ictp.training.trainer import Trainer, eval_metrics

from ictp.utils.config import update_config
from ictp.utils.misc import save_object
from ictp.utils.torch_geometric import DataLoader


class TrainingStrategy:
    """Strategy for training interatomic potentials.

    Args:
        config (Optional[Dict[str, Any]], optional): Configuration file with parameters listed in 'utils/config.py'. 
                                                     The default parameters of 'utils/config.py' will be updated by 
                                                     those provided in 'config'. Defaults to None.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Update config containing all parameters (including the model, training, fine-tuning, and evaluation)
        # We store all parameters in one config for simplicity. In the log and best folders the last training
        # config is stored and used when loading a model for inference.
        self.config = update_config(config.copy())

    def run(self,
            train_structures: AtomicStructures,
            valid_structures: AtomicStructures,
            folder: Union[str, Path],
            model_seed: Optional[int] = None) -> ForwardAtomisticNetwork:
        """Runs training using provided training and validation structures.

        Args:
            train_structures (AtomicStructures): Training structures.
            valid_structures (AtomicStructures): Validation structures.
            folder (Union[str, Path]): Folder where the trained model is stored.
            model_seed (int, optional): Random seed to initialize the atomistic model. Defaults to None.

        Returns:
            ForwardAtomisticNetwork: Trained atomistic model.
        """
        # define atomic type converter
        atomic_type_converter = AtomicTypeConverter.from_type_list(self.config['atomic_types'])
        
        # convert atomic numbers to type names
        train_structures = train_structures.to_type_names(atomic_type_converter, check=True)
        valid_structures = valid_structures.to_type_names(atomic_type_converter, check=True)
        
        # store the number of training and validation structures in config
        self.config['n_train'] = len(train_structures)
        self.config['n_valid'] = len(valid_structures)
        
        # build atomic data sets
        train_ds = train_structures.to_data(r_cutoff=self.config['r_cutoff'], 
                                            n_species=atomic_type_converter.get_n_type_names())
        valid_ds = valid_structures.to_data(r_cutoff=self.config['r_cutoff'], 
                                            n_species=atomic_type_converter.get_n_type_names())
        
        # update model seed if provided (can be used to re-run a calculation with a different seed) and build the model
        if model_seed is not None:
            self.config['model_seed'] = model_seed
        model = build_model(train_structures, n_species=atomic_type_converter.get_n_type_names(), **self.config)
            
        # define losses from config
        train_loss = config_to_loss(self.config['train_loss'])
        eval_losses = {l['type']: config_to_loss(l) for l in self.config['eval_losses']}
        early_stopping_loss = config_to_loss(self.config['early_stopping_loss'])
        
        # define callbacks to track training
        callbacks = [FileLoggingCallback()]
        
        # define model training
        trainer = Trainer(model, model_path=folder, callbacks=callbacks, 
                          lr=self.config['lr'], lr_factor=self.config['lr_factor'], scheduler_patience=self.config['scheduler_patience'], 
                          max_epoch=self.config['max_epoch'], save_epoch=self.config['save_epoch'], validate_epoch=self.config['valid_epoch'],
                          train_batch_size=min(self.config['train_batch_size'], len(train_structures)),
                          valid_batch_size=min(self.config['eval_batch_size'], len(valid_structures)),
                          train_loss=train_loss, eval_losses=eval_losses, early_stopping_loss=early_stopping_loss,
                          max_grad_norm=self.config['max_grad_norm'], device=self.config['device'], 
                          amsgrad=self.config['amsgrad'], weight_decay=self.config['weight_decay'],
                          ema=self.config['ema'], ema_decay=self.config['ema_decay'])
        
        # train the model
        trainer.fit(train_ds=train_ds, valid_ds=valid_ds)
        
        # return best models and move them to device
        model = load_model_from_folder(folder, key='best')
        model = model.to(self.config['device'])
        
        return model


class EvaluationStrategy:
    """Strategy for evaluating the performance of interatomic potentials.

    Args:
        config (Optional[Dict[str, Any]], optional): Configuration file with parameters listed in 'utils/config.py'. 
                                                     The default parameters of 'utils/config.py' will be updated by those 
                                                     provided in 'config'. Defaults to None.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Update config containing all parameters (including the model, training, fine-tuning, and evaluation)
        # We store all parameters in one config for simplicity. In the log and best folders the last training
        # config is stored and used when loading a model for inference.
        self.config = update_config(config.copy())

    def run(self,
            model: ForwardAtomisticNetwork,
            test_structures: AtomicStructures,
            folder: Union[str, Path]) -> Dict[str, Any]:
        """Evaluates models using the provided test data set.

        Args:
            model (ForwardAtomisticNetwork): Atomistic model.
            test_structures (AtomicStructures): Test structures.
            folder (Union[str, Path]): Folder where the evaluation results are stored.

        Returns:
            Dict[str, Any]: Results dictionary containing test error metrics.
        """
        folder = Path(folder)
        
        # apply property and ensemble calculators to models
        calc = StructurePropertyCalculator(model, training=False).to(self.config['device'])
        
        # define atomic type converter
        atomic_type_converter = AtomicTypeConverter.from_type_list(self.config['atomic_types'])
        
        # convert atomic numbers to type names
        test_structures = test_structures.to_type_names(atomic_type_converter, check=True)
        
        # build atomic data sets
        test_ds = test_structures.to_data(r_cutoff=self.config['r_cutoff'],
                                          n_species=atomic_type_converter.get_n_type_names())
        
        # define losses from config
        eval_losses = {l['type']: config_to_loss(l) for l in self.config['eval_losses']}
        eval_output_variables = list(set(sum([l.get_output_variables() for l in eval_losses.values()], [])))
        
        # evaluate model on the test data
        use_gpu = self.config['device'].startswith('cuda')
        test_dl = DataLoader(test_ds, batch_size=self.config['eval_batch_size'], shuffle=False, drop_last=False,
                             pin_memory=use_gpu, pin_memory_device=self.config['device'] if use_gpu else '')
        
        # evaluate metrics on test data and store results as a .json file
        test_metrics = eval_metrics(calc=calc, dl=test_dl, eval_loss_fns=eval_losses,
                                    eval_output_variables=eval_output_variables, device=self.config['device'])
        save_object(folder / f'test_results.json', test_metrics['eval_losses'], use_json=True)
        
        return test_metrics['eval_losses']

    def run_on_configs(self,
                       model: ForwardAtomisticNetwork,
                       test_structures: AtomicStructures,
                       folder: Union[str, Path],
                       file_name: str):
        """Evaluates models on the provided test data set and stores configurations in an .xyz file with predicted total energies and atomic forces.

        Args:
            model (ForwardAtomisticNetwork): Atomistic model.
            test_structures (AtomicStructures): Test structures.
            folder (Union[str, Path]): Folder where the evaluation results are stored.
            file_name (str): Name of the .xyz file, which stores predicted total energies and atomic forces.
        """
        folder = Path(folder)
        
        atoms_list = [s.to_atoms() for s in test_structures]
        
        # apply property and ensemble calculators to models
        calc = StructurePropertyCalculator(model, training=False).to(self.config['device'])
        
        # define atomic type converter
        atomic_type_converter = AtomicTypeConverter.from_type_list(self.config['atomic_types'])
        
        # convert atomic numbers to type names
        test_structures = test_structures.to_type_names(atomic_type_converter, check=True)
        
        # build atomic data sets
        test_ds = test_structures.to_data(r_cutoff=self.config['r_cutoff'],
                                          n_species=atomic_type_converter.get_n_type_names())
        
        # define losses from config
        eval_losses = {l['type']: config_to_loss(l) for l in self.config['eval_losses']}
        eval_output_variables = list(set(sum([l.get_output_variables() for l in eval_losses.values()], [])))
        
        # evaluate model on the test data
        use_gpu = self.config['device'].startswith('cuda')
        test_dl = DataLoader(test_ds, batch_size=self.config['eval_batch_size'], shuffle=False, drop_last=False,
                             pin_memory=use_gpu, pin_memory_device=self.config['device'] if use_gpu else '')
        
        # Collect data
        energies_list = []
        forces_collection = []

        for batch in test_dl:
            results = calc(batch.to(self.config['device']), 
                           forces='forces' in eval_output_variables,
                           stress='stress' in eval_output_variables,
                           virials='virials' in eval_output_variables,
                           create_graph=True)

            energies_list.append(results['energy'].detach().cpu().numpy())
            
            forces = np.split(results['forces'].detach().cpu().numpy(), indices_or_sections=batch.ptr[1:], axis=0)
            forces_collection.append(forces[:-1])  # drop last as its empty

        energies = np.concatenate(energies_list, axis=0)
        forces_list = [forces for forces_list in forces_collection for forces in forces_list]
        
        assert len(atoms_list) == len(energies) == len(forces_list)

        # Store data in atoms objects
        for i, (atoms, energy, forces) in enumerate(zip(atoms_list, energies, forces_list)):
            atoms.calc = None  # crucial
            atoms.info['ICTP_energy'] = energy
            atoms.arrays['ICTP_forces'] = forces

        # Write atoms to output path
        ase.io.write(folder / file_name, images=atoms_list, format="extxyz")
    
    def measure_inference_time(self,
                               model: ForwardAtomisticNetwork,
                               test_structures: AtomicStructures,
                               folder: Union[str, Path],
                               batch_size: int = 100,
                               n_reps: int = 100) -> Dict[str, Any]:
        """Provide inference time for the defined batch size, i.e., atomic system size.

        Args:
            models (ForwardAtomisticNetwork): Atomistic model.
            test_structures (AtomicStructures): Test structures
            folder (Union[str, Path]): Folder where the results of the inference time measurement are stored.
            batch_size (int, optional): Evaluation batch size. Defaults to 100.
            n_reps (int, optional): Number of repetitions. Defaults to 100.

        Returns:
            Dict[str, Any]: Results dictionary.
        """
        folder = Path(folder)
        
        calc = StructurePropertyCalculator(model, training=False).to(self.config['device'])
        
        atomic_type_converter = AtomicTypeConverter.from_type_list(self.config['atomic_types'])
        
        test_structures = test_structures.to_type_names(atomic_type_converter, check=True)
        
        test_ds = test_structures.to_data(r_cutoff=self.config['r_cutoff'], n_species=atomic_type_converter.get_n_type_names())
        test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)
        
        batch = next(iter(test_dl)).to(self.config['device'])
        
        # need to re-iterate before time measurement
        for _ in range(10):
            calc(batch, forces=True)
        
        # start with the time measurement
        if self.config['device'].startswith('cuda'):
            torch.cuda.synchronize()
        
        start_time = time.time()
        for _ in range(n_reps):
            calc(batch, forces=True)
            
        if self.config['device'].startswith('cuda'):
            torch.cuda.synchronize()
            
        end_time = time.time()
        
        to_save = {'total_time': end_time - start_time,
                   'time_per_repetition': (end_time - start_time) / n_reps,
                   'time_per_structure': (end_time - start_time) / n_reps / batch_size,
                   'time_per_atom': (end_time - start_time) / n_reps / batch.n_atoms.sum().item()}
        save_object(folder / f'timing_results.json', to_save, use_json=True)
        
        return to_save
