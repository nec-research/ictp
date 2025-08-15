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
from torch.utils.data import Dataset, Subset

import ase

from ictp.model.calculators import StructurePropertyCalculator
from ictp.model.forward import ForwardAtomisticNetwork, build_model, load_model_from_folder

from ictp.training.callbacks import FileLoggingCallback
from ictp.training.loss_fns import config_to_loss
from ictp.training.trainer import Trainer, eval_metrics

from ictp.utils.config import update_config
from ictp.utils.misc import save_object
from ictp.utils.torch_geometric import DataLoader


class TrainingStrategy:
    """
    Strategy for training interatomic potentials.

    Args:
        config (Optional[Dict[str, Any]], optional): 
            Configuration file with parameters listed in 'utils/config.py'. The default parameters of 'utils/config.py' 
            will be updated by those provided in 'config'. Defaults to None.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # initialize and update the configuration with provided parameters (including the model, training, fine-tuning, and evaluation)
        # all parameters, including those for the model, training, fine-tuning, and evaluation, are stored in `config`
        # the last training configuration is saved in the log and best folders for use during inference
        self.config = update_config(config.copy())

    def run(
        self,
        train_dataset: Dataset,
        valid_dataset: Dataset,
        folder: Union[str, Path],
        model_seed: Optional[int] = None,
        with_torch_script: bool = False,
        with_torch_compile: bool = False,
    ) -> ForwardAtomisticNetwork:
        """
        Runs training using provided training and validation data sets.

        Args:
            train_dataset (Dataset): The data set for training.
            valid_dataset (Dataset): The data set for validation.
            folder (Union[str, Path]): The directory where the trained model will be stored.
            model_seed (int, optional): Random seed for initializing the atomistic model. Defaults to None.
            with_torch_script (bool, optional): If True, the model is compiled using torch.jit.script().
                                                Defaults to False.
            with_torch_compile (bool, optional): If True, the model is compiled using torch.compile().
                                                 Defaults to False.

        Returns:
            ForwardAtomisticNetwork: The trained atomistic model.
        """
        # store data set sizes in the configuration
        n_train = len(train_dataset)
        n_valid = len(valid_dataset)
        self.config['n_train'] = n_train
        self.config['n_valid'] = n_valid
        
        # retrieve the atomic type converter from the data set
        atomic_type_converter = (
            train_dataset.dataset.atomic_type_converter
            if isinstance(train_dataset, Subset) else train_dataset.atomic_type_converter
        )
        
        # update the model seed if provided
        if model_seed is not None:
            self.config['model_seed'] = model_seed
        
        # build the model using data set information and configuration
        model = build_model(
            dataset=train_dataset, 
            n_species=atomic_type_converter.get_n_type_names(),
            Z_max=int(atomic_type_converter._to_atomic_numbers.max()), 
            **self.config,
        )
            
        # define training and evaluation losses from configuration
        train_loss = config_to_loss(self.config['train_loss'])
        eval_losses = {l['type']: config_to_loss(l) for l in self.config['eval_losses']}
        early_stopping_loss = config_to_loss(self.config['early_stopping_loss'])
        
        # set up callbacks for tracking training progress
        callbacks = [FileLoggingCallback()]
        
        # configure the trainer with model and training parameters
        trainer = Trainer(
            model=model,
            model_path=folder,
            callbacks=callbacks,
            lr=self.config['lr'],
            lr_factor=self.config['lr_factor'],
            scheduler_patience=self.config['scheduler_patience'],
            max_epoch=self.config['max_epoch'],
            save_epoch=self.config['save_epoch'],
            validate_epoch=self.config['valid_epoch'],
            train_batch_size=min(self.config['train_batch_size'], n_train),
            valid_batch_size=min(self.config['eval_batch_size'], n_valid),
            n_workers=self.config['n_workers'],
            train_loss=train_loss,
            eval_losses=eval_losses,
            early_stopping_loss=early_stopping_loss,
            max_grad_norm=self.config['max_grad_norm'],
            device=self.config['device'],
            amsgrad=self.config['amsgrad'],
            weight_decay=self.config['weight_decay'],
            ema=self.config['ema'],
            ema_decay=self.config['ema_decay'],
            with_sam=self.config['with_sam'],
            rho_sam=self.config['rho_sam'],
            adaptive_sam=self.config['adaptive_sam'],
            with_torch_script=with_torch_script,
            with_torch_compile=with_torch_compile,
        )
        
        # Train the model
        trainer.fit(train_dataset=train_dataset, valid_dataset=valid_dataset)
        
        # Load the best model and move it to the specified device
        model = load_model_from_folder(folder, key='best').to(self.config['device'])
        
        return model


class EvaluationStrategy:
    """
    Strategy for evaluating the performance of interatomic potentials.

    Args:
        config (Optional[Dict[str, Any]], optional): Configuration file with parameters listed in 'utils/config.py'. 
                                                     The default parameters of 'utils/config.py' will be updated by those 
                                                     provided in 'config'. Defaults to None.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # update config containing all parameters (including the model, training, fine-tuning, and evaluation)
        # we store all parameters in one config for simplicity. In the log and best folders the last training
        # configuration is stored and used when loading a model for inference.
        self.config = update_config(config.copy())

    def run(
        self,
        model: ForwardAtomisticNetwork,
        test_dataset: Dataset,
        folder: Union[str, Path],
        with_torch_script: bool = False,
        with_torch_compile: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluates models using the provided test data set.

        Args:
            model (ForwardAtomisticNetwork): Atomistic model.
            test_dataset (Dataset): Test data set.
            folder (Union[str, Path]): Folder where the evaluation results are stored.
            with_torch_script (bool, optional): If True, the model is compiled using torch.jit.script().
                                                Defaults to False.
            with_torch_compile (bool, optional): If True, the model is compiled using torch.compile().
                                                 Defaults to False.

        Returns:
            Dict[str, Any]: Results dictionary containing test error metrics.
        """
        folder = Path(folder)
        
        # apply property and ensemble calculators to models
        calc = StructurePropertyCalculator(
            model=model,
            with_torch_script=with_torch_script,
            with_torch_compile=with_torch_compile
        ).to(self.config['device'])
        
        # define losses from config
        eval_losses = {l['type']: config_to_loss(l) for l in self.config['eval_losses']}
        eval_output_variables = list(set(sum([l.get_output_variables() for l in eval_losses.values()], [])))
        
        # evaluate model on the test data
        use_gpu = self.config['device'].startswith('cuda')
        test_dl = DataLoader(
            dataset=test_dataset, 
            batch_size=self.config['eval_batch_size'], 
            num_workers=0, 
            shuffle=False, 
            drop_last=False,
            pin_memory=use_gpu, 
            pin_memory_device=self.config['device'] if use_gpu else ''
        )
        
        # evaluate metrics on test data and store results as a .json file
        test_metrics = eval_metrics(
            calc=calc, 
            dl=test_dl, 
            eval_loss_fns=eval_losses,
            eval_output_variables=eval_output_variables, 
            device=self.config['device']
        )
        save_object(folder / f'test_results.json', test_metrics['eval_losses'], use_json=True)
        
        return test_metrics['eval_losses']

    def run_on_configs(
        self,
        model: ForwardAtomisticNetwork,
        test_dataset: Dataset,
        folder: Union[str, Path],
        file_name: str,
        with_torch_script: bool = False,
        with_torch_compile: bool = False,
    ):
        """
        Evaluates models on the provided test data set and stores configurations in an .xyz file with predicted total energies and atomic forces.

        Args:
            model (ForwardAtomisticNetwork): Atomistic model.
            test_dataset (Dataset): Test data set.
            folder (Union[str, Path]): Folder where the evaluation results are stored.
            file_name (str): Name of the .xyz file, which stores predicted total energies and atomic forces.
            with_torch_script (bool, optional): If True, the model is compiled using torch.jit.script().
                                                Defaults to False.
            with_torch_compile (bool, optional): If True, the model is compiled using torch.compile().
                                                 Defaults to False.
            
        """
        folder = Path(folder)
            
        atoms_list = [d.to_atoms() for d in test_dataset]
        
        # apply property and ensemble calculators to models
        calc = StructurePropertyCalculator(
            model=model,
            with_torch_script=with_torch_script,
            with_torch_compile=with_torch_compile
        ).to(self.config['device'])
        
        # define losses from config
        eval_losses = {l['type']: config_to_loss(l) for l in self.config['eval_losses']}
        eval_output_variables = list(set(sum([l.get_output_variables() for l in eval_losses.values()], [])))
        
        # evaluate model on the test data
        use_gpu = self.config['device'].startswith('cuda')
        test_dl = DataLoader(
            dataset=test_dataset, 
            batch_size=self.config['eval_batch_size'],
            num_workers=0,
            shuffle=False,
            drop_last=False,
            pin_memory=use_gpu,
            pin_memory_device=self.config['device'] if use_gpu else ''
        )
        
        # collect data
        energies_list = []
        forces_collection = []
        
        partial_charges_collection = []
        dipole_moments_list = []
        quadrupole_moments_list = []

        for batch in test_dl:
            results = calc(batch.to(self.config['device']), 
                           forces='forces' in eval_output_variables,
                           stress='stress' in eval_output_variables,
                           virials='virials' in eval_output_variables,
                           dipole_moment='dipole_moment' in eval_output_variables,
                           quadrupole_moment='quadrupole_moment' in eval_output_variables,
                           create_graph=False)

            energies_list.append(results['energy'].detach().cpu().numpy())
            forces = np.split(results['forces'].detach().cpu().numpy(), indices_or_sections=batch.ptr[1:], axis=0)
            forces_collection.append(forces[:-1])  # drop last as its empty
            
            if 'partial_charges' in results:
                partial_charges = np.split(results['partial_charges'].detach().cpu().numpy(), indices_or_sections=batch.ptr[1:], axis=0)
                partial_charges_collection.append(partial_charges[:-1])
                
            if 'dipole_moment' in results:
                dipole_moments_list.append(results['dipole_moment'].detach().cpu().numpy())
                
            if 'quadrupole_moment' in results:
                quadrupole_moments_list.append(results['quadrupole_moment'].detach().cpu().numpy())

        energies = np.concatenate(energies_list, axis=0)
        forces_list = [forces for forces_list in forces_collection for forces in forces_list]
        
        if partial_charges_collection:
            partial_charges_list = [partial_charges for partial_charges_list in partial_charges_collection for partial_charges in partial_charges_list]
            
        if dipole_moments_list:
            dipole_moments = np.concatenate(dipole_moments_list, axis=0)
            
        if quadrupole_moments_list:
            quadrupole_moments = np.concatenate(quadrupole_moments_list, axis=0)
        
        assert len(atoms_list) == len(energies) == len(forces_list)
        
        if partial_charges_collection:
            assert len(atoms_list) == len(partial_charges_list)
        
        if dipole_moments_list:
            assert len(atoms_list) == len(dipole_moments)
            
        if quadrupole_moments_list:
            assert len(atoms_list) == len(quadrupole_moments)

        # store data in atoms objects
        for i, (atoms, energy, forces) in enumerate(zip(atoms_list, energies, forces_list)):
            atoms.calc = None  # crucial
            atoms.info['ICTP_energy'] = energy
            atoms.arrays['ICTP_forces'] = forces
            
            if partial_charges_collection:
                atoms.arrays['ICTP_partial_charges'] = partial_charges_list[i]
            
            if dipole_moments_list:
                atoms.info['ICTP_dipole_moment'] = dipole_moments[i]
                
            if quadrupole_moments_list:
                atoms.info['ICTP_quadrupole_moment'] = quadrupole_moments[i]

        # write atoms to output path
        ase.io.write(folder / file_name, images=atoms_list, format="extxyz")
    
    def measure_inference_time(
        self,
        model: ForwardAtomisticNetwork,
        test_dataset: Dataset,
        folder: Union[str, Path],
        batch_size: int = 100,
        n_reps: int = 100,
        with_torch_script: bool = False,
        with_torch_compile: bool = False,
    ) -> Dict[str, Any]:
        """
        Provide inference time for the defined batch size, i.e., atomic system size.

        Args:
            models (ForwardAtomisticNetwork): Atomistic model.
            test_dataset (Dataset): Test data set.
            folder (Union[str, Path]): Folder where the results of the inference time measurement are stored.
            batch_size (int, optional): Evaluation batch size. Defaults to 100.
            n_reps (int, optional): Number of repetitions. Defaults to 100.
            with_torch_script (bool, optional): If True, the model is compiled using torch.jit.script().
                                                Defaults to False.
            with_torch_compile (bool, optional): If True, the model is compiled using torch.compile().
                                                 Defaults to False.

        Returns:
            Dict[str, Any]: Results dictionary.
        """
        folder = Path(folder)
        
        calc = StructurePropertyCalculator(
            model=model,
            with_torch_script=with_torch_script,
            with_torch_compile=with_torch_compile
        ).to(self.config['device'])
        
        test_dl = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            num_workers=0,
            shuffle=False,
            drop_last=False
        )
        
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
        
        to_save = {
            'total_time': end_time - start_time,
            'time_per_repetition': (end_time - start_time) / n_reps,
            'time_per_structure': (end_time - start_time) / n_reps / batch_size,
            'time_per_atom': (end_time - start_time) / n_reps / batch.n_atoms.sum().item()
        }
        save_object(folder / f'timing_results.json', to_save, use_json=True)
        
        return to_save
