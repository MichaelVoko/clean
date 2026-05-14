from openbabel import openbabel
openbabel.obErrorLog.SetOutputLevel(0)
openbabel.cvar.obErrorLog.StopLogging()
import numpy as np
import pandas as pd
import sys
import torch
from ema_pytorch import EMA
import time
import json
import os
import pickle
import wandb

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import cifutils
import pdbutils
from na_data_utils import PDBDataset, make_batch_iter 
from na_model_utils import featurize, loss_smoothed, loss_nll, compute_canonical_base_pair_accuracy, get_std_opt, ProteinMPNN
from na_metric_manager import generate_metric_manager, DFMMetricManager

JSON = sys.argv[1]
params = json.load(open(JSON))

scaler = torch.cuda.amp.GradScaler()

local_rank = int(os.environ.get("LOCAL_RANK", 0))
global_rank = int(os.environ.get("RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))
is_distributed = world_size > 1

if is_distributed:
    dist.init_process_group(backend="nccl", init_method="env://", rank=global_rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if params["BASE_FOLDER"][-1] != '/':
    params["BASE_FOLDER"] += '/'
if not os.path.exists(params["BASE_FOLDER"]):
    os.makedirs(params["BASE_FOLDER"])

if not is_distributed or global_rank == 0:
    _wandb_id_file = params["BASE_FOLDER"] + "wandb_run_id.txt"
    _wandb_id = params.get("WANDB_RUN_ID", None)
    if _wandb_id is None:
        if os.path.exists(_wandb_id_file):
            with open(_wandb_id_file) as _f:
                _wandb_id = _f.read().strip()
    wandb.init(
        project=params.get("WANDB_PROJECT", "NA-MPNN"),
        name=params.get("WANDB_RUN_NAME", None),
        id=_wandb_id,
        config=params,
        resume="allow",
    )
    with open(_wandb_id_file, "w") as _f:
        _f.write(wandb.run.id)
logfile = params["BASE_FOLDER"] + 'log.txt'
if not params["PREV_CHECKPOINT"]:
    with open(logfile, 'w') as f:
        f.write('Epoch\tTrain\tValidation\n')

if params["ATOMS_TO_LOAD"] == "backbone":
    atom_list_to_save = ['N', 'CA', 'C', 'O', #protein atoms
                         'OP1', 'OP2', 'P', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'" #nucleic acid atoms
                        ]
elif params["ATOMS_TO_LOAD"] == "all":
    atom_list_to_save = ['N', 'CA', 'C', 'CB', 'O', 'CG', 'CG1', 'CG2', 'OG', 'OG1', 'SG', 'CD', 'CD1', 'CD2', 'ND1', 'ND2', 'OD1', 'OD2', 'SD', 'CE', 'CE1', 'CE2', 'CE3', 'NE', 'NE1', 'NE2', 'OE1', 'OE2', 'CH2', 'NH1', 'NH2', 'OH', 'CZ', 'CZ2', 'CZ3', 'NZ', 'OXT', #protein atoms
                         'OP1', 'OP2', 'P', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", 'N9', 'C8', 'C7', 'N7', 'C6', 'N6', 'O6', 'C5', 'C4', 'N4', 'O4', 'N3', 'C2', 'N2', 'O2', 'N1' #nucleic acid atoms
                        ]

cif_parser = cifutils.CIFParser(skip_res=params["EXCLUDE_RES"], randomize_nmr_model=params["RANDOMIZE_NMR_MODEL"])
pdb_parser = pdbutils.PDBParser()

# Load CD-HIT cluster lookup for cluster-aware DFM masking.
chain_cluster_lookup = {}
if "CHAIN_CLUSTER_LOOKUP_PATH" in params and params["CHAIN_CLUSTER_LOOKUP_PATH"]:
    with open(params["CHAIN_CLUSTER_LOOKUP_PATH"], "rb") as f:
        chain_cluster_lookup = pickle.load(f)
    print(f"Loaded chain_cluster_lookup: {len(chain_cluster_lookup)} structures")

pdb_dataset = PDBDataset(cif_parser=cif_parser,
                         pdb_parser=pdb_parser,
                         atom_list_to_save=atom_list_to_save,
                         parse_protein=params["PARSE_PROTEIN"],
                         parse_dna=params["PARSE_DNA"],
                         parse_rna=params["PARSE_RNA"],
                         parse_rna_as_dna=params["PARSE_RNA_AS_DNA"],
                         na_shared_tokens=params["NA_SHARED_TOKENS"],
                         protein_backbone_occ_cutoff=params["PROTEIN_BACKBONE_OCC_CUTOFF"],
                         protein_side_chain_occ_cutoff=params["PROTEIN_SIDE_CHAIN_OCC_CUTOFF"],
                         dna_backbone_occ_cutoff=params["DNA_BACKBONE_OCC_CUTOFF"],
                         dna_side_chain_occ_cutoff=params["DNA_SIDE_CHAIN_OCC_CUTOFF"],
                         rna_backbone_occ_cutoff=params["RNA_BACKBONE_OCC_CUTOFF"],
                         rna_side_chain_occ_cutoff=params["RNA_SIDE_CHAIN_OCC_CUTOFF"],
                         crop_large_structures=params["CROP_LARGE_STRUCTURES"],
                         batch_tokens=params["BATCH_TOKENS"],
                         na_ref_atom=params["NA_REF_ATOM"],
                         parse_ppms=params["PARSE_PPMS"],
                         min_overlap_length=params["MIN_OVERLAP_LENGTH"],
                         drop_protein_probability=params["DROP_PROTEIN_PROBABILITY"],
                         na_only_as_uniform_ppm=params["NA_ONLY_AS_UNIFORM_PPM"],
                         protein_interface_residue_mutation_probability=params["PROTEIN_INTERFACE_RESIDUE_MUTATION_PROBABILITY"],
                         mutate_base_pair_together=params["MUTATE_BASE_PAIR_TOGETHER"],
                         mutate_entire_side_chain_interface_probability=params["MUTATE_ENTIRE_SIDE_CHAIN_INTERFACE_PROBABILITY"],
                         na_non_interface_as_uniform_ppm=params["NA_NON_INTERFACE_AS_UNIFORM_PPM"],
                         chain_cluster_lookup=chain_cluster_lookup
                         )

model = ProteinMPNN(node_features=params["HIDDEN_DIM"],
                    edge_features=params["HIDDEN_DIM"],
                    hidden_dim=params["HIDDEN_DIM"],
                    num_encoder_layers=params["NUM_ENCODER_LAYERS"],
                    num_decoder_layers=params["NUM_DECODER_LAYERS"],
                    k_neighbors=params["NUM_NEIGHBORS"],
                    dropout=params["DROPOUT"],
                    atom_dict=pdb_dataset.atom_dict,
                    restype_to_int=pdb_dataset.restype_to_int,
                    polytype_to_int=pdb_dataset.polytype_to_int,
                    protein_augment_eps=params["PROTEIN_BACKBONE_NOISE"],
                    dna_augment_eps=params["DNA_BACKBONE_NOISE"],
                    rna_augment_eps=params["RNA_BACKBONE_NOISE"],
                    decode_protein_first=params["DECODE_PROTEIN_FIRST"],
                    na_ref_atom=params["NA_REF_ATOM"],
                    include_pred_na_N=params["INCLUDE_PRED_NA_N"],
                    device=device,
                    vocab=params["VOCAB_SIZE"],
                    num_letters=params["NUM_LETTERS"],
                    mode=params.get("MODE", "ar"))
model.to(device)

ema = EMA(model, beta=params.get("EMA_DECAY", 0.9999))

if is_distributed:
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

if not is_distributed or global_rank == 0:
    wandb.watch(model, log="all", log_freq=100)

if params["PREV_CHECKPOINT"]:
    if not os.path.exists(params["PREV_CHECKPOINT"]):
        print(f"Checkpoint not found at {params['PREV_CHECKPOINT']}, starting from scratch.")
        total_step = 0
        epoch = 0
        save_step = 0
        params["PREV_CHECKPOINT"] = []
    else:
        checkpoint = torch.load(params["PREV_CHECKPOINT"])
        total_step = checkpoint['step'] # write total_step from the checkpoint
        save_step = checkpoint['save_step']
        epoch = checkpoint['epoch'] # write epoch from the checkpoint
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'ema_state_dict' in checkpoint:
            ema.load_state_dict(checkpoint['ema_state_dict'])
        else:
            # Old checkpoint without EMA: sync EMA from the freshly loaded model weights.
            ema.copy_params_from_model_to_ema()
        print(f"Starting from step {total_step}")
else:
    total_step = 0
    epoch = 0
    save_step = 0


optimizer = get_std_opt(model.parameters(), params["HIDDEN_DIM"], total_step)

if params["PREV_CHECKPOINT"]:
    optimizer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

df_valid = pd.read_csv(params["DF_PATH_VALID"])
df_train = pd.read_csv(params["DF_PATH_TRAIN"])

# Convert the dates to datetime.
df_valid["date"] = pd.to_datetime(df_valid["date"], format = "%Y-%m-%d")
df_train["date"] = pd.to_datetime(df_train["date"], format = "%Y-%m-%d")

# Convert the date cutoff to datetime.
date_cutoff = pd.to_datetime(params["DATE_CUTOFF"], format = "%Y-%m-%d")

metric_manager = generate_metric_manager(pdb_dataset.restype_to_int, metrics_to_compute = params["METRICS_TO_COMPUTE"])
# Two separate DFM managers: training uses rolling N-step windows to preserve
# temporal dynamics; validation uses epoch-level aggregation (flush_every=None).
if params["MODE"] == "dfm":
    dfm_train_manager = DFMMetricManager(
        splits=["train"], polymer_names=["protein", "dna", "rna"],
        n_bins=params.get("DFM_T_BINS", 10),
        flush_every=params.get("DFM_LOG_INTERVAL", 200)
    )
    dfm_valid_manager = DFMMetricManager(
        splits=["valid"], polymer_names=["protein", "dna", "rna"],
        n_bins=params.get("DFM_T_BINS", 10),
        flush_every=None
    )
    dfm_logfile = params["BASE_FOLDER"] + "dfm_log.txt"
else:
    dfm_train_manager = dfm_valid_manager = dfm_logfile = None

tokens_with_no_loss = torch.tensor([pdb_dataset.restype_to_int["UNK"], 
                                    pdb_dataset.restype_to_int["DX"], 
                                    pdb_dataset.restype_to_int["RX"], 
                                    pdb_dataset.restype_to_int["MAS"], 
                                    pdb_dataset.restype_to_int["PAD"]], 
                                   device=device) 
# restypes that are masked in the loss function, but still included in the input and metrics.
# UNK for unknown residues, DX and RX for DNA and RNA residues when NA_SHARED_TOKENS is True, MAS for masked positions in DFM, and PAD for padding.

# Masks used for loss function.
protein_restype_mask = torch.zeros(params["NUM_LETTERS"], device=device)
protein_restype_mask[pdb_dataset.protein_restype_ints] = 1

dna_restype_mask = torch.zeros(params["NUM_LETTERS"], device=device)
dna_restype_mask[pdb_dataset.dna_restype_ints] = 1
    
rna_restype_mask = torch.zeros(params["NUM_LETTERS"], device=device)
rna_restype_mask[pdb_dataset.rna_restype_ints] = 1

polymer_restype_masks = {"protein": protein_restype_mask,
                         "dna": dna_restype_mask,
                         "rna": rna_restype_mask}

polymer_restype_nums = {"protein": len(pdb_dataset.protein_restype_ints),
                        "dna": len(pdb_dataset.dna_restype_ints),
                        "rna": len(pdb_dataset.rna_restype_ints)}

                        
# The main training loop. Loop over epochs, but the actual number of epochs is determined by the TOTAL_STEPS parameter and the number of steps we take in each epoch.              
for e in range(100000):
    metric_manager.zero_metrics()
    if dfm_train_manager is not None:
        dfm_train_manager.zero_metrics()
        dfm_valid_manager.zero_metrics()
    # grad_norm_sum/count track the rolling-window average flushed alongside DFM metrics.
    grad_norm_sum, grad_norm_count = 0.0, 0

    # Use a shared epoch seed so all ranks generate the same batch ordering.
    # This is required for correct DDP sharding: all ranks must do the same
    # number of forward/backward passes or ALLREDUCE will deadlock.
    epoch_seed = e  # deterministic per epoch, different each epoch
    np.random.seed(epoch_seed)
    batch_iter_valid = make_batch_iter(df = df_valid,
                                       batch_tokens = params["BATCH_TOKENS"],
                                       length_cutoff = params["MIN_PROTEIN_LENGTH_CUTOFF"],
                                       date_cutoff = date_cutoff,
                                       crop_large_structures = params["CROP_LARGE_STRUCTURES"],
                                       max_number_of_pdbs = params["MAX_NUMBER_OF_PDBS_VALID"])
    np.random.seed(epoch_seed + 100000)
    batch_iter_train = make_batch_iter(df = df_train,
                                       batch_tokens = params["BATCH_TOKENS"],
                                       length_cutoff = params["MIN_PROTEIN_LENGTH_CUTOFF"],
                                       date_cutoff = date_cutoff,
                                       crop_large_structures = params["CROP_LARGE_STRUCTURES"],
                                       max_number_of_pdbs = params["MAX_NUMBER_OF_PDBS_TRAIN"])

    if is_distributed:
        # Slice the pre-built batch list so each rank gets its own equal-sized
        # contiguous portion. Truncate to a multiple of world_size so no rank
        # gets an extra step (which would cause an ALLREDUCE deadlock).
        train_batches_list = list(batch_iter_train)
        train_batches_list = train_batches_list[:(len(train_batches_list) // world_size) * world_size]
        train_batches_list = train_batches_list[global_rank::world_size]

        valid_batches_list = list(batch_iter_valid)
        valid_batches_list = valid_batches_list[global_rank::world_size]

        train_sampler = torch.utils.data.sampler.BatchSampler(
            iter(train_batches_list), batch_size=1, drop_last=False)
        valid_sampler = torch.utils.data.sampler.BatchSampler(
            iter(valid_batches_list), batch_size=1, drop_last=False)
    else:
        valid_sampler = torch.utils.data.sampler.BatchSampler(
            batch_iter_valid, batch_size=1, drop_last=False)
        train_sampler = torch.utils.data.sampler.BatchSampler(
            batch_iter_train, batch_size=1, drop_last=False)

    valid_loader = torch.utils.data.DataLoader(
        pdb_dataset,
        sampler=valid_sampler,
        num_workers=params["NUM_WORKERS"],
        pin_memory=True,
        persistent_workers=params["NUM_WORKERS"] > 0,
        prefetch_factor=4 if params["NUM_WORKERS"] > 0 else None)

    train_loader = torch.utils.data.DataLoader(
        pdb_dataset,
        sampler=train_sampler,
        num_workers=params["NUM_WORKERS"],
        pin_memory=True,
        persistent_workers=params["NUM_WORKERS"] > 0,
        prefetch_factor=4 if params["NUM_WORKERS"] > 0 else None)

    model.train() # training mode sets self.training = True for modules like dropout, which behave differently during training and evaluation.
    e = epoch + e
    t0 = time.time()
    train_batches_total = 0
    train_batches_skipped = 0
    train_batches_updated = 0

    #############################################################################################

    for ix, batch in enumerate(train_loader):
        train_batches_total += 1
        optimizer.zero_grad()
        feature_dict = featurize(batch, pdb_dataset.polytype_to_int, pdb_dataset.restype_to_int, pdb_dataset.atom_dict, device)
        if type(feature_dict) == str:
            train_batches_skipped += 1
            continue

        S = feature_dict["S"]

        mask = feature_dict["mask"]
        S_mask = 1 - (torch.any(S[:,:,None] == tokens_with_no_loss[None,None,:], dim = -1)).long()
        mask_for_loss = mask * S_mask
        feature_dict["mask_for_loss"] = mask_for_loss

        polymer_masks = {"protein": feature_dict["protein_mask"], "dna": feature_dict["dna_mask"], "rna": feature_dict["rna_mask"]}
        if params["METRICS_TO_COMPUTE"] == "all":
            interface_masks = {"interface": feature_dict["interface_mask"],
                            "nonInterface": 1 - feature_dict["interface_mask"]}
        else:
            interface_masks = {}

        if params["MIXED_PRECISION"]:
            with torch.cuda.amp.autocast():
                # Forward pass
                if params["MODE"] == "ar":
                    log_probs, probs = model(feature_dict)
                elif params["MODE"] == "dfm":
                    X_1 = S.clone() # OG seqeunce
                    X_0 = torch.full_like(X_1, pdb_dataset.restype_to_int["MAS"]) # Fully masked sequence
                    B, L = X_1.shape
                    t_matrix = torch.zeros(B, L, device=X_1.device) # per-position t (varies by chain)
                    t_sample = torch.rand_like(X_1, dtype=torch.float32) # [B, L]
                    t_anchor = torch.zeros(B, 1, device=X_1.device) # anchor t for W_t conditioning

                    # Cluster-aware chain selection: sample an anchor chain, find all
                    # same-cluster chains in the assembly, independently mask each.
                    selected_chain_mask = torch.zeros_like(mask_for_loss, dtype=torch.float32)
                    cluster_ids = feature_dict["cluster_id_labels"] # [B, L]
                    chain_letters = feature_dict["chain_letter_labels"] # [B, L]
                    for b in range(B):
                        valid_pos = mask_for_loss[b].bool()
                        if not valid_pos.any():
                            continue
                        valid_letters = torch.unique(chain_letters[b][valid_pos])
                        valid_letters = valid_letters[valid_letters >= 0]
                        if len(valid_letters) == 0:
                            continue
                        anchor_letter = valid_letters[np.random.randint(len(valid_letters))]
                        anchor_pos = (chain_letters[b] == anchor_letter) & valid_pos
                        anchor_cluster = cluster_ids[b][anchor_pos][0].item()

                        # Anchor chain: sample t and mark for loss.
                        t_a = torch.rand(1, device=X_1.device).item()
                        t_anchor[b, 0] = t_a
                        t_matrix[b][anchor_pos] = t_a
                        selected_chain_mask[b][anchor_pos] = 1

                        # Find other chains sharing the anchor's cluster (skip if anchor has no cluster).
                        if anchor_cluster >= 0:
                            for other_letter in valid_letters:
                                if other_letter == anchor_letter:
                                    continue
                                other_pos = (chain_letters[b] == other_letter) & valid_pos
                                if not other_pos.any():
                                    continue
                                if cluster_ids[b][other_pos][0].item() == anchor_cluster:
                                    t_matrix[b][other_pos] = t_a
                                    selected_chain_mask[b][other_pos] = 1

                    valid = mask_for_loss.bool()
                    X_t = torch.where(t_sample < t_matrix, X_1, X_0) # DFM noise per chain
                    X_t = torch.where(valid, X_t, X_1) # restore invalid (PAD/UNK) positions
                    # Non-selected chains: restore true sequence as background context.
                    X_t = torch.where((mask_for_loss - selected_chain_mask).bool(), X_1, X_t)
                    # Restrict loss to masked positions only (ELBO only sums over masked tokens).
                    mas_mask = selected_chain_mask * (X_t == pdb_dataset.restype_to_int["MAS"]).float()
                    mask_for_loss = mas_mask
                    feature_dict["mask_for_loss"] = mask_for_loss
                    feature_dict["S"] = X_t
                    # Stash t and X_t for DFM metric accumulation after the backward pass.
                    t_batch, X_t_batch = t_anchor, X_t
                    log_probs, probs = model(feature_dict,mode='dfm',t=t_anchor)
                else:
                    raise ValueError("MODE not recognized")

                loss_for_metric, loss_av_smoothed = loss_smoothed(S,
                                                log_probs,
                                                mask_for_loss,
                                                polymer_masks=polymer_masks,
                                                polymer_restype_masks=polymer_restype_masks,
                                                polymer_restype_nums=polymer_restype_nums,
                                                weight=params["LABEL_SMOOTHING"],
                                                tokens=params["LOSS_TOKENS"],
                                                num_letters=params["NUM_LETTERS"],
                                                ppm_mask=feature_dict["ppm_mask"],
                                                aligned_ppm=feature_dict["aligned_ppm"],
                                                mean_over_mask=(params["MODE"] == "dfm"))

            scaler.scale(loss_av_smoothed).backward()

            if params["GRADIENT_NORM"] > 0.0:
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), params["GRADIENT_NORM"])
                # Track gradient norm for epoch-average logging.
                grad_norm_sum += total_norm.item()
                grad_norm_count += 1

            scaler.step(optimizer)
            scaler.update()
            ema.update()
        else:
            # Forward pass
            if params["MODE"] == "ar":
                log_probs, probs = model(feature_dict)
            elif params["MODE"] == "dfm":
                X_1 = S.clone() # OG seqeunce
                X_0 = torch.full_like(X_1, pdb_dataset.restype_to_int["MAS"]) # Fully masked sequence
                B, L = X_1.shape
                t_matrix = torch.zeros(B, L, device=X_1.device)
                t_sample = torch.rand_like(X_1, dtype=torch.float32)
                t_anchor = torch.zeros(B, 1, device=X_1.device)

                selected_chain_mask = torch.zeros_like(mask_for_loss, dtype=torch.float32)
                cluster_ids = feature_dict["cluster_id_labels"]
                chain_letters = feature_dict["chain_letter_labels"]
                for b in range(B):
                    valid_pos = mask_for_loss[b].bool()
                    if not valid_pos.any():
                        continue
                    valid_letters = torch.unique(chain_letters[b][valid_pos])
                    valid_letters = valid_letters[valid_letters >= 0]
                    if len(valid_letters) == 0:
                        continue
                    anchor_letter = valid_letters[np.random.randint(len(valid_letters))]
                    anchor_pos = (chain_letters[b] == anchor_letter) & valid_pos
                    anchor_cluster = cluster_ids[b][anchor_pos][0].item()

                    t_a = torch.rand(1, device=X_1.device).item()
                    t_anchor[b, 0] = t_a
                    t_matrix[b][anchor_pos] = t_a
                    selected_chain_mask[b][anchor_pos] = 1

                    if anchor_cluster >= 0:
                        for other_letter in valid_letters:
                            if other_letter == anchor_letter:
                                continue
                            other_pos = (chain_letters[b] == other_letter) & valid_pos
                            if not other_pos.any():
                                continue
                            if cluster_ids[b][other_pos][0].item() == anchor_cluster:
                                t_matrix[b][other_pos] = t_a
                                selected_chain_mask[b][other_pos] = 1

                valid = mask_for_loss.bool()
                X_t = torch.where(t_sample < t_matrix, X_1, X_0)
                X_t = torch.where(valid, X_t, X_1)
                X_t = torch.where((mask_for_loss - selected_chain_mask).bool(), X_1, X_t)
                mas_mask = selected_chain_mask * (X_t == pdb_dataset.restype_to_int["MAS"]).float()
                mask_for_loss = mas_mask
                feature_dict["mask_for_loss"] = mask_for_loss
                feature_dict["S"] = X_t
                t_batch, X_t_batch = t_anchor, X_t
                log_probs, probs = model(feature_dict,mode='dfm',t=t_anchor)
            else:
                raise ValueError("MODE not recognized")

            loss_for_metric, loss_av_smoothed = loss_smoothed(S,
                                            log_probs,
                                            mask_for_loss,
                                            polymer_masks=polymer_masks,
                                            polymer_restype_masks=polymer_restype_masks,
                                            polymer_restype_nums=polymer_restype_nums,
                                            weight=params["LABEL_SMOOTHING"],
                                            tokens=params["LOSS_TOKENS"],
                                            num_letters=params["NUM_LETTERS"],
                                            ppm_mask=feature_dict["ppm_mask"],
                                            aligned_ppm=feature_dict["aligned_ppm"],
                                            mean_over_mask=(params["MODE"] == "dfm"))

            loss_av_smoothed.backward()

            if params["GRADIENT_NORM"] > 0.0:
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), params["GRADIENT_NORM"])
                # Track gradient norm for epoch-average logging.
                grad_norm_sum += total_norm.item()
                grad_norm_count += 1

            optimizer.step()
            ema.update()

        loss, loss_av, true_false = loss_nll(S, log_probs, mask_for_loss)
        canonical_base_pair_accuracy = \
            compute_canonical_base_pair_accuracy(log_probs,
                                                feature_dict["canonical_base_pair_mask"],
                                                feature_dict["canonical_base_pair_index"],
                                                pdb_dataset)

        S_pred = torch.argmax(log_probs, -1)

        metric_manager.accumulate(loss_for_metric,
                                true_false,
                                canonical_base_pair_accuracy,
                                feature_dict["canonical_base_pair_mask"],
                                S,
                                S_pred,
                                "train",
                                mask_for_loss,
                                polymer_masks,
                                interface_masks)
        # Accumulate DFM-specific metrics: t-binned loss/accuracy and masked-position entropy.
        if dfm_train_manager is not None:
            pred_entropy = -(probs * log_probs).sum(-1)  # [B, L] Shannon entropy
            dfm_train_manager.accumulate(
                loss, true_false, pred_entropy, t_batch, X_t_batch,
                polymer_masks, "train",
                pdb_dataset.restype_to_int["MAS"]
            )
            # Flush the rolling window every DFM_LOG_INTERVAL steps, writing to dfm_log.txt.
            # This preserves temporal resolution so we can see the entropy-vs-t curve evolve.
            if dfm_train_manager.should_flush():
                gn_mean = (
                    np.format_float_positional(
                        np.float32(grad_norm_sum / grad_norm_count), unique=False, precision=3
                    ) if grad_norm_count > 0 else "nan"
                )
                dfm_record = dfm_train_manager.flush(
                    total_step, dfm_logfile,
                    extra_fields={"grad_norm_mean": gn_mean}
                )
                if dfm_record is not None and (not is_distributed or global_rank == 0):
                    wandb.log(
                        {f"dfm_train/{k}": v for k, v in dfm_record.items()
                         if v is not None and k not in ("step", "window")},
                        step=total_step,
                    )
                grad_norm_sum, grad_norm_count = 0.0, 0
        train_batches_updated += 1
        if not is_distributed or global_rank == 0:
            loss_val = np.format_float_positional(np.float32(loss_av.item()), unique=False, precision=4)
            print(f"step: {total_step}, loss: {loss_val}", flush=True)
#############################################################################################
        total_step += 1

    model.eval() # Evaluate on the validation set. No backpropagation, but still compute metrics.
    t1 = time.time()

    #############################################################################################
    with torch.no_grad():
        for ix, batch in enumerate(valid_loader):
            feature_dict = featurize(batch, pdb_dataset.polytype_to_int, pdb_dataset.restype_to_int, pdb_dataset.atom_dict, device)
            if type(feature_dict) == str:
                continue
            S = feature_dict["S"]
            mask = feature_dict["mask"]
            S_mask = 1 - (torch.any(S[:,:,None] == tokens_with_no_loss[None,None,:], dim = -1)).long()
            mask_for_loss = mask * S_mask
            feature_dict["mask_for_loss"] = mask_for_loss

            polymer_masks = {"protein": feature_dict["protein_mask"], "dna": feature_dict["dna_mask"], "rna": feature_dict["rna_mask"]}
            if params["METRICS_TO_COMPUTE"] == "all":
                interface_masks = {"interface": feature_dict["interface_mask"],
                                   "nonInterface": 1 - feature_dict["interface_mask"]}
            else:
                interface_masks = {}

            # Forward pass (use EMA model for more stable validation metrics)
            if params["MODE"] == "ar":
                log_probs, probs = model(feature_dict)
            elif params["MODE"] == "dfm":
                X_1 = S.clone() # OG seqeunce
                X_0 = torch.full_like(X_1, pdb_dataset.restype_to_int["MAS"]) # Fully masked sequence
                B, L = X_1.shape
                t_matrix = torch.zeros(B, L, device=X_1.device)
                t_sample = torch.rand_like(X_1, dtype=torch.float32)
                t_anchor = torch.zeros(B, 1, device=X_1.device)

                selected_chain_mask = torch.zeros_like(mask_for_loss, dtype=torch.float32)
                cluster_ids = feature_dict["cluster_id_labels"]
                chain_letters = feature_dict["chain_letter_labels"]
                for b in range(B):
                    valid_pos = mask_for_loss[b].bool()
                    if not valid_pos.any():
                        continue
                    valid_letters = torch.unique(chain_letters[b][valid_pos])
                    valid_letters = valid_letters[valid_letters >= 0]
                    if len(valid_letters) == 0:
                        continue
                    anchor_letter = valid_letters[np.random.randint(len(valid_letters))]
                    anchor_pos = (chain_letters[b] == anchor_letter) & valid_pos
                    anchor_cluster = cluster_ids[b][anchor_pos][0].item()

                    t_a = torch.rand(1, device=X_1.device).item()
                    t_anchor[b, 0] = t_a
                    t_matrix[b][anchor_pos] = t_a
                    selected_chain_mask[b][anchor_pos] = 1

                    if anchor_cluster >= 0:
                        for other_letter in valid_letters:
                            if other_letter == anchor_letter:
                                continue
                            other_pos = (chain_letters[b] == other_letter) & valid_pos
                            if not other_pos.any():
                                continue
                            if cluster_ids[b][other_pos][0].item() == anchor_cluster:
                                t_matrix[b][other_pos] = t_a
                                selected_chain_mask[b][other_pos] = 1

                valid = mask_for_loss.bool()
                X_t = torch.where(t_sample < t_matrix, X_1, X_0)
                X_t = torch.where(valid, X_t, X_1)
                X_t = torch.where((mask_for_loss - selected_chain_mask).bool(), X_1, X_t)
                mas_mask = selected_chain_mask * (X_t == pdb_dataset.restype_to_int["MAS"]).float()
                mask_for_loss = mas_mask
                feature_dict["mask_for_loss"] = mask_for_loss
                feature_dict["S"] = X_t
                t_batch, X_t_batch = t_anchor, X_t
                log_probs, probs = ema.ema_model(feature_dict,mode='dfm',t=t_anchor)
            else:
                raise ValueError("MODE not recognized")

            loss, loss_av, true_false = loss_nll(S, log_probs, mask_for_loss)
            canonical_base_pair_accuracy = \
                compute_canonical_base_pair_accuracy(log_probs,
                                                     feature_dict["canonical_base_pair_mask"],
                                                     feature_dict["canonical_base_pair_index"],
                                                     pdb_dataset)
            S_pred = torch.argmax(log_probs, -1)

            loss_for_metric, _ = loss_smoothed(S,
                                               log_probs,
                                               mask_for_loss,
                                               polymer_masks=polymer_masks,
                                               polymer_restype_masks=polymer_restype_masks,
                                               polymer_restype_nums=polymer_restype_nums,
                                               weight=params["LABEL_SMOOTHING"],
                                               tokens=params["LOSS_TOKENS"],
                                               num_letters=params["NUM_LETTERS"],
                                               ppm_mask=feature_dict["ppm_mask"],
                                               aligned_ppm=feature_dict["aligned_ppm"])

            metric_manager.accumulate(loss_for_metric,
                                      true_false,
                                      canonical_base_pair_accuracy,
                                      feature_dict["canonical_base_pair_mask"],
                                      S,
                                      S_pred,
                                      "valid",
                                      mask_for_loss,
                                      polymer_masks,
                                      interface_masks)
            # Accumulate DFM-specific metrics: t-binned loss/accuracy and masked-position entropy.
            if dfm_valid_manager is not None:
                pred_entropy = -(probs * log_probs).sum(-1)  # [B, L] Shannon entropy
                dfm_valid_manager.accumulate(
                    loss, true_false, pred_entropy, t_batch, X_t_batch,
                    polymer_masks, "valid",
                    pdb_dataset.restype_to_int["MAS"]
                )
            #############################################################################################

    t2 = time.time()

    if is_distributed:
        metrics_tensor = torch.tensor(metric_manager.metrics, dtype=torch.float64, device=device)
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
        metric_manager.metrics = metrics_tensor.cpu().numpy()

    metric_manager.compute_metrics()

    train_dt = np.format_float_positional(np.float32(t1-t0), unique=False, precision=3)
    valid_dt = np.format_float_positional(np.float32(t2-t1), unique=False, precision=3)

    output_string = metric_manager.create_print_string(e, total_step, train_dt, valid_dt)

####################### SAVE MODEL AND LOGGING #######################
    if is_distributed:
        dist.barrier()

    if not is_distributed or global_rank == 0:
        with open(logfile, 'a') as f:
            f.write(output_string + "\n")
        if params.get("PRINT_TRAIN_LOSS_ONLY", 0):
            train_row = metric_manager.mask_to_row["train"]
            loss_col = metric_manager.metric_to_col["loss"]
            train_loss = np.format_float_positional(
                np.float32(metric_manager.metrics[train_row, loss_col]),
                unique=False,
                precision=3,
            )
            print(f"epoch: {e+1}, step: {total_step}, train_loss: {train_loss}")
        else:
            print(output_string)
        print(
            "train_batches: "
            f"total={train_batches_total}, "
            f"updated={train_batches_updated}, "
            f"skipped={train_batches_skipped}"
        )

        # wandb: log all metric_manager values, DFM epoch-end records, and grad norm.
        wandb_log = {"epoch": e + 1}
        if grad_norm_count > 0:
            wandb_log["grad_norm_mean"] = grad_norm_sum / grad_norm_count
        for mask_name, mask_row in metric_manager.mask_to_row.items():
            for metric_name, metric_col in metric_manager.metric_to_col.items():
                val = metric_manager.metrics[mask_row, metric_col]
                if not np.isnan(val):
                    wandb_log[f"{mask_name}/{metric_name}"] = float(val)
        if dfm_train_manager is not None:
            gn_mean = (
                np.format_float_positional(
                    np.float32(grad_norm_sum / grad_norm_count), unique=False, precision=3
                ) if grad_norm_count > 0 else "nan"
            )
            train_tail_record = dfm_train_manager.flush(
                total_step, dfm_logfile,
                extra_fields={"grad_norm_mean": gn_mean, "epoch_tail": True}
            )
            valid_record = dfm_valid_manager.flush(total_step, dfm_logfile)
            for record, prefix in ((train_tail_record, "dfm_train"), (valid_record, "dfm_valid")):
                if record is not None:
                    for k, v in record.items():
                        if v is not None and k not in ("step", "window"):
                            wandb_log[f"{prefix}/{k}"] = v
        wandb.log(wandb_log, step=total_step)

        _last_pt = params["BASE_FOLDER"] + 'last.pt'
        _last_pt_tmp = _last_pt + '.tmp'
        torch.save({'epoch': e+1,
                    'step': total_step,
                    'save_step': save_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.optimizer.state_dict(),
                    'ema_state_dict': ema.state_dict(),
                    }, _last_pt_tmp)
        os.replace(_last_pt_tmp, _last_pt)  # atomic on POSIX
        
        if total_step > save_step + params["SAVE_EVERY_N_STEPS"]:
            save_step += params["SAVE_EVERY_N_STEPS"]
            torch.save({'epoch': e+1,
                        'step': total_step,
                        'save_step': save_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.optimizer.state_dict(),
                        'ema_state_dict': ema.state_dict(),
                        }, params["BASE_FOLDER"]+f's_{total_step}.pt')

    if is_distributed:
        dist.barrier()

    if total_step > params["TOTAL_STEPS"]:
        break

if is_distributed:
    dist.destroy_process_group()

if not is_distributed or global_rank == 0:
    wandb.finish()
