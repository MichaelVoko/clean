################################################################################
# Imports
################################################################################
# Python Standard Libraries
import argparse
import ast
import copy
import glob
import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

# Third-Party Libraries
import numpy as np
import pandas as pd

################################################################################
# Local Evaluation Paths (override with environment variables)
################################################################################
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EVALUATION_MODEL_DIR = os.environ.get(
    "NA_EVAL_MODEL_DIR",
    os.path.join(REPO_ROOT, "evaluation", "models")
)

DEFAULT_DSSR_PATH = os.environ.get(
    "NA_EVAL_DSSR_PATH",
    os.path.join(EVALUATION_MODEL_DIR, "dssr", "x3dna-dssr")
)
DEFAULT_ETERNAFOLD_PATH = os.environ.get(
    "NA_EVAL_ETERNAFOLD_PATH",
    os.path.join(EVALUATION_MODEL_DIR, "eternafold", "contrafold")
)
DEFAULT_RIBONANZA_NET_PATH = os.environ.get(
    "NA_EVAL_RIBONANZA_NET_PATH",
    os.path.join(REPO_ROOT, "evaluation", "run_ribonanza_net.py")
)
DEFAULT_RIBONANZA_NET_APPTAINER_PATH = os.environ.get(
    "NA_EVAL_RIBONANZA_NET_APPTAINER_PATH",
    ""
)

DEFAULT_NA_MPNN_APPTAINER_PATH = os.environ.get(
    "NA_EVAL_NA_MPNN_APPTAINER_PATH",
    ""
)
DEFAULT_NA_MPNN_PATH = os.environ.get(
    "NA_EVAL_NA_MPNN_PATH",
    os.path.join(REPO_ROOT, "inference", "run.py")
)
DEFAULT_NA_MPNN_MODEL_PATH = os.environ.get(
    "NA_EVAL_NA_MPNN_MODEL_PATH",
    os.path.join(REPO_ROOT, "models", "design_model", "s_19137.pt")
)
DEFAULT_ARNIE_PATH = os.environ.get(
    "NA_EVAL_ARNIE_PATH",
    os.path.join(EVALUATION_MODEL_DIR, "ribonanzanet")
)
DEFAULT_OPENKNOT_SCORE_PATH = os.environ.get(
    "NA_EVAL_OPENKNOT_SCORE_PATH",
    os.path.join(EVALUATION_MODEL_DIR, "ribonanzanet", "kaggle", "input", "OpenKnotScorePipeline", "src", "openknotscore", "pipeline")
)
DEFAULT_USALIGN_PATH = os.environ.get(
    "NA_EVAL_USALIGN_PATH",
    os.path.join(EVALUATION_MODEL_DIR, "alignment", "USalign")
)

################################################################################
# Common Functions
################################################################################
def read_text_file(path):
    """
    Given a path to a text file, reads the file and returns the contents as a
    string.

    Args:
        path (str): The path to the text file to read.
    
    Returns:
        contents (str): The contents of the file as a string.
    """
    with open(path, mode = "rt") as f:
        contents = f.read()
        return contents

def write_text_file(path, contents):
    """
    Given a path and contents, writes the contents to the file at the given 
    path.

    Args:
        path (str): The path to the file to write.
        contents (str): The contents to write to the file.
    
    Side Effects:
        Writes the contents to the file at the given path.
    """
    with open(path, mode = "wt") as f:
        f.write(contents)

def read_cluster_ids_text_file(path):
    """
    Read a text file containing cluster IDs and return a list of the cluster
    IDs as integers.

    Args:
        path (str): The path to the text file containing cluster IDs.
    
    Returns:
        cluster_ids (int list): A list of the cluster IDs as integers.
    """
    cluster_ids_text = read_text_file(path)
    cluster_ids = cluster_ids_text.strip().split("\n")
    cluster_ids = [int(cluster_id) for cluster_id in cluster_ids]
    return cluster_ids

def read_json_file(path):
    """
    Given a path to a json file, reads the file and returns the contents as a
    dictionary.

    Args:
        path (str): The path to the json file to read.
    
    Returns:
        contents (dict): The contents of the file as a dictionary.
    """
    with open(path, mode = "rt") as f:
        contents = json.load(f)
        return contents

def write_json_file(path, contents):
    """
    Given a path and contents, writes the contents to the file at the given 
    path.

    Args:
        path (str): The path to the file to write.
        contents (dict): The contents to write to the file.
    
    Side Effects:
        Writes the contents to the file at the given path.
    """
    with open(path, mode = "wt") as f:
        json.dump(contents, f, indent = 4)

def read_fasta_file(path):
    """
    Given a path to a fasta file, reads the file and returns a list of tuples,
    where each tuple contains the header and sequence of a fasta entry.

    Args:
        path (str): The path to the fasta file to read.

    Returns:
        fasta_entries ((str, str) list): A list of tuples, where each tuple
            contains the header and sequence of a fasta entry.
    """
    fasta_text = read_text_file(path)

    fasta_text = fasta_text.strip()
    
    if fasta_text.startswith(">"):
        fasta_text = fasta_text[1:]

    fasta_lines = fasta_text.split("\n>")

    fasta_entries = []
    for fasta_line in fasta_lines:
        fasta_line = fasta_line.strip()
        
        fasta_header, fasta_sequence = fasta_line.split("\n", 1)

        fasta_header = fasta_header.strip()
        fasta_sequence = fasta_sequence.strip()

        fasta_entries.append((fasta_header, fasta_sequence))
    
    return fasta_entries

def write_fasta_file(path, fasta_entries):
    """
    Given a path and a list of tuples, where each tuple contains the header and
    sequence of a fasta entry, writes the fasta entries to the file at the given
    path.

    Args:
        path (str): The path to the fasta file to write.
        fasta_entries ((str, str) list): A list of tuples, where each tuple
            contains the header and sequence of a fasta entry.
    
    Side Effects:
        Writes the fasta entries to the file at the given path.
    """
    fasta_lines = []
    for fasta_header, fasta_sequence in fasta_entries:
        fasta_line = f">{fasta_header}\n{fasta_sequence}"
        fasta_lines.append(fasta_line)
    
    fasta_text = "\n".join(fasta_lines)

    write_text_file(path, fasta_text)

def read_cdhit_cluster_file(path):
    """
    Given a path to a CD-HIT cluster file, reads the file and returns a
    dictionary where the keys are the cluster IDs and the values are the
    cluster members.

    Args:
        path (str): The path to the CD-HIT cluster file to read.
    
    Returns:
        clusters (dict): A dictionary where the keys are the cluster IDs and the
            values are the cluster members
    """
    clusters_text = read_text_file(path).strip()
    cluster_entries = clusters_text[1:].split("\n>")
    clusters = dict()
    for cluster_entry in cluster_entries:
        cluster_entry_lines = cluster_entry.strip().split("\n")

        # Extract the cluster id from the header.
        cluster_header_line = cluster_entry_lines[0]
        cluster_id = int(cluster_header_line.strip().split(" ")[1])

        # Extract the cluster members.
        cluster_member_lines = cluster_entry_lines[1:]
        cluster_members = []
        for cluster_member_line in cluster_member_lines:
            member_length, member_entry = \
                cluster_member_line.strip().split(", >")
            member_id, _ = member_entry.split("...")
            cluster_members.append(member_id)

        clusters[cluster_id] = cluster_members
    
    return clusters

def chain_num_to_chain_id(chain_num):
    """
    Given a number chain_num, converts the number to a chain ID of letters.
    This uses "reverse spreadsheet style":
      0, 1, ...
      A, B, ..., Z, AA, BA, CA, ..., ZA, AB, BB, CB, ..., ZB, ...

    Args:
        chain_num (int): The number to convert to a chain ID. i starts at 0.
    
    Returns:
        chain_id (str): The chain ID corresponding to the number.
    """
    alphabet_length = 26
    
    # This algorithm is similar to converting to base 26, but we need to
    # subtract 1 from the number since mapping A to 0 base 26 results in some
    # issues (e.g. if A = 0 base 26, then AA = 00 base 26, which is not 
    # correct).
    chain_letter_list = []
    while chain_num >= 0:
        chain_letter_list.append(chr(ord("A") + (chain_num % alphabet_length))) 
        chain_num = (chain_num // 26) - 1

    chain_id = "".join(chain_letter_list)
    return chain_id


################################################################################
# Constants
################################################################################
class NAConstants:
    # 1 letter codes for RNA residues.
    rna_restypes = [
        "A",
        "C",
        "G",
        "U",
    ]
    rna_restype_to_int = dict(zip(rna_restypes, range(len(rna_restypes))))

    # Unknown residues.
    rna_unknown_restype = "X"
    dssr_unknown_restype = "?"

    # Chain break characters.
    chain_break_character = "/"
    dssr_chain_break_character = "&"

    # DSSR represents modifications of residues with the lower case of their
    # base residue.
    dssr_modified_restypes = [rna_restype.lower() for rna_restype in rna_restypes]

    # NA-MPNN RNA residue type mapping.
    na_mpnn_rna_restype_to_rna_restype = {
        "b": "A",
        "d": "C",
        "h": "G",
        "u": "U",
        "y": "X"
    }

    # NA-MPNN na shared token representation.
    na_mpnn_na_shared_tokens = True

    # NA-MPNN residue type ordering.
    na_mpnn_restypes = [
        'ALA',
        'ARG',
        'ASN',
        'ASP',
        'CYS',
        'GLN',
        'GLU',
        'GLY',
        'HIS',
        'ILE',
        'LEU',
        'LYS',
        'MET',
        'PHE',
        'PRO',
        'SER',
        'THR',
        'TRP',
        'TYR',
        'VAL',
        'UNK',
        'DA',
        'DC',
        'DG',
        'DT',
        'DX',
        'A',
        'C',
        'G',
        'U',
        'RX',
        'MAS',
        'PAD'
    ]

    # NA-MPNN residue type to int mapping.
    na_mpnn_restype_to_int = dict(zip(na_mpnn_restypes, range(len(na_mpnn_restypes))))
    na_mpnn_int_to_restype = dict(zip(range(len(na_mpnn_restypes)), na_mpnn_restypes))

    if na_mpnn_na_shared_tokens:
        na_mpnn_restype_to_int["A"] = na_mpnn_restype_to_int["DA"]
        na_mpnn_restype_to_int["C"] = na_mpnn_restype_to_int["DC"]
        na_mpnn_restype_to_int["G"] = na_mpnn_restype_to_int["DG"]
        na_mpnn_restype_to_int["U"] = na_mpnn_restype_to_int["DT"]
        na_mpnn_restype_to_int["RX"] = na_mpnn_restype_to_int["DX"]
    
    # DeepPBS restype ordering.
    deep_pbs_restypes = [
        "DA",
        "DC",
        "DG",
        "DT"
    ]

    # DeepPBS restype to int mapping.
    deep_pbs_restype_to_int = dict(zip(deep_pbs_restypes, range(len(deep_pbs_restypes))))
    deep_pbs_int_to_restype = dict(zip(range(len(deep_pbs_restypes)), deep_pbs_restypes))

    # Min overlap length for ppm alignment.
    min_overlap_length = 5

    # 2D structure symbols for RNA.
    pair_symbols_list = [
        ("(", ")"),
        ("[", "]"),
        ("{", "}"),
        ("<", ">"),
        ("A", "a"),
        ("B", "b"),
        ("C", "c"),
        ("D", "d"),
        ("E", "e"),
        ("E", "e"),
        ("F", "f"),
        ("G", "g"),
        ("H", "h"),
        ("I", "i"),
        ("J", "j"),
        ("K", "k"),
        ("L", "l"),
        ("M", "m"),
        ("N", "n"),
        ("O", "o"),
        ("P", "p"),
        ("Q", "q"),
        ("R", "r"),
        ("S", "s"),
        ("T", "t"),
        ("U", "u"),
        ("V", "v"),
        ("W", "w"),
        ("X", "x"),
        ("Y", "y"),
        ("Z", "z"),
    ]

    # Create lists of open, close, and loop symbols.
    open_symbols = [pair_symbols[0] for pair_symbols in pair_symbols_list]
    close_symbols = [pair_symbols[1] for pair_symbols in pair_symbols_list]
    loop_symbols = [".", ","]

    # Create dictionaries to map open symbols to close symbols and vice versa.
    open_to_close = {pair_symbols[0]: pair_symbols[1] for pair_symbols in pair_symbols_list}
    close_to_open = {pair_symbols[1]: pair_symbols[0] for pair_symbols in pair_symbols_list}

################################################################################
# Sequence and Structure Standardization
################################################################################
def check_rna_sequence_validity(sequence, 
                                unknown_residue_allowed,
                                chain_breaks_allowed):
    """
    Given an rna sequence, checks the validity of the sequence.

    Args:
        sequence (str): The RNA sequence to check.
        unknown_residue_allowed (bool): Whether unknown residues are allowed in
            the sequence.
        chain_breaks_allowed (bool): Whether chain breaks are allowed in the
            sequence.
    
    Side Effects:
        Raises a ValueError if the sequence is invalid.
    """
    for c in sequence:
        if c in NAConstants.rna_restype_to_int:
            continue
        elif unknown_residue_allowed and c == NAConstants.rna_unknown_restype:
            continue
        elif chain_breaks_allowed and c == NAConstants.chain_break_character:
            continue
        else:
            raise ValueError(f"Invalid character in sequence: {c}")   

def standardize_rna_sequence(sequence, 
                             method = None,
                             remove_chain_breaks = False):
    """
    Given an RNA sequence, standardizes the sequence to a canonical form.

    NOTE: This method is only intended for use with RNA sequences.

    Args:
        sequence (str): The RNA sequence to standardize.
        method (str): The method to use for standardization.
            Options:
                "na_mpnn": Standardize the sequence using the NA-MPNN RNA
                    residue type mapping.
                "dssr": Standardize the sequence using the DSSR unknown
                    residue and chain break characters.
                None: no standardization.
        remove_chain_breaks (bool): Whether to remove chain breaks from the
            sequence. 
            NOTE: This option should only be True if the user is certain that 
                the sequence does not contain any chain breaks and that the
                presence of any chain breaks is an error.
    
    Returns:
        standard_sequence (str): The standardized RNA sequence.
    """
    standard_sequence = []

    # Standardize the sequence.
    for c in sequence:
        # Convert the bdhuy characters from NA-MPNN to ACGUX.
        if method == "na_mpnn" and \
           c in NAConstants.na_mpnn_rna_restype_to_rna_restype:
            standard_sequence.append(NAConstants.na_mpnn_rna_restype_to_rna_restype[c])
        # Standardize the dssr unknown residue.
        elif method == "dssr" and c == NAConstants.dssr_unknown_restype:
            standard_sequence.append(NAConstants.rna_unknown_restype)
        # Standardize the dssr chain break character.
        elif method == "dssr" and c == NAConstants.dssr_chain_break_character:
            standard_sequence.append(NAConstants.chain_break_character)
        # DSSR represents modifications of residues with the lower case of their
        # base residue. We convert them to the unknown residue.
        elif method == "dssr" and c in NAConstants.dssr_modified_restypes:
            standard_sequence.append(NAConstants.rna_unknown_restype)
        else:
            standard_sequence.append(c.upper())

    # Remove chain breaks if specified.
    if remove_chain_breaks:
        standard_sequence = [c for c in standard_sequence if c != NAConstants.chain_break_character]
    
    standard_sequence = "".join(standard_sequence)

    # Check the validity of the standard sequence.
    check_rna_sequence_validity(standard_sequence, 
                                unknown_residue_allowed = True,
                                chain_breaks_allowed = True)

    return standard_sequence

def check_secondary_structure_validity(secondary_structure):
    """
    Given a secondary structure string, checks the validity of the secondary
    structure string. 

    Args:
        secondary_structure (str): The secondary structure string.
    
    Side Effects:
        Raises a ValueError if the secondary structure string is invalid.
    """
    calculate_base_pairs_and_loops_from_secondary_structure(secondary_structure)

def standardize_secondary_structure(secondary_structure,
                                    method = None,
                                    replace_unknown_restypes = False,
                                    remove_chain_breaks = False):
    """
    Given a secondary structure string, standardizes the secondary structure
    to a canonical form.

    NOTE: This method is only intended for use with NA secondary structure.

    Args:
        secondary_structure (str): The secondary structure string to 
            standardize.
        method (str): The method to use for standardization.
            Options:
                "dssr": Standardize the secondary structure using the DSSR
                    unknown residue and chain break characters.
                None: no standardization.
        replace_unknown_restypes (bool): Whether to replace unknown residues
            with loop symbols in the secondary structure. This option is only
            valid if method is "dssr". This option should only be True if the
            user is certain that the secondary structure does not contain any
            unknown residues and that the presence of any unknown residues is an
            error.
        remove_chain_breaks (bool): Whether to remove chain breaks from the
            secondary structure. This option is only valid if method is "dssr".
            This option should only be True if the user is certain that the
            secondary structure does not contain any chain breaks and that the
            presence of any chain breaks is an error.
    """
    standard_secondary_structure = []

    # Standardize the secondary structure.
    for c in secondary_structure:
        if method == "dssr" and \
           replace_unknown_restypes and \
           c == NAConstants.dssr_unknown_restype:
            standard_secondary_structure.append(NAConstants.loop_symbols[0])
        elif method == "dssr" and \
             remove_chain_breaks and \
             c == NAConstants.dssr_chain_break_character:
            continue
        else:
            standard_secondary_structure.append(c)
    
    standard_secondary_structure = "".join(standard_secondary_structure)

    # Check the validity of the standard secondary structure.
    check_secondary_structure_validity(standard_secondary_structure)

    return standard_secondary_structure

################################################################################
# Structure to Sequence and Secondary Structure
################################################################################
def run_dssr(structure_path, 
             dssr_path = "/rds/user/mh2167/hpc-work/NA-MPNN/evaluation/models/dssr/bin/x3dna-dssr"):
    """
    Given a path to a tertiary structure file containing nucleic acid, runs the
    DSSR algorithm to extract the nucleic acid sequence and determine the
    nucleic acid secondary structure.

    Args:
        structure_path (str): The path to the tertiary structure file.
        dssr_path (str): The path to the DSSR executable.
    
    Returns:
        result (dict): A dictionary containing:
            sequence (str): The nucleic acid sequence from the tertiary 
                structure.
            secondary_structure (str): The nucleic acid secondary structure from 
                the tertiary structure.
    """
    # Turn the structure_path into an absolute path.
    structure_path = os.path.abspath(structure_path)

    # Check that the structure_path exists.
    if not os.path.exists(structure_path):
        raise ValueError(f"Invalid structure path: {structure_path}")

    # Get the file name of the structure path (removing extension).
    structure_name = os.path.splitext(os.path.basename(structure_path))[0]

    # Create a temporary directory for the outputs, and ensure it gets removed
    # on script exit.
    tmp_directory = tempfile.TemporaryDirectory()

    # Compute the paths for the output files.
    out_path = os.path.join(tmp_directory.name, f"{structure_name}.out")
    dbn_path = os.path.join(tmp_directory.name, f"{structure_name}-2ndstrs.dbn")

    # Run the DSSR algorithm.
    try:
        subprocess.run(
            [
                str(dssr_path),
                f"-i={structure_path}",
                f"-o={out_path}",
                f"--prefix={structure_name}"
            ], 
            check = True,
            cwd = tmp_directory.name,
            stdout = subprocess.DEVNULL,
            stderr = subprocess.DEVNULL
        )

        # Read the dbn file.
        dbn_text = read_text_file(dbn_path)

        # Extract the sequence string.
        sequence = dbn_text.split("\n")[1]

        # Extract the secondary structure string.
        secondary_structure = dbn_text.split("\n")[2]

        tmp_directory.cleanup()

        result = {
            "sequence": sequence,
            "secondary_structure": secondary_structure
        }

        return result
    except subprocess.CalledProcessError as e:
        tmp_directory.cleanup()
        raise e

################################################################################
# Sequence to Predicted Secondary Structure and Reactivity Profile
################################################################################
def run_eternafold(sequence,
                   eternafold_path = "evaluation/models/eternafold/EternaFold/src/contrafold"):
    """
    Given a sequence, run the EternaFold algorithm to predict the secondary
    structure of the sequence.

    Args:
        sequence (str): The sequence to predict the secondary structure for.
        eternafold_path (str): The path to the EternaFold executable.

    Returns:
        result (dict): A dictionary containing:
            predicted_secondary_structure (str): The predicted secondary 
                structure of the sequence.
    """
    # Check that the RNA sequence is valid.
    check_rna_sequence_validity(sequence, 
                                unknown_residue_allowed = False, 
                                chain_breaks_allowed = False)

    # Create the input and output files for EternaFold.
    eternafold_input_file = tempfile.NamedTemporaryFile(mode = "wt")
    eternafold_output_file = tempfile.NamedTemporaryFile(mode = "wt")

    # Write the sequence to the input file.
    eternafold_input_file.write(sequence)
    eternafold_input_file.flush()

    # Run EternaFold.
    try:
        subprocess.run(
            [
                str(eternafold_path),
                "predict",
                eternafold_input_file.name
            ],
            check = True,
            stdout = eternafold_output_file,
            stderr = subprocess.DEVNULL
        )

        eternafold_output_text = read_text_file(eternafold_output_file.name)

        # Extract the predicted secondary structure from the EternaFold output.
        eternafold_output_lines = eternafold_output_text.strip().split("\n")

        # The predicted secondary structure is the last line of the output.
        predicted_secondary_structure = eternafold_output_lines[-1]

        eternafold_input_file.close()
        eternafold_output_file.close()

        result = {
            "predicted_secondary_structure": predicted_secondary_structure
        }

        return result
    except (subprocess.CalledProcessError, ValueError) as e:
        eternafold_input_file.close()
        eternafold_output_file.close()
        raise e

def run_ribonanza_net_reactivity_profile(sequence,
                                         batch_size = 1,
                                         ribonanza_net_apptainer_path = "evaluation/models/ribonanzanet/ribonanza.sif",
                                         ribonanza_net_path = "evaluation/run_ribonanza_net.py"):
    """
    Given a sequence, runs the RibonanzaNet algorithm to predict the reactivity
    profile of the sequence.

    Args:
        sequence (str): The sequence to predict the reactivity profile for.
        batch_size (int): The number of samples to predict in a batch.
        ribonanza_net_apptainer_path (str): The path to the RibonanzaNet
            apptainer for running RibonanzaNet.
        ribonanza_net_path (str): The path to the RibonanzaNet run file.
    
    Returns:
        result (dict): A dictionary containing:
            predicted_2A3_reactivity_profiles (list of float lists): A list of
                predicted reactivity profiles of the sequence for the 2A3 probe.
            predicted_DMS_reactivity_profiles (list of float lists): A list of
                predicted reactivity profiles of the sequence for the DMS probe.
    """    
    # Check that the RNA sequence is valid.
    check_rna_sequence_validity(sequence,
                                unknown_residue_allowed = False,
                                chain_breaks_allowed = False)
    
    # Create a temporary directory for the outputs, and ensure it gets removed
    # on script exit.
    tmp_directory = tempfile.TemporaryDirectory()

    # Compute the paths for t
    # Compute the paths for the output files.
    out_path = os.path.join(tmp_directory.name, "output.npy")

    # Run the RibonanzaNet algorithm to predict the reactivity profile.
    try:
        if str(ribonanza_net_apptainer_path):
            command = [
                "apptainer",
                "exec",
                "--bind", "evaluation/models/ribonanzanet/arnie:/arnie",
                "--env", "PYTHONPATH=/arnie",
                str(ribonanza_net_apptainer_path),
                "python",
                str(ribonanza_net_path),
                "reactivity_profile",
                str(sequence),
                str(tmp_directory.name),
                str(batch_size)
            ]
        else:
            command = [
                sys.executable,
                str(ribonanza_net_path),
                "reactivity_profile",
                str(sequence),
                str(tmp_directory.name),
                str(batch_size)
            ]

        subprocess.run(
            command,
            check = True,
            stdout = subprocess.DEVNULL,
            stderr = subprocess.DEVNULL
        )

        # Read the output file.
        out_dict = np.load(out_path, allow_pickle = True).item()

        # Extract the predicted reactivity profiles.
        result = {
            "predicted_2A3_reactivity_profiles": out_dict["predicted_2A3_reactivity_profiles"],
            "predicted_DMS_reactivity_profiles": out_dict["predicted_DMS_reactivity_profiles"]
        }

        # Clean up the temporary directory.
        tmp_directory.cleanup()

        return result
    except subprocess.CalledProcessError as e:
        tmp_directory.cleanup()
        raise e

def run_ribonanza_net_secondary_structure(sequence,
                                          batch_size = 1,
                                          ribonanza_net_apptainer_path = "evaluation/models/ribonanzanet/ribonanza.sif",
                                          ribonanza_net_path = "evaluation/run_ribonanza_net.py"):
    """
    Given a sequence, runs the RibonanzaNet algorithm to predict the secondary
    structure of the sequence.

    Args:
        sequence (str): The sequence to predict the secondary structure for.
        batch_size (int): The number of samples to predict in a batch.
        ribonanza_net_apptainer_path (str): The path to the RibonanzaNet
            apptainer for running RibonanzaNet.
        ribonanza_net_path (str): The path to the RibonanzaNet run file.
    
    Returns:
        result (dict): A dictionary containing:
            predicted_secondary_structures (str list): The predicted secondary
                structures of the sequence.
    """    
    # Check that the RNA sequence is valid.
    check_rna_sequence_validity(sequence,
                                unknown_residue_allowed = False,
                                chain_breaks_allowed = False)
    
    # Create a temporary directory for the outputs, and ensure it gets removed
    # on script exit.
    tmp_directory = tempfile.TemporaryDirectory()

    # Compute the paths for the output files.
    out_path = os.path.join(tmp_directory.name, "output.npy")

    # Run the RibonanzaNet algorithm to predict the secondary structure.
    try:
        if str(ribonanza_net_apptainer_path):
            command = [
                "apptainer",
                "exec",
                "--bind", "evaluation/models/ribonanzanet/arnie:/arnie",
                "--env", "PYTHONPATH=/arnie",
                str(ribonanza_net_apptainer_path),
                "python",
                str(ribonanza_net_path),
                "secondary_structure",
                str(sequence),
                str(tmp_directory.name),
                str(batch_size)
            ]
        else:
            command = [
                sys.executable,
                str(ribonanza_net_path),
                "secondary_structure",
                str(sequence),
                str(tmp_directory.name),
                str(batch_size)
            ]

        subprocess.run(
            command,
            check = True,
            stdout = subprocess.DEVNULL,
            stderr = subprocess.DEVNULL
        )

        # Read the output file.
        out_dict = np.load(out_path, allow_pickle = True).item()

        # Extract the predicted secondary structures.
        result = {
            "predicted_secondary_structures": out_dict["predicted_secondary_structures"]
        }

        # Clean up the temporary directory.
        tmp_directory.cleanup()

        return result
    except subprocess.CalledProcessError as e:
        tmp_directory.cleanup()
        raise e

################################################################################
# Sequence to Predicted Structure
################################################################################
def run_alphafold3(name, 
                   sequences_and_polytypes,
                   output_dir,
                   num_diffusion_samples = 5,
                   num_seeds = 1,
                   fixed_seeds = None,
                   run_data_pipeline = False,
                   buckets = "1",
                   alphafold3_apptainer_path = "evaluation/models/alphafold3/docker/alphafold3_amd64.sif",
                   alphafold3_path = "evaluation/models/alphafold3/run_alphafold.py",
                   model_dir = "evaluation/models/alphafold3/model_dir"):
    """
    Given a name, a list of sequences and polytypes, and an output directory,
    runs AlphaFold3 to predict the structure of the complex.

    Args:
        name (str): A name of the complex.
        sequences_and_polytypes ((str, str) list): A list of tuples, where each
            tuple contains the sequence and polytype of a sequence.
        output_dir (str): The path to the output directory.
        num_diffusion_samples (int): The number of diffusion samples to 
            generate. Default is 5.
        num_seeds (int): The number of model seeds to generate and use. Default
            is 1. This argument is mutually exclusive with fixed_seeds.
        fixed_seeds (int list): A list of fixed seeds to use for the model. This
            argument is mutually exclusive with num_seeds.
        run_data_pipeline (bool): Whether to run the data pipeline (whether to
            perform the MSA and templates searches).
        buckets (str): A comma separated list of integers. Strictly increasing 
            order of token sizes for which to cache compilations. For any input 
            with more tokens than the largest bucket size, a new bucket is 
            created for exactly that number of tokens. The "1" bucket is a 
            trick to ensure no padding occurs; although if running batches could
            cause a lot of model recompilation. The alphafold3 default is
            "256,512,768,1024,1280,1536,2048,2560,3072,3584,4096,4608,5120".
        alphafold3_apptainer_path (str): The path to the AlphaFold3 apptainer
            for running AlphaFold3.
        alphafold3_path (str): The path to the AlphaFold3 run file.
        model_dir (str): The path to the AlphaFold3 model directory.
    
    Returns:
        result (dict): A dictionary containing:
            json_input_path (str): The path to the input JSON file.
            predicted_structure_path (str): The path to the predicted structure
                file.
            predicted_confidences_path (str): The path to the predicted
                confidences file.
            summary_confidences_path (str): The path to the summary confidences
                file.
            ptm (float): The predicted PTM score.
            plddt (float): The predicted pLDDT score.
            pae (float): The predicted pAE score.
    """
    # Check that both num_seeds and fixed_seeds are not set.
    if num_seeds is not None and fixed_seeds is not None:
        raise ValueError("Both num_seeds and fixed_seeds cannot be set at the same time.")

    # If the output directory for the specified name already exists,
    # raise an error.
    name_output_directory = os.path.join(output_dir, name)

    # Check if AF3 already ran successfully (resume support).
    _af3_done = (
        os.path.exists(os.path.join(name_output_directory, f"{name}_data.json")) and
        os.path.exists(os.path.join(name_output_directory, f"{name}_model.cif")) and
        os.path.exists(os.path.join(name_output_directory, f"{name}_confidences.json")) and
        os.path.exists(os.path.join(name_output_directory, f"{name}_summary_confidences.json"))
    )

    # Prepare the model seed input.
    if fixed_seeds is not None:
        model_seeds = fixed_seeds
    else:
        # Generate random seeds.
        seed_rng = np.random.default_rng()
        model_seeds = [int(seed_rng.integers(0, 2 ** 32 - 1)) for i in range(num_seeds)]

    # Prepare the sequences input, with no MSA or template.
    sequences_input = []
    for i, (sequence, polytype) in enumerate(sequences_and_polytypes):
        sequences_entry_dict = {
            polytype: {
                "id": chain_num_to_chain_id(i),
                "sequence": sequence,
                "unpairedMsa": ""
            }
        }
        sequences_input.append(sequences_entry_dict)

    alphafold3_input_json_dict = {
        "dialect": "alphafold3",
        "version": 3,
        "name": name,
        "modelSeeds": model_seeds,
        "sequences": sequences_input
    }
    
    # Set up the input JSON file.
    temp_json_file = tempfile.NamedTemporaryFile(mode = "wt", suffix = ".json")

    # Write the input JSON file.
    write_json_file(temp_json_file.name, alphafold3_input_json_dict)

    # Run AlphaFold3 (skip if outputs already exist).
    if not _af3_done:
        try:
            if str(alphafold3_apptainer_path):
                command = [
                    "apptainer",
                    "exec",
                    "--nv",
                    alphafold3_apptainer_path,
                    "python",
                    alphafold3_path
                ]
            else:
                command = [
                    sys.executable,
                    alphafold3_path
                ]

            command.extend([
                f"--model_dir={model_dir}",
                f"--run_data_pipeline={run_data_pipeline}",
                f"--buckets={buckets}",
                f"--num_diffusion_samples={num_diffusion_samples}",
                f"--output_dir={output_dir}",
                f"--json_path={temp_json_file.name}",
                "--force_output_dir=True",
            ])

            subprocess.run(
                command,
                check = True
            )
        except (subprocess.CalledProcessError, ValueError) as e:
            temp_json_file.close()
            raise e

    # Close the temporary file.
    temp_json_file.close()

    # Process the outputs.
    json_input_path = os.path.join(name_output_directory, f"{name}_data.json")

    # AF3 writes top-level aggregated files (best-ranked model).
    predicted_structure_path = os.path.join(name_output_directory, f"{name}_model.cif")
    predicted_confidences_path = os.path.join(name_output_directory, f"{name}_confidences.json")
    summary_confidences_path = os.path.join(name_output_directory, f"{name}_summary_confidences.json")

    # Check that the output files exist.
    if not os.path.exists(json_input_path):
        raise ValueError(f"Output JSON file not found: {json_input_path}")
    if not os.path.exists(predicted_structure_path):
        raise ValueError(f"Predicted structure file not found: {predicted_structure_path}")
    if not os.path.exists(predicted_confidences_path):
        raise ValueError(f"Predicted confidences file not found: {predicted_confidences_path}")
    if not os.path.exists(summary_confidences_path):
        raise ValueError(f"Summary confidences file not found: {summary_confidences_path}")
    # Extract confidence scores.
    summary_confidences_dict = read_json_file(summary_confidences_path)
    ptm = summary_confidences_dict["ptm"]

    predicted_confidences_dict = read_json_file(predicted_confidences_path)

    atom_plddts = predicted_confidences_dict["atom_plddts"]
    plddt = np.mean(atom_plddts)

    pae_matrix = predicted_confidences_dict["pae"]
    pae = np.mean(pae_matrix)

    result = {
        "json_input_path": json_input_path,
        "predicted_structure_path": predicted_structure_path,
        "predicted_confidences_path": predicted_confidences_path,
        "summary_confidences_path": summary_confidences_path,
        "ptm": ptm,
        "plddt": plddt,
        "pae": pae
    }
    
    return result


def prepare_alphafold3_input(name,
                             sequences_and_polytypes,
                             staging_dir,
                             num_seeds = 1,
                             fixed_seeds = None):
    """
    Prepare an AF3 input JSON file and write it to staging_dir.
    This is the input-preparation half of run_alphafold3.

    Args:
        name (str): A name of the complex.
        sequences_and_polytypes ((str, str) list): A list of tuples, where each
            tuple contains the sequence and polytype of a sequence.
        staging_dir (str): The directory to write the input JSON file to.
        num_seeds (int): The number of model seeds to generate and use. Default
            is 1. This argument is mutually exclusive with fixed_seeds.
        fixed_seeds (int list): A list of fixed seeds to use for the model. This
            argument is mutually exclusive with num_seeds.

    Returns:
        json_path (str): The path to the written input JSON file.
    """
    # Check that both num_seeds and fixed_seeds are not set.
    if num_seeds is not None and fixed_seeds is not None:
        raise ValueError("Both num_seeds and fixed_seeds cannot be set at the same time.")

    # Prepare the model seed input.
    if fixed_seeds is not None:
        model_seeds = fixed_seeds
    else:
        # Generate random seeds.
        seed_rng = np.random.default_rng()
        model_seeds = [int(seed_rng.integers(0, 2 ** 32 - 1)) for i in range(num_seeds)]

    # Prepare the sequences input, with no MSA or template.
    sequences_input = []
    for i, (sequence, polytype) in enumerate(sequences_and_polytypes):
        sequences_entry_dict = {
            polytype: {
                "id": chain_num_to_chain_id(i),
                "sequence": sequence,
                "unpairedMsa": ""
            }
        }
        sequences_input.append(sequences_entry_dict)

    alphafold3_input_json_dict = {
        "dialect": "alphafold3",
        "version": 3,
        "name": name,
        "modelSeeds": model_seeds,
        "sequences": sequences_input
    }

    # Write the input JSON file to the staging directory.
    json_path = os.path.join(staging_dir, f"{name}.json")
    write_json_file(json_path, alphafold3_input_json_dict)
    return json_path


def run_alphafold3_batch(staging_dir, output_dir,
                         num_diffusion_samples = 5,
                         run_data_pipeline = False,
                         buckets = "256,512,768,1024,1280,1536,2048",
                         alphafold3_apptainer_path = "evaluation/models/alphafold3/docker/alphafold3_amd64.sif",
                         alphafold3_path = "evaluation/models/alphafold3/run_alphafold.py",
                         model_dir = "evaluation/models/alphafold3/model_dir"):
    """
    Run AlphaFold3 on all input JSONs in staging_dir with a single invocation,
    using --input_dir instead of --json_path.

    Args:
        staging_dir (str): Directory containing AF3 input JSON files (written
            by prepare_alphafold3_input).
        output_dir (str): Directory where AF3 writes its outputs (one
            subdirectory per input, named after the input's "name" field).
        num_diffusion_samples (int): The number of diffusion samples to
            generate. Default is 5.
        run_data_pipeline (bool): Whether to run the data pipeline (whether to
            perform the MSA and templates searches).
        buckets (str): A comma separated list of integers for compilation
            caching. Wider buckets reduce recompilation when sequence lengths
            vary within a batch.
        alphafold3_apptainer_path (str): The path to the AlphaFold3 apptainer
            for running AlphaFold3.
        alphafold3_path (str): The path to the AlphaFold3 run file.
        model_dir (str): The path to the AlphaFold3 model directory.
    """
    if str(alphafold3_apptainer_path):
        command = [
            "apptainer",
            "exec",
            "--nv",
            alphafold3_apptainer_path,
            "python",
            alphafold3_path
        ]
    else:
        command = [
            sys.executable,
            alphafold3_path
        ]

    command.extend([
        f"--model_dir={model_dir}",
        f"--run_data_pipeline={run_data_pipeline}",
        f"--buckets={buckets}",
        f"--num_diffusion_samples={num_diffusion_samples}",
        f"--output_dir={output_dir}",
        f"--input_dir={staging_dir}",
        "--force_output_dir=True",
    ])

    subprocess.run(
        command,
        check = True
    )


def parse_alphafold3_output(name, output_dir):
    """
    Parse AF3 output files for a given sample. This is the output-parsing
    half of run_alphafold3.

    Args:
        name (str): The sample name (AF3 writes to output_dir/name/).
        output_dir (str): The AF3 output directory.

    Returns:
        result (dict): Same schema as run_alphafold3.
    """
    name_output_directory = os.path.join(output_dir, name)

    # AF3 writes top-level aggregated files (best-ranked model).
    json_input_path = os.path.join(name_output_directory, f"{name}_data.json")
    predicted_structure_path = os.path.join(name_output_directory, f"{name}_model.cif")
    predicted_confidences_path = os.path.join(name_output_directory, f"{name}_confidences.json")
    summary_confidences_path = os.path.join(name_output_directory, f"{name}_summary_confidences.json")

    # Check that the output files exist.
    if not os.path.exists(json_input_path):
        raise ValueError(f"Output JSON file not found: {json_input_path}")
    if not os.path.exists(predicted_structure_path):
        raise ValueError(f"Predicted structure file not found: {predicted_structure_path}")
    if not os.path.exists(predicted_confidences_path):
        raise ValueError(f"Predicted confidences file not found: {predicted_confidences_path}")
    if not os.path.exists(summary_confidences_path):
        raise ValueError(f"Summary confidences file not found: {summary_confidences_path}")

    # Extract confidence scores.
    summary_confidences_dict = read_json_file(summary_confidences_path)
    ptm = summary_confidences_dict["ptm"]

    predicted_confidences_dict = read_json_file(predicted_confidences_path)

    atom_plddts = predicted_confidences_dict["atom_plddts"]
    plddt = np.mean(atom_plddts)

    pae_matrix = predicted_confidences_dict["pae"]
    pae = np.mean(pae_matrix)

    result = {
        "json_input_path": json_input_path,
        "predicted_structure_path": predicted_structure_path,
        "predicted_confidences_path": predicted_confidences_path,
        "summary_confidences_path": summary_confidences_path,
        "ptm": ptm,
        "plddt": plddt,
        "pae": pae
    }

    return result


################################################################################
# Sequence Comparison
################################################################################
def calculate_sequence_recovery(reference_sequence, 
                                subject_sequence,
                                chain_breaks_allowed = False,
                                unknown_residue_allowed_in_reference = False):
    """
    Given a reference sequence and a subject sequence, calculates the sequence 
    recovery of the subject sequence.

    Args:
        reference_sequence (str): The reference sequence to calculate the
            sequence recovery against.
        subject_sequence (str): The sequence to calculate the sequence recovery 
            for.
        chain_breaks_allowed (bool): Whether chain breaks are allowed in the
            sequence.
        unknown_residue_allowed_in_reference (bool): Whether unknown residues
            are allowed in the reference sequence.
    
    Returns:
        result (dict): A dictionary containing:
            sequence_recovery (float): The sequence recovery of the sequence.
    """
    # Check that the subject sequence and reference sequence have the same 
    # length.
    if len(subject_sequence) != len(reference_sequence):
        raise ValueError(f"Length of subject sequence ({len(subject_sequence)}) must match length of reference sequence ({len(reference_sequence)}).")
    
    # Check the validity of the subject sequence.
    check_rna_sequence_validity(subject_sequence,
                                unknown_residue_allowed = False,
                                chain_breaks_allowed = chain_breaks_allowed)

    # Check the validity of the reference sequence.
    check_rna_sequence_validity(reference_sequence,
                                unknown_residue_allowed = unknown_residue_allowed_in_reference,
                                chain_breaks_allowed = chain_breaks_allowed)
    
    # Calculate the number of correct residues.
    num_correct = 0
    num_residues = 0
    for subject_residue, reference_residue in zip(subject_sequence, reference_sequence):
        # Skip unknown residues in the reference sequence.
        if unknown_residue_allowed_in_reference and \
           reference_residue == NAConstants.rna_unknown_restype:
            continue
        # Skip chain breaks if they occur in both sequences.
        elif chain_breaks_allowed and \
           (subject_residue == NAConstants.chain_break_character or \
            reference_residue == NAConstants.chain_break_character):
            if not (subject_residue == NAConstants.chain_break_character and \
                    reference_residue == NAConstants.chain_break_character):
                raise ValueError("Chain breaks must occur at the same position in both sequences.")
            continue
        else:
            num_residues += 1
            if subject_residue == reference_residue:
                num_correct += 1

    # Calculate the sequence recovery.
    if num_residues == 0:
        raise ValueError("Number of residues must be greater than 0.")
    
    sequence_recovery = num_correct / num_residues

    result = {
        "sequence_recovery": sequence_recovery
    }

    return result

################################################################################
# Secondary Structure and Reactivity Profile Comparison
################################################################################
def calculate_base_pairs_and_loops_from_secondary_structure(secondary_structure):
    """
    Given a secondary structure string, calculates the base pair and loop 
    indices. Note, this function can also be used to check the validity of
    secondary structure strings.

    Args:
        secondary_structure (str): The secondary structure string.
    
    Returns:
        pairs_indices (int tuple list): A list of tuples, where each tuple
            contains the indices of a base pair.
        loop_indices (int list): A list of loop indices.
    """
    # Check that the secondary structure only contains valid characters.
    for c in secondary_structure:
        if c not in NAConstants.open_symbols and \
           c not in NAConstants.close_symbols and \
           c not in NAConstants.loop_symbols:
            raise ValueError(f"Invalid character in secondary structure: {c}")
    
    # Check that the number of open and close symbols are equal.
    num_opens = len([c for c in secondary_structure if c in NAConstants.open_symbols])
    num_closes = len([c for c in secondary_structure if c in NAConstants.close_symbols])
    if num_opens != num_closes:
        raise ValueError(f"Number of open ({num_opens}) and close ({num_closes}) symbols must be equal.")

    pairs_indices = []
    loop_indices = []
    open_symbol_stacks = {open_symbol: [] for open_symbol in NAConstants.open_symbols}
    for i, c in enumerate(secondary_structure):
        # If the symbol is an open symbol, record the index.
        if c in NAConstants.open_symbols:
            open_symbol_stacks[c].append(i)
        # If the symbol is a close symbol, pop the last corresponding open
        # symbol index and record the pair.
        elif c in NAConstants.close_symbols:
            # Get the corresponding open symbol.
            open_symbol = NAConstants.close_to_open[c]

            # Check that there is a corresponding open symbol.
            if len(open_symbol_stacks[open_symbol]) == 0:
                raise ValueError(f"No matching open symbol for close symbol at index {i}.")
            
            # Get the index of the last corresponding open symbol.
            open_index = open_symbol_stacks[open_symbol].pop()

            # Record the pair.
            close_index = i
            pairs_indices.append((open_index, close_index))
        # If the symbol is a loop symbol, record the index.
        elif c in NAConstants.loop_symbols:
            loop_indices.append(i)
        else:
            raise ValueError(f"Invalid character in secondary structure: {c}")
    
    # Check that all open symbols have been closed.
    for open_symbol, open_indices in open_symbol_stacks.items():
        if len(open_indices) > 0:
            raise ValueError(f"No matching close symbol ({NAConstants.open_to_close[open_symbol]}) for open symbol ({open_symbol}) at indices {open_indices}.")

    return pairs_indices, loop_indices

def calculate_secondary_structure_stats(reference_secondary_structure, 
                                        subject_secondary_structure):
    """
    Given a reference secondary structure and a subject secondary structure, 
    calculates the F1 score for the base pairs and loops of the subject.

    Args:
        reference_secondary_structure (str): The reference secondary structure.
        subject_secondary_structure (str): The secondary structure.

    Returns:
        result (dict): A dictionary containing:
            f1_score_pairs (float): The F1 score for the base pairs.
            f1_score_loops (float): The F1 score for the loops.
    """
    # Check that the subject secondary structure and reference secondary
    # structure have the same length.
    if len(subject_secondary_structure) != len(reference_secondary_structure):
        raise ValueError(f"Length of subject secondary structure ({len(subject_secondary_structure)}) must match length of reference secondary structure ({len(reference_secondary_structure)}).")

    # Calculate the base pairs and loops from the secondary structure strings.
    # Also, this function will check the validity of the secondary structures.
    subject_pairs_indices, subject_loop_indices = calculate_base_pairs_and_loops_from_secondary_structure(subject_secondary_structure)
    reference_pairs_indices, reference_loop_indices = calculate_base_pairs_and_loops_from_secondary_structure(reference_secondary_structure)

    # Convert the indices to sets.
    subject_pairs_indices = set(subject_pairs_indices)
    subject_loop_indices = set(subject_loop_indices)

    reference_pairs_indices = set(reference_pairs_indices)
    reference_loop_indices = set(reference_loop_indices)

    # Calculate the number of true positives, false positives, and false 
    # negatives for pairs.
    TP_pairs = len(subject_pairs_indices.intersection(reference_pairs_indices))
    FP_pairs = len(subject_pairs_indices - reference_pairs_indices)
    FN_pairs = len(reference_pairs_indices - subject_pairs_indices)

    # Calculate precision and recall for pairs.
    if TP_pairs + FP_pairs == 0:
        precision_pairs = 0
    else:
        precision_pairs = TP_pairs / (TP_pairs + FP_pairs)
    
    if TP_pairs + FN_pairs == 0:
        recall_pairs = 0
    else:
        recall_pairs = TP_pairs / (TP_pairs + FN_pairs)

    # Calculate F1 score for pairs.
    if precision_pairs + recall_pairs == 0:
        f1_score_pairs = 0
    else:
        f1_score_pairs = 2 * (precision_pairs * recall_pairs) / (precision_pairs + recall_pairs)

    # Calculate the number of true positives, false positives, and false
    # negatives for loops.
    TP_loops = len(subject_loop_indices.intersection(reference_loop_indices))
    FP_loops = len(subject_loop_indices - reference_loop_indices)
    FN_loops = len(reference_loop_indices - subject_loop_indices)

    # Calculate precision and recall for loops.
    if TP_loops + FP_loops == 0:
        precision_loops = 0
    else:
        precision_loops = TP_loops / (TP_loops + FP_loops)
    
    if TP_loops + FN_loops == 0:
        recall_loops = 0
    else:
        recall_loops = TP_loops / (TP_loops + FN_loops)
    
    # Calculate F1 score for loops.
    if precision_loops + recall_loops == 0:
        f1_score_loops = 0
    else:
        f1_score_loops = 2 * (precision_loops * recall_loops) / (precision_loops + recall_loops)
    
    result = {
        "f1_score_pairs": f1_score_pairs,
        "f1_score_loops": f1_score_loops
    }

    return result

def calculate_reactivity_profile_score(reference_secondary_structure,
                                       subject_reactivity_profile):
    """
    Given a reference secondary structure and a subject reactivity profile,
    calculates the EternaFold Classic Score, Crossed Pair Quality Score, and
    OpenKnot score.

    Args:
        reference_secondary_structure (str): The reference secondary structure.
        subject_reactivity_profile (np.ndarray): The reactivity profile.
    
    Returns:
        result (dict): A dictionary containing:
            eternafold_class_score (float): The EternaFold Classic Score.
            crossed_pair_quality_score (float): The Crossed Pair Quality Score.
            openknot_score (float): The OpenKnot score.
    """
    # Setup ARNIE.
    sys.path.append(DEFAULT_ARNIE_PATH)
    with tempfile.NamedTemporaryFile(mode = "wt", suffix = ".txt") as f:
        # Setup the ARNIE config file.
        f.write("linearpartition: . \nTMP: /tmp")
        f.flush()
        arnie_config_path = f.name
        os.environ["ARNIEFILE"] = arnie_config_path
        
        # Import the scoring module from OpenKnotScorePipeline.
        sys.path.append(DEFAULT_OPENKNOT_SCORE_PATH)
        import scoring

    # Check that the subject reactivity profile and reference secondary 
    # structure have the same length.
    if len(subject_reactivity_profile) != len(reference_secondary_structure):
        raise ValueError(f"Length of subject reactivity profile ({len(subject_reactivity_profile)}) must match length of reference secondary structure ({len(reference_secondary_structure)}).")

    # Check the validity of the reference secondary structure.
    check_secondary_structure_validity(reference_secondary_structure)

    # Convert the reactivity profile to a list.
    subject_reactivity_profile = list(subject_reactivity_profile)

    # Calculate the Eterna Classic Score and Crossed Pair Quality Score.
    eternafold_class_score = \
        scoring.calculateEternaClassicScore(reference_secondary_structure, 
                                            subject_reactivity_profile, 
                                            0, 
                                            0)
    crossed_pair_quality_score = \
        scoring.calculateCrossedPairQualityScore(reference_secondary_structure,
                                                 subject_reactivity_profile,
                                                 0,
                                                 0)[1]

    # Calculate the OpenKnot score.
    openknot_score = (0.5 * eternafold_class_score + 0.5 * crossed_pair_quality_score) / 100

    result = {
        "eternafold_class_score": eternafold_class_score,
        "crossed_pair_quality_score": crossed_pair_quality_score,
        "openknot_score": openknot_score
    }

    return result

################################################################################
# Structure Comparison
################################################################################
def run_us_align(reference_structure_path,
                 subject_structure_path,
                 mol = "RNA",
                 mm = 0,
                 ter = 2,
                 atom = "auto",
                 het = 0,
                 us_align_path = DEFAULT_USALIGN_PATH):
    """
    Given a reference structure path and a subject structure path, aligns the
    subject structure to the reference structure using US-Align, and calculates
    the root mean square deviation (RMSD) and TM-score between the aligned
    structures.

    Args:
        reference_structure_path (str): The path to the reference structure to
            align to. Reference structure will remain fixed.
        subject_structure_path (str): The path to the structure to align. This 
            structure will be superimposed onto the reference structure.
        mol (str): Type of molecule(s) to align.
            Options:
                "auto": align both protein and nucleic acids.
                "prot": only align proteins in a structure.
                "RNA": (default) only align RNA and DNA in a structure.
        mm (int): Multimeric alignment option.
            Options:
                0: (default) alignment of two monomeric structures.
                1: alignment of two multi-chain oligomeric structures.
                2: alignment of individual chains to an oligomeric structure.
                3: alignment of circularly permuted structure.
                4: alignment of multiple monomeric chains into a consensus alignment.
                5: fully non-sequential (fNS) alignment.
                6: semi-non-sequential (sNS) alignment.
                To use -mm 1 or -mm 2, '-ter' option must be 0 or 1.
        ter (int): Number of chains to align.
            Options:
                3: only align the first chain, or the first segment of the
                    first chain as marked by the 'TER' string in PDB file.
                2: (default) only align the first chain.
                1: align all chains of the first model (recommended for aligning
                    asymmetric units).
                0: align all chains from all models (recommended for aligning
                    biological assemblies, i.e. biounits).
        atom (str): 4-character atom name used to represent a residue. This is
            the atom that will be used to align the structures.
            Options:
                "auto": (default) " C3'" for RNA/DNA and " CA " for proteins.
                four-character atom name: e.g. " C3'" for RNA/DNA and " CA " 
                    for proteins. Note, if mol is set to "auto", atom must also
                    be set to "auto". This is because it is not possible to
                    specify atoms for both protein and nucleic acids. This will
                    result in only the corresponding molecule being aligned.
        het (int): Whether to align residues marked as 'HETATM' in addition to
            'ATOM  '.
            Options:
                0: (default) only align 'ATOM  ' residues.
                1: align both 'ATOM  ' and 'HETATM' residues.
                2: align both 'ATOM  ' and MSE residues.
        us_align_path (str): The path to the US-Align executable.

    Returns:
        result (dict): A dictionary containing:
            rmsd (float): The root mean square deviation (RMSD) between the 
                aligned structures.
            tm_score (float): The TM-score between the aligned structures,
                normalized by the length of the reference structure.
    """
    # Check that the mol and atom options agree.
    if mol == "auto" and atom != "auto":
        raise ValueError("If mol is set to 'auto', atom must also be set to 'auto'.")

    # Convert the structure paths to absolute paths.
    subject_structure_path = os.path.abspath(subject_structure_path)
    reference_structure_path = os.path.abspath(reference_structure_path)

    # Check that the structure paths exist.
    if not os.path.exists(subject_structure_path):
        raise ValueError(f"Structure file not found: {subject_structure_path}")
    if not os.path.exists(reference_structure_path):
        raise ValueError(f"Structure file not found: {reference_structure_path}")

    # Create a temporary file for the US-Align output.
    us_align_output_file = tempfile.NamedTemporaryFile(mode = "wt")

    # Run US-Align.
    try:
        subprocess.run(
            [
                str(us_align_path),
                "-mol",
                str(mol),
                "-mm",
                str(mm),
                "-ter",
                str(ter), 
                "-atom",
                str(atom),
                "-het",
                str(het),
                str(subject_structure_path),
                str(reference_structure_path)
            ],
            check = True,
            stdout = us_align_output_file,
            stderr = subprocess.DEVNULL
        )

        us_align_output_text = read_text_file(us_align_output_file.name)

        # Extract the TM-score and RMSD from the US-Align output.
        rmsd = None
        tm_score = None
        for line in us_align_output_text.split("\n"):
            if line.startswith("Aligned length="):
                rmsd = float(line.split("RMSD=")[1].split(",")[0].strip())
            elif line.startswith("TM-score=") and "normalized by length of Structure_2" in line:
                tm_score = float(line.split("TM-score=")[1].split("(normalized by length of Structure_2")[0].strip())
        
        us_align_output_file.close()
        
        if rmsd is None or tm_score is None:
            raise ValueError("Failed to extract RMSD and TM-score from US-Align output.")
        
        result = {
            "rmsd": rmsd,
            "tm_score": tm_score
        }

        return result
    except (subprocess.CalledProcessError, ValueError) as e:
        us_align_output_file.close()
        raise e


################################################################################
# Sequence design
################################################################################
def run_na_mpnn_sequence(structure_path,
                         output_directory = None,
                         batch_size = 1,
                         number_of_batches = 1,
                         temperature = 0.1,
                         omit_AA = "",
                         design_na_only = 0,
                         load_residues_with_missing_atoms = 0,
                         output_pdbs = 0,
                         catch_failed_inferences = 1,
                         na_mpnn_apptainer_path = DEFAULT_NA_MPNN_APPTAINER_PATH,
                         na_mpnn_path = DEFAULT_NA_MPNN_PATH,
                         na_mpnn_model_path = None,
                         model_mode = None,
                         dfm_dt = None,
                         dfm_schedule = None,
                         eds_schedule_path = None,
                         trajectory_dir = None):
    """
    Given a structure path, runs the NA-MPNN sequence design algorithm to
    generate sequences for the structure. The output is a list of dictionaries
    containing the design ID, name, design sequence, and tool-reported sequence
    recovery.

    Args:
        structure_path (str): The path to the structure file.
        output_directory (str): The path to the output directory. If not
            specified, a temporary directory will be created.
        batch_size (int): The batch size for the NA-MPNN algorithm.
        number_of_batches (int): The number of batches to run.
        temperature (float): The temperature for the NA-MPNN algorithm.
        omit_AA (str): The amino acids to omit from the design.
        design_na_only (int): Whether to design only nucleic acids.
        load_residues_with_missing_atoms (int): Whether to load residues with
            missing atoms.
        output_pdbs (int): Whether to output PDB files.
        catch_failed_inferences (int): Whether to catch failed inferences.
        na_mpnn_apptainer_path (str): The path to the NA-MPNN apptainer.
        na_mpnn_path (str): The path to the NA-MPNN run file.
        na_mpnn_model_path (str): The path to the NA-MPNN model file.
    
    Returns:
        design_data (dict list): A list of dictionaries containing:
            input_structure_name (str): The name of the input structure.
            input_structure_path (str): The path to the input structure.
            design_id (str): The design ID.
            name (str): The name of the design.
            design_sequence (str): The design sequence.
            tool_reported_sequence_recovery (float): The tool-reported sequence
                recovery.
            design_method (str): The design method used.
            model_weights_path (str): The path to the model weights used.
    """
    # Convert the structure path to an absolute path.
    structure_path = os.path.abspath(structure_path)

    # Check that the structure path exists.
    if not os.path.exists(structure_path):
        raise ValueError(f"Structure file not found: {structure_path}")

    # If the output directory is not specified, create a temporary directory.
    # The temporary directory will be automatically cleaned up when the script
    # exits.
    if output_directory is None:
        tmp_directory = tempfile.TemporaryDirectory()
        output_directory = tmp_directory.name
    else:
        output_directory = os.path.abspath(output_directory)
    
    # Compute the output directory for the sequences.
    seqs_output_directory = os.path.join(output_directory, "seqs")
    
    # Compute the name of the structure.
    structure_name = os.path.splitext(os.path.basename(structure_path))[0]

    # Run the NA-MPNN sequence design algorithm.
    try:
        if str(na_mpnn_apptainer_path):
            command = [
                "apptainer",
                "exec",
                na_mpnn_apptainer_path,
                "python",
                na_mpnn_path
            ]
        else:
            command = [
                sys.executable,
                na_mpnn_path
            ]

        command.extend([
            "--model_type",
            str("na_mpnn"),
            "--checkpoint_na_mpnn",
            str(na_mpnn_model_path),
            "--pdb_path",
            str(structure_path),
            "--out_folder",
            str(output_directory),
            "--number_of_batches",
            str(number_of_batches),
            "--batch_size",
            str(batch_size),
            "--temperature",
            str(temperature),
            "--omit_AA",
            str(omit_AA),
            "--design_na_only",
            str(design_na_only),
            "--load_residues_with_missing_atoms",
            str(load_residues_with_missing_atoms),
            "--output_pdbs",
            str(output_pdbs),
            "--catch_failed_inferences",
            str(catch_failed_inferences)
        ])

        if model_mode is not None:
            command.extend(["--model_mode", str(model_mode)])

        if dfm_dt is not None:
            command.extend(["--dfm_dt", str(dfm_dt)])

        if dfm_schedule is not None:
            command.extend(["--dfm_schedule", str(dfm_schedule)])

        if eds_schedule_path is not None:
            command.extend(["--eds_schedule_path", str(eds_schedule_path)])

        if trajectory_dir is not None:
            command.extend(["--trajectory_dir", str(trajectory_dir)])

        subprocess.run(
            command,
            check = True,
            stdout = subprocess.DEVNULL,
            stderr = subprocess.DEVNULL
        )

        # Check that the output fasta file exists.
        fasta_path = os.path.join(seqs_output_directory, f"{structure_name}.fa")
        if not os.path.exists(fasta_path):
            raise ValueError(f"Output fasta file not found: {fasta_path}")

        # Read the output fasta file.
        fasta_entries = read_fasta_file(fasta_path)

        # Skip the first entry of the fasta, which contains the parent sequence.
        fasta_entries = fasta_entries[1:]

        design_data = []
        for fasta_header, fasta_sequence in fasta_entries:
            fasta_header = fasta_header.strip()
            fasta_header_metadata = fasta_header.split(", ")

            metadata_dict = dict()
            for metadata in fasta_header_metadata[1:]:
                metadata = metadata.strip()
                metadata_name, metadata_value = metadata.split("=", maxsplit=1)
                metadata_dict[metadata_name] = metadata_value

            design_dict = {
                "input_structure_name": structure_name,
                "input_structure_path": structure_path,
                "design_id": metadata_dict["id"],
                "name": f"{structure_name}_{metadata_dict['id']}",
                "design_sequence": fasta_sequence,
                "tool_reported_sequence_recovery": float(metadata_dict["seq_rec"]),
                "design_method": "na_mpnn",
                "model_weights_path": na_mpnn_model_path
            }

            design_data.append(design_dict)

        # Clean up the temporary directory if it was created.
        if output_directory is None:
            tmp_directory.cleanup()

        return design_data
    except (subprocess.CalledProcessError, ValueError) as e:
        # Clean up the temporary directory if it was created.
        if output_directory is None:
            tmp_directory.cleanup()
        raise e
    



################################################################################
# Combined Functionality
################################################################################
def design_nucleic_acid_sequence(structure_path,
                                 overall_output_directory,
                                 num_samples,
                                 temperature,
                                 method = "na_mpnn",
                                 na_mpnn_model_path = None,
                                 model_mode = None,
                                 dfm_dt = None,
                                 dfm_schedule = None,
                                 eds_schedule_path = None,
                                 trajectory_dir = None,
                                 design_mode = "na"):
    """
    Given a structure path, an overall output directory, the number of samples,
    the temperature, and the sequence design method, runs the specified
    sequence design method to generate sequences for the structure. A JSON is
    created for each design, containing the design ID, name, design sequence,
    and tool-reported sequence recovery.

    Args:
        structure_path (str): The path to the structure file.
        overall_output_directory (str): The path to the overall output directory.
        num_samples (int): The number of samples to generate.
        temperature (float): The temperature for the sequence design algorithm.
        method (str): The sequence design method to use. Options are "na_mpnn",
            "grnade", and "rhodesign". Default is "na_mpnn".
        na_mpnn_model_path (str): The path to the NA-MPNN model file. Required
            if method is "na_mpnn".
    
    Side Effects:
        Creates an output directory for the structure, copies the structure to
            the output directory, creates a subdirectory for the design JSON
            files, and saves a JSON file for each design containing the design
            ID, name, design sequence, and tool-reported sequence recovery.
    """
    # Convert the structure path and overall output directory to absolute paths.
    structure_path = os.path.abspath(structure_path)
    overall_output_directory = os.path.abspath(overall_output_directory)

    if temperature is None:
        temperature = 0.1
    
    if na_mpnn_model_path is None:
        na_mpnn_model_path = DEFAULT_NA_MPNN_MODEL_PATH

    # Check that the structure path exists.
    if not os.path.exists(structure_path):
        raise ValueError(f"Structure file not found: {structure_path}")
    
    # Create the overall output directory if it does not exist.
    os.makedirs(overall_output_directory, exist_ok = True)

    # Get the basename without the ".gz" extension.
    if structure_path.endswith(".gz"):
        structure_basename = os.path.splitext(os.path.basename(structure_path))[0]
    else:
        structure_basename = os.path.basename(structure_path)
    # Extract the name of the structure (without the extension).
    if structure_basename.endswith(".pdb") or structure_basename.endswith(".cif"):
        structure_name = os.path.splitext(structure_basename)[0]
    else:
        raise ValueError(f"Invalid structure file extension: {structure_basename}")

    # Create the specific output directory for the structure. If the directory
    # already exists, remove it and create a new one.
    output_directory = os.path.join(overall_output_directory, structure_name)
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)
    os.makedirs(output_directory)

    # Copy the structure to the output directory. If it is a gzipped file,
    # decompress it first.
    copy_structure_path = os.path.join(output_directory, structure_basename)
    if structure_path.endswith(".gz"):
        with gzip.open(structure_path, "rb") as f_in:
            with open(copy_structure_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
    else:
        shutil.copy(structure_path, copy_structure_path)

    # Save the original and new structure paths.
    original_structure_path = structure_path
    structure_path = copy_structure_path

    # Design JSON output directory.
    design_json_output_directory = os.path.join(output_directory, "design_json")
    os.makedirs(design_json_output_directory)

    if method == "na_mpnn":
        # Run NA-MPNN sequence design.
        _omit_AA      = "X" if design_mode == "all" else "ARNDCQEGHILKMFPSTWYVXbdhuy"
        _design_na_only = 0 if design_mode == "all" else 1
        design_data = run_na_mpnn_sequence(
            structure_path,
            output_directory = output_directory,
            batch_size = num_samples,
            number_of_batches = 1,
            temperature = temperature,
            omit_AA = _omit_AA,
            design_na_only = _design_na_only,
            load_residues_with_missing_atoms = 0,
            output_pdbs = 0,
            catch_failed_inferences = 1,
            na_mpnn_model_path = na_mpnn_model_path,
            model_mode = model_mode,
            dfm_dt = dfm_dt,
            dfm_schedule = dfm_schedule,
            eds_schedule_path = eds_schedule_path,
            trajectory_dir = trajectory_dir,
        )
    elif method == "grnade":
        # Run gRNAde sequence design.
        design_data = run_grnade(
            structure_path,
            output_directory = output_directory,
            n_samples = num_samples,
            temperature = temperature
        )
    elif method == "rhodesign":
        # Run RhoDesign sequence design.
        design_data = run_rhodesign(
            structure_path,
            output_directory = output_directory,
            n_samples = num_samples,
            temperature = temperature
        )
    else:
        raise ValueError(f"Invalid sequence design method: {method}")

    # Write the design data to a JSON file.
    for design_dict in design_data:
        design_dict["original_input_structure_path"] = original_structure_path
        design_json_path = os.path.join(
            design_json_output_directory,
            f"{design_dict['name']}.json"
        )
        write_json_file(design_json_path, design_dict)            

def process_reference_monomer_rna(reference_structure_path, 
                                  overall_output_directory):
    """
    Given a reference structure path and an overall output directory,
    processes the reference structure, extracts its sequence and secondary
    structure with DSSR, and saves the results to a JSON file.

    Args:
        reference_structure_path (str): The path to the reference structure.
        overall_output_directory (str): The path to the overall output 
            directory.
    
    Side Effects:
        Creates an output directory for the reference structure, copies the 
            reference structure to the output directory, and saves a JSON file 
            with the results of the predictions.
    """
    # Convert the structure path and overall output directory to absolute paths.
    reference_structure_path = os.path.abspath(reference_structure_path)
    overall_output_directory = os.path.abspath(overall_output_directory)

    # Check that the reference structure path.
    if not os.path.exists(reference_structure_path):
        raise ValueError(f"Reference structure file not found: {reference_structure_path}")
    
    # Create the output directory if it does not exist.
    os.makedirs(overall_output_directory, exist_ok = True)
    
    # Get the basename without the ".gz" extension.
    if reference_structure_path.endswith(".gz"):
        reference_structure_basename = os.path.splitext(os.path.basename(reference_structure_path))[0]
    else:
        reference_structure_basename = os.path.basename(reference_structure_path)
    # Extract the name of the structure (without the extension).
    if reference_structure_basename.endswith(".pdb") or reference_structure_basename.endswith(".cif"):
        structure_name = os.path.splitext(reference_structure_basename)[0]
    else:
        raise ValueError(f"Invalid structure file extension: {reference_structure_basename}")

    # Create the specific output directory for the structure. If the directory
    # already exists, remove it and create a new one.
    output_directory = os.path.join(overall_output_directory, structure_name)
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)
    os.makedirs(output_directory)

    # Copy the reference structure to the output directory. If it is a gzipped
    # file, decompress it first.
    copy_reference_structure_path = os.path.join(output_directory, reference_structure_basename)
    if reference_structure_path.endswith(".gz"):
        with gzip.open(reference_structure_path, "rb") as f_in:
            with open(copy_reference_structure_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
    else:
        shutil.copy(reference_structure_path, copy_reference_structure_path)
    
    # Save the original and new structure paths.
    original_reference_structure_path = reference_structure_path
    reference_structure_path = copy_reference_structure_path

    # Create the output directory for the reference json results.
    reference_json_output_directory = os.path.join(output_directory, "reference_json")
    os.makedirs(reference_json_output_directory)

    # Run dssr.
    dssr_output = run_dssr(reference_structure_path)
    
    # Standardize the dssr sequence.
    dssr_output["sequence"] = \
        standardize_rna_sequence(dssr_output["sequence"], 
                                 method = "dssr")
    
    # Check that sequence is valid.
    check_rna_sequence_validity(dssr_output["sequence"],
                                unknown_residue_allowed = True,
                                chain_breaks_allowed = False)
    
    # Standardize the dssr secondary structure.
    dssr_output["secondary_structure"] = \
        standardize_secondary_structure(dssr_output["secondary_structure"], 
                                        method = "dssr")

    output_dict = {
        "name": structure_name,
        "original_reference_structure_path": original_reference_structure_path,
        "reference_structure_path": reference_structure_path,
        "dssr": dssr_output,
    }

    # Save the output dictionary to a JSON file.
    output_json_path = os.path.join(reference_json_output_directory, 
                                    f"{structure_name}.json")
    write_json_file(output_json_path, output_dict)

def process_design_monomer_rna(subject_path,
                               overall_output_directory,
                               skip_alphafold3=False):
    """
    Given a design path and an overall output directory, processes the design
    by extracting its sequence and secondary structure with DSSR, predicting
    its secondary structure with EternaFold, predicting its secondary
    structure and reactivity profile with RiboNanzaNet, and predicting its
    structure with AlphaFold3. The results are saved to a JSON file.
    
    Args:
        subject_path (str): The path to the design JSON file.
        overall_output_directory (str): The path to the overall output 
            directory.
    
    Side Effects:
        Creates an output directory for the design, copies the design fasta 
            file to the output directory, and saves a JSON file with the 
            results of the predictions.
    """
    # Convert the subject path and overall output directory to absolute paths.
    subject_path = os.path.abspath(subject_path)
    overall_output_directory = os.path.abspath(overall_output_directory)

    # Check that the subject path exists.
    if not os.path.exists(subject_path):
        raise ValueError(f"Design fasta file not found: {subject_path}")
    
    # Create the output directory if it does not exist.
    os.makedirs(overall_output_directory, exist_ok = True)
    
    # Read the subject JSON file.
    design_json = read_json_file(subject_path)

    # Get the name of the design.
    design_name = design_json["name"]

    # Create the specific output directory for the design.
    output_directory = os.path.join(overall_output_directory, design_name)

    # Resume: skip if the final output JSON already exists.
    processed_design_json_output_directory = os.path.join(output_directory, "processed_design_json")
    output_json_path = os.path.join(processed_design_json_output_directory, f"{design_name}.json")
    if os.path.exists(output_json_path) and not skip_alphafold3:
        return

    # If skipping AlphaFold3, load existing AF3 result from the current JSON.
    existing_alphafold3_result = None
    if skip_alphafold3 and os.path.exists(output_json_path):
        existing_json = read_json_file(output_json_path)
        existing_alphafold3_result = existing_json.get("alphafold3", None)

    os.makedirs(output_directory, exist_ok=True)
    os.makedirs(processed_design_json_output_directory, exist_ok=True)

    # Get the design sequence.
    design_sequence = design_json["design_sequence"]
    design_method = design_json["design_method"]

    # For protein-RNA complexes, strip protein chains (chains containing
    # characters not in the NA-MPNN nucleotide alphabet).
    if design_method == "na_mpnn":
        _na_chars = set('acgtubdhyACGUX')
        _chains = design_sequence.split('/')
        _na_chains = [c for c in _chains if c and all(ch in _na_chars for ch in c)]
        design_sequence = '/'.join(_na_chains)

    # Skip sequences containing DNA (thymine). After protein-chain stripping,
    # any remaining 't'/'T' is a DNA base that the RNA-only pipeline cannot
    # handle (EternaFold, RibonanzaNet, AF3 RNA mode all expect pure RNA).
    _na_seq = design_sequence.replace('/', '').lower()
    if 't' in _na_seq:
        print(f"Skipping {design_name}: sequence contains DNA (thymine)")
        return

    # Standardize the design sequence.
    design_sequence = standardize_rna_sequence(design_sequence,
                                               method = design_method)
    
    # Check that sequence is valid.
    check_rna_sequence_validity(design_sequence,
                                unknown_residue_allowed = False,
                                chain_breaks_allowed = False)

    # Predict the secondary structure of the design sequence with EternaFold.
    # Handle gracefully if EternaFold fails (e.g., due to CPU compatibility issues)
    try:
        eternafold_result = run_eternafold(design_sequence)
    except subprocess.CalledProcessError as e:
        if e.returncode == -4:  # SIGILL signal
            print(f"WARNING: EternaFold failed with SIGILL (CPU compatibility issue). Skipping EternaFold prediction.", file=sys.stderr)
            eternafold_result = {
                "predicted_secondary_structure": None,
                "error": "EternaFold SIGILL: CPU compatibility issue"
            }
        else:
            # Re-raise other errors
            raise e

    # Predict the secondary structure and reactivity profile of the design
    # sequence with RiboNanzaNet.
    ribonanza_net_secondary_structure_result = \
        run_ribonanza_net_secondary_structure(design_sequence)
    ribonanza_net_reactivity_profile_result = \
        run_ribonanza_net_reactivity_profile(design_sequence)
    
    # Predict the structure of the design sequence with AlphaFold3.
    if skip_alphafold3 and existing_alphafold3_result is not None:
        alphafold3_result = existing_alphafold3_result
    else:
        _af3_t0 = time.time()
        alphafold3_result = run_alphafold3(
            name = design_name,
            sequences_and_polytypes = [(design_sequence, "rna")],
            output_dir = output_directory,
            num_diffusion_samples = 5,
            num_seeds = 1,
            alphafold3_apptainer_path = "/rds/user/mh2167/hpc-work/NA-MPNN/evaluation/models/alphafold3/docker/alphafold3_amd64.sif",
            run_data_pipeline = False,
            buckets = "1"
        )
        print(f"AlphaFold3 inference time: {time.time() - _af3_t0:.1f}s", flush=True)

    # Create the output dictionary.
    output_dict = {
        "name": design_name,
        "sequence": design_sequence,
        "design_input_path": subject_path,
        "eternafold": eternafold_result,
        "ribonanza_net_secondary_structure": ribonanza_net_secondary_structure_result,
        "ribonanza_net_reactivity_profile": ribonanza_net_reactivity_profile_result,
        "alphafold3": alphafold3_result
    }

    # Save the output dictionary to a JSON file.
    output_json_path = os.path.join(processed_design_json_output_directory, 
                                    f"{design_name}.json")
    write_json_file(output_json_path, output_dict)

def process_designs_monomer_rna_batch(subject_paths,
                                      overall_output_directory,
                                      skip_alphafold3=False):
    """
    Batch version of process_design_monomer_rna. Runs EternaFold and
    RibonanzaNet per-sample, then runs AlphaFold3 once for all samples
    using --input_dir, then assembles the final output JSONs.

    Args:
        subject_paths (str list): Paths to design JSON files.
        overall_output_directory (str): The path to the overall output
            directory.
        skip_alphafold3 (bool): Skip AlphaFold3 and reuse existing AF3
            results from the output JSON if available.
    """
    overall_output_directory = os.path.abspath(overall_output_directory)
    os.makedirs(overall_output_directory, exist_ok = True)

    # Per-sample pre-processing: run EternaFold and RibonanzaNet, and
    # prepare AF3 input JSONs in a staging directory.
    af3_staging_dir = tempfile.mkdtemp(prefix="af3_batch_")
    # Collect (design_name, output_directory, pre_results_dict) for each
    # sample that needs AF3, so we can assemble final JSONs after the
    # batch AF3 run.
    samples_for_af3 = []
    # Also track samples that were skipped or already done, so we can
    # still write their output JSONs.
    all_samples = []

    for subject_path in subject_paths:
        subject_path = os.path.abspath(subject_path)

        # Check that the subject path exists.
        if not os.path.exists(subject_path):
            print(f"WARNING: Design file not found, skipping: {subject_path}",
                  file=sys.stderr)
            continue

        # Read the subject JSON file.
        design_json = read_json_file(subject_path)
        design_name = design_json["name"]

        # Create the specific output directory for the design.
        output_directory = os.path.join(overall_output_directory, design_name)

        # Resume: skip if the final output JSON already exists.
        processed_design_json_output_directory = os.path.join(
            output_directory, "processed_design_json")
        output_json_path = os.path.join(
            processed_design_json_output_directory, f"{design_name}.json")
        if os.path.exists(output_json_path) and not skip_alphafold3:
            print(f"Skipping {design_name} (already processed)")
            continue

        # If skipping AlphaFold3, load existing AF3 result from the current JSON.
        existing_alphafold3_result = None
        if skip_alphafold3 and os.path.exists(output_json_path):
            existing_json = read_json_file(output_json_path)
            existing_alphafold3_result = existing_json.get("alphafold3", None)

        os.makedirs(output_directory, exist_ok=True)
        os.makedirs(processed_design_json_output_directory, exist_ok=True)

        # Get the design sequence.
        design_sequence = design_json["design_sequence"]
        design_method = design_json["design_method"]

        # For protein-RNA complexes, strip protein chains.
        if design_method == "na_mpnn":
            _na_chars = set('acgtubdhyACGUX')
            _chains = design_sequence.split('/')
            _na_chains = [c for c in _chains if c and all(ch in _na_chars for ch in c)]
            design_sequence = '/'.join(_na_chains)

        # Skip sequences containing DNA (thymine).
        _na_seq = design_sequence.replace('/', '').lower()
        if 't' in _na_seq:
            print(f"Skipping {design_name}: sequence contains DNA (thymine)")
            continue

        # Standardize the design sequence.
        design_sequence = standardize_rna_sequence(design_sequence,
                                                   method = design_method)

        # Check that sequence is valid.
        check_rna_sequence_validity(design_sequence,
                                    unknown_residue_allowed = False,
                                    chain_breaks_allowed = False)

        # Predict the secondary structure with EternaFold.
        try:
            eternafold_result = run_eternafold(design_sequence)
        except subprocess.CalledProcessError as e:
            if e.returncode == -4:  # SIGILL signal
                print(f"WARNING: EternaFold failed with SIGILL (CPU compatibility issue). Skipping EternaFold prediction.", file=sys.stderr)
                eternafold_result = {
                    "predicted_secondary_structure": None,
                    "error": "EternaFold SIGILL: CPU compatibility issue"
                }
            else:
                raise e

        # Predict secondary structure and reactivity with RiboNanzaNet.
        ribonanza_net_secondary_structure_result = \
            run_ribonanza_net_secondary_structure(design_sequence)
        ribonanza_net_reactivity_profile_result = \
            run_ribonanza_net_reactivity_profile(design_sequence)

        pre_results = {
            "name": design_name,
            "sequence": design_sequence,
            "design_input_path": subject_path,
            "eternafold": eternafold_result,
            "ribonanza_net_secondary_structure": ribonanza_net_secondary_structure_result,
            "ribonanza_net_reactivity_profile": ribonanza_net_reactivity_profile_result,
        }

        # Handle AF3: either reuse existing, prepare for batch, or skip.
        if skip_alphafold3 and existing_alphafold3_result is not None:
            pre_results["alphafold3"] = existing_alphafold3_result
            all_samples.append((design_name, output_directory, pre_results, True))
        else:
            # Check if AF3 already ran successfully (resume support).
            af3_output_directory = os.path.join(output_directory, design_name)
            _af3_done = (
                os.path.exists(os.path.join(af3_output_directory, f"{design_name}_data.json")) and
                os.path.exists(os.path.join(af3_output_directory, f"{design_name}_model.cif")) and
                os.path.exists(os.path.join(af3_output_directory, f"{design_name}_confidences.json")) and
                os.path.exists(os.path.join(af3_output_directory, f"{design_name}_summary_confidences.json"))
            )

            if not _af3_done:
                # Prepare AF3 input JSON in the staging directory.
                prepare_alphafold3_input(
                    name = design_name,
                    sequences_and_polytypes = [(design_sequence, "rna")],
                    staging_dir = af3_staging_dir,
                    num_seeds = 1,
                )
            all_samples.append((design_name, output_directory, pre_results, False))
            samples_for_af3.append(design_name)

    # Run AF3 batch if there are any samples to process.
    # AF3 writes outputs to output_dir/name/, so we use a temp output dir and
    # then move each result into the per-sample output_directory to match the
    # original directory structure (overall_output_directory/name/name/).
    af3_output_dir = tempfile.mkdtemp(prefix="af3_batch_out_")
    af3_input_count = len([f for f in os.listdir(af3_staging_dir) if f.endswith(".json")])
    if af3_input_count > 0 and not skip_alphafold3:
        print(f"Running AlphaFold3 batch on {af3_input_count} samples...", flush=True)
        _af3_t0 = time.time()
        run_alphafold3_batch(
            staging_dir = af3_staging_dir,
            output_dir = af3_output_dir,
            num_diffusion_samples = 5,
            run_data_pipeline = False,
            alphafold3_apptainer_path = "/rds/user/mh2167/hpc-work/NA-MPNN/evaluation/models/alphafold3/docker/alphafold3_amd64.sif",
        )
        print(f"AlphaFold3 batch inference time: {time.time() - _af3_t0:.1f}s "
              f"({af3_input_count} samples)", flush=True)

        # Move AF3 outputs into the per-sample directories to match the
        # original structure: overall_output_directory/name/name/
        for design_name in samples_for_af3:
            src = os.path.join(af3_output_dir, design_name)
            dst = os.path.join(overall_output_directory, design_name, design_name)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.move(src, dst)
    elif af3_input_count == 0 and len(samples_for_af3) > 0:
        print("All AF3 outputs already exist (resume), skipping batch run.", flush=True)

    # Clean up staging and temp output directories.
    shutil.rmtree(af3_staging_dir, ignore_errors=True)
    shutil.rmtree(af3_output_dir, ignore_errors=True)

    # Assemble final output JSONs.
    for design_name, output_directory, pre_results, af3_already_done in all_samples:
        if not af3_already_done:
            # Parse AF3 output (lives at output_directory/design_name/).
            alphafold3_result = parse_alphafold3_output(
                name = design_name,
                output_dir = output_directory,
            )
            pre_results["alphafold3"] = alphafold3_result

        # Save the output dictionary to a JSON file.
        processed_design_json_output_directory = os.path.join(
            output_directory, "processed_design_json")
        os.makedirs(processed_design_json_output_directory, exist_ok=True)
        output_json_path = os.path.join(
            processed_design_json_output_directory, f"{design_name}.json")
        write_json_file(output_json_path, pre_results)
        print(f"Successfully processed {design_name}")


def score_design_monomer_rna(reference_path, subject_path, overall_output_directory):
    """
    Given a reference path and a subject path, scores the design by comparing
    the reference and subject sequences, secondary structures, reactivity
    profiles, and structures.

    Args:
        reference_path (str): The path to the reference output json.
        subject_path (str): The path to the subject output json.
        overall_output_directory (str): The path to the overall output 
            directory.
    
    Side Effects:
        Creates an output directory for the subject and saves a JSON file
            with the results of the scoring.
    """
    import biotite
    import biotite.structure.io

    # Convert the reference path and subject path to absolute paths.
    reference_path = os.path.abspath(reference_path)
    subject_path = os.path.abspath(subject_path)

    # Check that the reference path exists.
    if not os.path.exists(reference_path):
        raise ValueError(f"Reference structure file not found: {reference_path}")
    
    # Check that the subject path exists.
    if not os.path.exists(subject_path):
        raise ValueError(f"Subject structure file not found: {subject_path}")
    
    # Create the output directory if it does not exist.
    os.makedirs(overall_output_directory, exist_ok = True)

    # Load the reference output.
    reference_output = read_json_file(reference_path)

    # Load the subject output.
    subject_output = read_json_file(subject_path)

    # Make the output directory for the subject if it does not exist. If the
    # directory already exists, remove it and create a new one.
    output_directory = os.path.join(overall_output_directory,
                                    subject_output["name"])
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)
    os.makedirs(output_directory)

    # Load the C1' atoms from the reference and subject structures.
    subject_atom_array = biotite.structure.io.load_structure(subject_output["alphafold3"]["predicted_structure_path"])
    reference_atom_array = biotite.structure.io.load_structure(reference_output["reference_structure_path"])
    
    # Handle AtomArrayStack (multiple models) by selecting the first model
    if hasattr(subject_atom_array, 'shape') and len(subject_atom_array.shape) > 1:
        subject_atom_array = subject_atom_array[0]  # Select first model
    if hasattr(reference_atom_array, 'shape') and len(reference_atom_array.shape) > 1:
        reference_atom_array = reference_atom_array[0]  # Select first model
    
    reference_atom_array = reference_atom_array[reference_atom_array.atom_name == "C1'"]
    subject_atom_array = subject_atom_array[subject_atom_array.atom_name == "C1'"]

    # Handle the case where the subject sequence is shorter than the reference
    # sequence. This can happen if residues at the end get chopped off.
    subject_sequence_length = len(subject_output["sequence"])
    reference_sequence_length = len(reference_output["dssr"]["sequence"])
    if subject_sequence_length == reference_sequence_length:
        best_start_idx = None
        best_end_idx = None
    elif subject_sequence_length < reference_sequence_length:
        # Perform an rmsd calculation to determine the best overlap.
        best_rmsd = None
        best_start_idx = None
        for possible_start_idx in range(reference_sequence_length - subject_sequence_length + 1):
            reference_start_idx = possible_start_idx
            reference_end_idx = reference_start_idx + subject_sequence_length

            # Subset the reference atom array.
            reference_atom_subarray = reference_atom_array[
                reference_start_idx:reference_end_idx
            ]

            # Superimpose the reference and subject atom arrays.
            superimposed, _ = biotite.structure.superimpose(
                reference_atom_subarray,
                subject_atom_array
            )

            # Calculate the RMSD.
            c1_prime_rmsd = biotite.structure.rmsd(
                reference_atom_subarray,
                superimposed
            )

            if best_rmsd is None or c1_prime_rmsd < best_rmsd:
                best_rmsd = c1_prime_rmsd
                best_start_idx = possible_start_idx
        
        best_end_idx = best_start_idx + subject_sequence_length

        # Subset the sequence, and atom array to the best overlap.
        reference_output["dssr"]["sequence"] = \
            reference_output["dssr"]["sequence"][best_start_idx:best_end_idx]
        reference_atom_array = reference_atom_array[best_start_idx:best_end_idx]

        # The secondary structure needs to be modified in an appropriate way
        # (any base pairs to the removed residues need to be removed).
        base_pair_indices, _ = calculate_base_pairs_and_loops_from_secondary_structure(
            reference_output["dssr"]["secondary_structure"]
        )
        updated_secondary_structure = reference_output["dssr"]["secondary_structure"]
        for (i, j) in base_pair_indices:
            if i < best_start_idx or j < best_start_idx or \
               i >= best_end_idx or j >= best_end_idx:
                
                # Turn i and j indices into loops.
                updated_secondary_structure = \
                    updated_secondary_structure[:i] + \
                    NAConstants.loop_symbols[0] + \
                    updated_secondary_structure[i + 1:]
                updated_secondary_structure = \
                    updated_secondary_structure[:j] + \
                    NAConstants.loop_symbols[0] + \
                    updated_secondary_structure[j + 1:]
        
        # Now trim the updated secondary structure.
        reference_output["dssr"]["secondary_structure"] = \
            updated_secondary_structure[best_start_idx:best_end_idx]
    else:
        raise ValueError("Subject sequence is longer than reference sequence.")
    
    # Compare the sequences.
    sequence_recovery_result = calculate_sequence_recovery(
        reference_output["dssr"]["sequence"],
        subject_output["sequence"]
    )

    # Compare the reference secondary structure to the eternafold predicted
    # secondary structure.
    eternafold_secondary_structure_result = \
        calculate_secondary_structure_stats(
            reference_output["dssr"]["secondary_structure"],
            subject_output["eternafold"]["predicted_secondary_structure"]
        )
    
    # Compare the reference secondary structure to the ribonanza net
    # predicted secondary structures.
    ribonanza_net_secondary_structure_result = dict()
    for predicted_secondary_structure in subject_output["ribonanza_net_secondary_structure"]["predicted_secondary_structures"]:
        individual_result = \
            calculate_secondary_structure_stats(
                reference_output["dssr"]["secondary_structure"],
                predicted_secondary_structure
            )
        
        # Append the results for each ribonanza net predicted secondary
        # structure to the ribonanza net secondary structure result.
        for metric_name, metric_value in individual_result.items():
            if metric_name not in ribonanza_net_secondary_structure_result:
                ribonanza_net_secondary_structure_result[metric_name] = []
            ribonanza_net_secondary_structure_result[metric_name].append(metric_value)
    
    # Calculate the mean of the ribonanza net secondary structure results.
    for metric_name, metric_values in list(ribonanza_net_secondary_structure_result.items()):
        ribonanza_net_secondary_structure_result[f"mean_{metric_name}"] = \
            np.mean(metric_values)
        
    # Compare the reference secondary structure to the ribonanza net
    # predicted reactivity profiles.
    ribonanza_net_reactivity_profile_result = dict()
    for predicted_reactivity_profile in subject_output["ribonanza_net_reactivity_profile"]["predicted_2A3_reactivity_profiles"]:
        individual_result = \
            calculate_reactivity_profile_score(
                reference_output["dssr"]["secondary_structure"],
                predicted_reactivity_profile
            )
        
        # Append the results for each ribonanza net predicted reactivity
        # profile to the ribonanza net reactivity profile result.
        for metric_name, metric_value in individual_result.items():
            if metric_name not in ribonanza_net_reactivity_profile_result:
                ribonanza_net_reactivity_profile_result[metric_name] = []
            ribonanza_net_reactivity_profile_result[metric_name].append(metric_value)
        
    # Calculate the mean of the ribonanza net reactivity profile results.
    for metric_name, metric_values in list(ribonanza_net_reactivity_profile_result.items()):
        ribonanza_net_reactivity_profile_result[f"mean_{metric_name}"] = \
            np.mean(metric_values)

    # Check that the reference and subject structures contain the same number
    # of C1' atoms.
    if reference_atom_array.shape[0] != subject_atom_array.shape[0]:
        raise ValueError("Reference and subject structures must contain the same number of C1' atoms.")
    
    superimposed, _ = biotite.structure.superimpose(
        reference_atom_array,
        subject_atom_array
    )
    c1_prime_rmsd = biotite.structure.rmsd(
        reference_atom_array,
        superimposed
    )

    c1_prime_lddt = biotite.structure.lddt(
        reference_atom_array,
        subject_atom_array
    )

    c1_prime_gddt = biotite.structure.lddt(
        reference_atom_array,
        subject_atom_array,
        inclusion_radius = 10000,
        distance_bins = (1.0, 2.0, 4.0, 8.0)
    )

    # Create the output dictionary.
    output_dict = {
        "reference_name": reference_output["name"],
        "reference_path": reference_path,
        "reference_sequence_length": reference_sequence_length,
        "subject_name": subject_output["name"],
        "subject_path": subject_path,
        "subject_sequence_length": subject_sequence_length,
        "best_start_idx": best_start_idx,
        "best_end_idx": best_end_idx,
        "sequence_recovery": sequence_recovery_result["sequence_recovery"],
        "eternafold_f1_score_pairs": eternafold_secondary_structure_result["f1_score_pairs"],
        "eternafold_f1_score_loops": eternafold_secondary_structure_result["f1_score_loops"],
        "ribonanza_net_f1_score_pairs": ribonanza_net_secondary_structure_result["mean_f1_score_pairs"],
        "ribonanza_net_f1_score_loops": ribonanza_net_secondary_structure_result["mean_f1_score_loops"],
        "ribonanza_net_eternafold_class_score": ribonanza_net_reactivity_profile_result["mean_eternafold_class_score"],
        "ribonanza_net_crossed_pair_quality_score": ribonanza_net_reactivity_profile_result["mean_crossed_pair_quality_score"],
        "ribonanza_net_openknot_score": ribonanza_net_reactivity_profile_result["mean_openknot_score"],
        "alphafold3_c1_prime_rmsd": float(c1_prime_rmsd),
        "alphafold3_c1_prime_lddt": c1_prime_lddt,
        "alphafold3_c1_prime_gddt": c1_prime_gddt,
        "alphafold3_ptm": subject_output["alphafold3"]["ptm"],
        "alphafold3_pae": subject_output["alphafold3"]["pae"],
        "alphafold3_plddt": subject_output["alphafold3"]["plddt"]
    }

    # Save the output dictionary to a JSON file.
    output_json_path = os.path.join(output_directory, 
                                    f"{subject_output['name']}.json")
    write_json_file(output_json_path, output_dict)



################################################################################
# Run from Command Line
################################################################################
if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argument_parser.add_argument(
        "--function_name", 
        type = str,
        help = "The name of the function to run."
    )
    argument_parser.add_argument(
        "--structure_path", 
        type = str,
        help = "The path to the structure file."
    )
    argument_parser.add_argument(
        "--overall_output_directory", 
        type = str,
        help = "The path to the overall output directory."
    )
    argument_parser.add_argument(
        "--num_samples", 
        type = int,
        help = "The number of samples to generate.",
        default = None
    )
    argument_parser.add_argument(
        "--temperature", 
        type = float,
        help = "The temperature for the sequence design algorithm.",
        default = None
    )
    argument_parser.add_argument(
        "--method", 
        type = str,
        help = "The method to use."
    )
    argument_parser.add_argument(
        "--na_mpnn_model_path", 
        type = str,
        help = "The path to the NA-MPNN model file.",
        default = None
    )
    argument_parser.add_argument(
        "--reference_structure_path", 
        type = str,
        help = "The path to the reference structure."
    )
    argument_parser.add_argument(
        "--subject_path", 
        type = str,
        help = "The path to the subject data."
    )
    argument_parser.add_argument(
        "--reference_path", 
        type = str,
        help = "The path to the reference data."
    )
    argument_parser.add_argument(
        "--reference_ppms_list_str",
        type = str,
        help = "The reference PPM list string."
    )
    argument_parser.add_argument(
        "--subject_paths",
        type = str,
        nargs = "+",
        help = "A list of paths to subject data files (for batch functions).",
        default = None
    )
    argument_parser.add_argument(
        "--skip_alphafold3",
        action = "store_true",
        help = "Skip AlphaFold3 and reuse existing AF3 results from the output JSON if available."
    )
    argument_parser.add_argument(
        "--model_mode",
        type = str,
        default = None,
        help = "Inference mode: 'ar' for autoregressive, 'dfm' for discrete flow matching."
    )
    argument_parser.add_argument(
        "--dfm_dt",
        type = float,
        default = None,
        help = "DFM Euler step size (only used when --model_mode dfm)."
    )
    argument_parser.add_argument(
        "--dfm_schedule",
        type = str,
        default = None,
        choices = [None, "uniform", "eds"],
        help = "DFM schedule: 'uniform' (default, uses --dfm_dt) or 'eds' (uses --eds_schedule_path)."
    )
    argument_parser.add_argument(
        "--eds_schedule_path",
        type = str,
        default = None,
        help = "Path to EDS schedule JSON (required when --dfm_schedule eds)."
    )
    argument_parser.add_argument(
        "--trajectory_dir",
        type = str,
        default = None,
        help = "If set, write per-step DFM sampling trajectories as JSONL into this directory (one file per PDB)."
    )
    argument_parser.add_argument(
        "--design_mode",
        type = str,
        default = "na",
        choices = ["na", "all"],
        help = "'na' to design only nucleic acid chains (default), 'all' to design all chains (protein + NA)."
    )

    # Parse the command line arguments.
    args = argument_parser.parse_args()

    if args.function_name == "design_nucleic_acid_sequence":
        design_nucleic_acid_sequence(args.structure_path,
                                     args.overall_output_directory,
                                     args.num_samples,
                                     args.temperature,
                                     method = args.method,
                                     na_mpnn_model_path = args.na_mpnn_model_path,
                                     model_mode = args.model_mode,
                                     dfm_dt = args.dfm_dt,
                                     dfm_schedule = args.dfm_schedule,
                                     eds_schedule_path = args.eds_schedule_path,
                                     trajectory_dir = args.trajectory_dir,
                                     design_mode = args.design_mode)
    elif args.function_name == "process_reference_monomer_rna":
        process_reference_monomer_rna(args.reference_structure_path,
                                      args.overall_output_directory)
    elif args.function_name == "process_design_monomer_rna":
        process_design_monomer_rna(args.subject_path,
                                   args.overall_output_directory,
                                   skip_alphafold3=args.skip_alphafold3)
    elif args.function_name == "process_designs_monomer_rna_batch":
        process_designs_monomer_rna_batch(args.subject_paths,
                                          args.overall_output_directory,
                                          skip_alphafold3=args.skip_alphafold3)
    elif args.function_name == "score_design_monomer_rna":
        score_design_monomer_rna(args.reference_path,
                                 args.subject_path,
                                 args.overall_output_directory)
    else:
        raise ValueError(f"Function {args.function_name} not recognized.")
