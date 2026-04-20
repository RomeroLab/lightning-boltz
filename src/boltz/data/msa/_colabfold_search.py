"""Vendored subset of colabfold.mmseqs.search (upstream 1.6.1).

This module is a direct, minimally-modified copy of the three public
functions used by ``colabfold_local.py``:

    * run_mmseqs
    * mmseqs_search_monomer
    * mmseqs_search_pair

Vendoring exists because the upstream ``colabfold`` PyPI package cannot be
co-installed with boltz: the 1.6.x series pins ``numpy>=2.0.2`` while boltz
pins ``numpy<2.0``, and 1.5.x transitively requires the heavyweight
``alphafold`` stack at import time (it raises RuntimeError on bare import).
The functions we need are pure subprocess wrappers around the ``mmseqs``
binary with only stdlib dependencies, so copying them is the clean way out.

Upstream source:
    https://github.com/sokrypton/ColabFold/blob/v1.6.1/colabfold/mmseqs/search.py
License: MIT (https://github.com/sokrypton/ColabFold/blob/main/LICENSE)
Copyright (c) 2021 Sergey Ovchinnikov

Deviations from upstream:
    * Dropped ``from colabfold.batch import ...`` / ``colabfold.utils`` imports
      — those are only used by the upstream ``main()`` CLI entrypoint, which
      is not vendored here.
    * Dropped the ``main()`` CLI entrypoint and its argparse setup.
    * Module docstring and attribution header are new.

If upstream adds useful improvements, re-vendor by copying the new search.py
body between the dashed markers below, preserving this header.
"""

# ----------------------------------------------------------------------
# BEGIN vendored colabfold/mmseqs/search.py body (v1.6.1)
# ----------------------------------------------------------------------

import logging
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Union

logger = logging.getLogger(__name__)

MODULE_OUTPUT_POS = {
    "align":        4,
    "convertalis":  4,
    "expandaln":    5,
    "filterresult": 4,
    "lndb":         2,
    "mergedbs":     2,
    "mvdb":         2,
    "pairaln":      4,
    "result2msa":   4,
    "search":       3,
}

def run_mmseqs(mmseqs: Path, params: List[Union[str, Path]]):
    module = params[0]
    if module in MODULE_OUTPUT_POS:
        output_pos = MODULE_OUTPUT_POS[module]
        output_path = Path(params[output_pos]).with_suffix('.dbtype')
        if output_path.exists():
            logger.info(f"Skipping {module} because {output_path} already exists")
            return

    params_log = " ".join(str(i) for i in params)
    logger.info(f"Running {mmseqs} {params_log}")
    # hide MMseqs2 verbose paramters list that clogs up the log
    os.environ["MMSEQS_CALL_DEPTH"] = "1"
    subprocess.check_call([mmseqs] + params)


def mmseqs_search_monomer(
    dbbase: Path,
    base: Path,
    uniref_db: Path = Path("uniref30_2302_db"),
    template_db: Path = Path(""),  # Unused by default
    metagenomic_db: Path = Path("colabfold_envdb_202108_db"),
    mmseqs: Path = Path("mmseqs"),
    use_env: bool = True,
    use_templates: bool = False,
    filter: bool = True,
    expand_eval: float = math.inf,
    align_eval: int = 10,
    diff: int = 3000,
    qsc: float = -20.0,
    max_accept: int = 1000000,
    prefilter_mode: int = 0,
    s: float = 8,
    db_load_mode: int = 2,
    threads: int = 32,
    gpu: int = 0,
    gpu_server: int = 0,
    unpack: bool = True,
):
    """Run mmseqs with a local colabfold database set

    db1: uniprot db (UniRef30)
    db2: Template (unused by default)
    db3: metagenomic db (colabfold_envdb_202108 or bfd_mgy_colabfold, the former is preferred)
    """
    if filter:
        # 0.1 was not used in benchmarks due to POSIX shell bug in line above
        #  EXPAND_EVAL=0.1
        align_eval = 10
        qsc = 0.8
        max_accept = 100000

    used_dbs = [uniref_db]
    if use_templates:
        used_dbs.append(template_db)
    if use_env:
        used_dbs.append(metagenomic_db)

    for db in used_dbs:
        if not dbbase.joinpath(f"{db}.dbtype").is_file():
            raise FileNotFoundError(f"Database {db} does not exist")
        if (
            (
                not dbbase.joinpath(f"{db}.idx").is_file()
                and not dbbase.joinpath(f"{db}.idx.index").is_file()
            )
            or os.environ.get("MMSEQS_IGNORE_INDEX", False)
        ):
            logger.info("Search does not use index")
            db_load_mode = 0
            dbSuffix1 = "_seq"
            dbSuffix2 = "_aln"
            dbSuffix3 = ""
        else:
            dbSuffix1 = ".idx"
            dbSuffix2 = ".idx"
            dbSuffix3 = ".idx"

    search_param = ["--num-iterations", "3", "--db-load-mode", str(db_load_mode), "-a", "-e", "0.1", "--max-seqs", "10000"]
    template_search_param = []
    if gpu:
        search_param += ["--gpu", str(gpu), "--prefilter-mode", "1"] # gpu version only supports ungapped prefilter currently
        template_search_param += ["--gpu", str(gpu), "--prefilter-mode", "1"]
    else:
        search_param += ["--prefilter-mode", str(prefilter_mode)]
        template_search_param += ["-s", "7.5", "--prefilter-mode", str(prefilter_mode)]
        if s is not None: # sensitivy can only be set for non-gpu version, gpu version runs at max sensitivity
            search_param += ["-s", "{:.1f}".format(s)]
        else:
            search_param += ["--k-score", "'seq:96,prof:80'"]
    if gpu_server:
        search_param += ["--gpu-server", str(gpu_server)]

    filter_param = ["--filter-msa", str(1 if filter else 0), "--filter-min-enable", "1000", "--diff", str(diff), "--qid", "0.0,0.2,0.4,0.6,0.8,1.0", "--qsc", "0", "--max-seq-id", "0.95",]
    expand_param = ["--expansion-mode", "0", "-e", str(expand_eval), "--expand-filter-clusters", str(1 if filter else 0), "--max-seq-id", "0.95",]

    if not base.joinpath("uniref.a3m").with_suffix('.a3m.dbtype').exists():
        run_mmseqs(mmseqs, ["search", base.joinpath("qdb"), dbbase.joinpath(uniref_db), base.joinpath("res"), base.joinpath("tmp"), "--threads", str(threads)] + search_param)
        run_mmseqs(mmseqs, ["mvdb", base.joinpath("tmp/latest/profile_1"), base.joinpath("prof_res")])
        run_mmseqs(mmseqs, ["lndb", base.joinpath("qdb_h"), base.joinpath("prof_res_h")])
        run_mmseqs(mmseqs, ["expandaln", base.joinpath("qdb"), dbbase.joinpath(f"{uniref_db}{dbSuffix1}"), base.joinpath("res"), dbbase.joinpath(f"{uniref_db}{dbSuffix2}"), base.joinpath("res_exp"), "--db-load-mode", str(db_load_mode), "--threads", str(threads)] + expand_param)
        run_mmseqs(mmseqs, ["align", base.joinpath("prof_res"), dbbase.joinpath(f"{uniref_db}{dbSuffix1}"), base.joinpath("res_exp"), base.joinpath("res_exp_realign"), "--db-load-mode", str(db_load_mode), "-e", str(align_eval), "--max-accept", str(max_accept), "--threads", str(threads), "--alt-ali", "10", "-a"])
        run_mmseqs(mmseqs, ["filterresult", base.joinpath("qdb"), dbbase.joinpath(f"{uniref_db}{dbSuffix1}"),
                            base.joinpath("res_exp_realign"), base.joinpath("res_exp_realign_filter"), "--db-load-mode",
                            str(db_load_mode), "--qid", "0", "--qsc", str(qsc), "--diff", "0", "--threads",
                            str(threads), "--max-seq-id", "1.0", "--filter-min-enable", "100"])
        run_mmseqs(mmseqs, ["result2msa", base.joinpath("qdb"), dbbase.joinpath(f"{uniref_db}{dbSuffix1}"),
                            base.joinpath("res_exp_realign_filter"), base.joinpath("uniref.a3m"), "--msa-format-mode",
                            "6", "--db-load-mode", str(db_load_mode), "--threads", str(threads)] + filter_param)
        run_mmseqs(mmseqs, ["rmdb", base.joinpath("res_exp_realign_filter")])
        run_mmseqs(mmseqs, ["rmdb", base.joinpath("res_exp_realign")])
        run_mmseqs(mmseqs, ["rmdb", base.joinpath("res_exp")])
        run_mmseqs(mmseqs, ["rmdb", base.joinpath("res")])
    else:
        logger.info(f"Skipping {uniref_db} search because uniref.a3m already exists")

    if use_env and not base.joinpath("bfd.mgnify30.metaeuk30.smag30.a3m").with_suffix('.a3m.dbtype').exists():
        run_mmseqs(mmseqs, ["search", base.joinpath("prof_res"), dbbase.joinpath(metagenomic_db), base.joinpath("res_env"),
                            base.joinpath("tmp3"), "--threads", str(threads)] + search_param)
        run_mmseqs(mmseqs, ["expandaln", base.joinpath("prof_res"), dbbase.joinpath(f"{metagenomic_db}{dbSuffix1}"), base.joinpath("res_env"),
                            dbbase.joinpath(f"{metagenomic_db}{dbSuffix2}"), base.joinpath("res_env_exp"), "-e", str(expand_eval),
                            "--expansion-mode", "0", "--db-load-mode", str(db_load_mode), "--threads", str(threads)])
        run_mmseqs(mmseqs, ["align", base.joinpath("tmp3/latest/profile_1"), dbbase.joinpath(f"{metagenomic_db}{dbSuffix1}"),
                            base.joinpath("res_env_exp"), base.joinpath("res_env_exp_realign"), "--db-load-mode",
                            str(db_load_mode), "-e", str(align_eval), "--max-accept", str(max_accept), "--threads",
                            str(threads), "--alt-ali", "10", "-a"])
        run_mmseqs(mmseqs, ["filterresult", base.joinpath("qdb"), dbbase.joinpath(f"{metagenomic_db}{dbSuffix1}"),
                            base.joinpath("res_env_exp_realign"), base.joinpath("res_env_exp_realign_filter"),
                            "--db-load-mode", str(db_load_mode), "--qid", "0", "--qsc", str(qsc), "--diff", "0",
                            "--max-seq-id", "1.0", "--threads", str(threads), "--filter-min-enable", "100"])
        run_mmseqs(mmseqs, ["result2msa", base.joinpath("qdb"), dbbase.joinpath(f"{metagenomic_db}{dbSuffix1}"),
                            base.joinpath("res_env_exp_realign_filter"),
                            base.joinpath("bfd.mgnify30.metaeuk30.smag30.a3m"), "--msa-format-mode", "6",
                            "--db-load-mode", str(db_load_mode), "--threads", str(threads)] + filter_param)
        run_mmseqs(mmseqs, ["rmdb", base.joinpath("res_env_exp_realign_filter")])
        run_mmseqs(mmseqs, ["rmdb", base.joinpath("res_env_exp_realign")])
        run_mmseqs(mmseqs, ["rmdb", base.joinpath("res_env_exp")])
        run_mmseqs(mmseqs, ["rmdb", base.joinpath("res_env")])
    elif use_env:
        logger.info(f"Skipping {metagenomic_db} search because bfd.mgnify30.metaeuk30.smag30.a3m already exists")

    if use_templates and not base.joinpath(f"{template_db}.m8").with_suffix('.m8.dbtype').exists():
        run_mmseqs(mmseqs, ["search", base.joinpath("prof_res"), dbbase.joinpath(template_db), base.joinpath("res_pdb"),
                            base.joinpath("tmp2"), "--db-load-mode", str(db_load_mode), "--threads", str(threads), "-a", "-e", "0.1"] + template_search_param)
        run_mmseqs(mmseqs, ["convertalis", base.joinpath("prof_res"), dbbase.joinpath(f"{template_db}{dbSuffix3}"), base.joinpath("res_pdb"),
                            base.joinpath(f"{template_db}"), "--format-output",
                            "query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits,cigar",
                            "--db-output", "1",
                            "--db-load-mode", str(db_load_mode), "--threads", str(threads)])
        run_mmseqs(mmseqs, ["rmdb", base.joinpath("res_pdb")])
    elif use_templates:
        logger.info(f"Skipping {template_db} search because {template_db}.m8 already exists")

    if use_env:
        run_mmseqs(mmseqs, ["mergedbs", base.joinpath("qdb"), base.joinpath("final.a3m"), base.joinpath("uniref.a3m"), base.joinpath("bfd.mgnify30.metaeuk30.smag30.a3m")])
        run_mmseqs(mmseqs, ["rmdb", base.joinpath("bfd.mgnify30.metaeuk30.smag30.a3m")])
        run_mmseqs(mmseqs, ["rmdb", base.joinpath("uniref.a3m")])
    else:
        run_mmseqs(mmseqs, ["mvdb", base.joinpath("uniref.a3m"), base.joinpath("final.a3m")])
        run_mmseqs(mmseqs, ["rmdb", base.joinpath("uniref.a3m")])

    if unpack:
        run_mmseqs(mmseqs, ["unpackdb", base.joinpath("final.a3m"), base.joinpath("."), "--unpack-name-mode", "0", "--unpack-suffix", ".a3m"])
        run_mmseqs(mmseqs, ["rmdb", base.joinpath("final.a3m")])

        if use_templates:
            run_mmseqs(mmseqs, ["unpackdb", base.joinpath(f"{template_db}"), base.joinpath("."), "--unpack-name-mode", "0", "--unpack-suffix", ".m8"])
            if base.joinpath(f"{template_db}").exists():
                run_mmseqs(mmseqs, ["rmdb", base.joinpath(f"{template_db}")])

    run_mmseqs(mmseqs, ["rmdb", base.joinpath("prof_res")])
    run_mmseqs(mmseqs, ["rmdb", base.joinpath("prof_res_h")])
    shutil.rmtree(base.joinpath("tmp"))
    if use_templates:
        shutil.rmtree(base.joinpath("tmp2"))
    if use_env:
        shutil.rmtree(base.joinpath("tmp3"))

def mmseqs_search_pair(
    dbbase: Path,
    base: Path,
    uniref_db: Path = Path("uniref30_2302_db"),
    spire_db: Path = Path("spire_ctg10_2401_db"),
    mmseqs: Path = Path("mmseqs"),
    pair_env: bool = True,
    filter: bool = False,
    prefilter_mode: int = 0,
    s: float = 8,
    threads: int = 64,
    gpu: bool = False,
    gpu_server: bool = False,
    db_load_mode: int = 2,
    pairing_strategy: int = 0,
    unpack: bool = True,
):
    if not dbbase.joinpath(f"{uniref_db}.dbtype").is_file():
        raise FileNotFoundError(f"Database {uniref_db} does not exist")
    if (
        (
            not dbbase.joinpath(f"{uniref_db}.idx").is_file()
            and not dbbase.joinpath(f"{uniref_db}.idx.index").is_file()
        )
        or os.environ.get("MMSEQS_IGNORE_INDEX", False)
    ):
        logger.info("Search does not use index")
        db_load_mode = 0
        dbSuffix1 = "_seq"
        dbSuffix2 = "_aln"
    else:
        dbSuffix1 = ".idx"
        dbSuffix2 = ".idx"

    if pair_env:
        db = spire_db
        output = ".env.paired.a3m"
    else:
        db = uniref_db
        output = ".paired.a3m"

    # fmt: off
    # @formatter:off
    search_param = ["--num-iterations", "3", "--db-load-mode", str(db_load_mode), "-a", "-e", "0.1", "--max-seqs", "10000",]
    if gpu:
        search_param += ["--gpu", str(gpu), "--prefilter-mode", "1"] # gpu version only supports ungapped prefilter currently
    else:
        search_param += ["--prefilter-mode", str(prefilter_mode)]
        if s is not None: # sensitivy can only be set for non-gpu version, gpu version runs at max sensitivity
            search_param += ["-s", "{:.1f}".format(s)]
        else:
            search_param += ["--k-score", "'seq:96,prof:80'"]
    if gpu_server:
        search_param += ["--gpu-server", str(gpu_server)]
    expand_param = ["--expansion-mode", "0", "-e", "inf", "--expand-filter-clusters", "0", "--max-seq-id", "0.95",]
    filter_param = ["--filter-msa", str(1 if filter else 0), "--filter-min-enable", "1000", "--diff", "3000", "--qid", "0.2,0.4,0.6,0.8,1.0", "--qsc", "0", "--max-seq-id", "0.95",]
    run_mmseqs(mmseqs, ["search", base.joinpath("qdb"), dbbase.joinpath(db), base.joinpath("res"), base.joinpath("tmp"), "--threads", str(threads),] + search_param,)
    run_mmseqs(mmseqs, ["mvdb", base.joinpath("tmp/latest/profile_1"), base.joinpath("prof_res")])
    run_mmseqs(mmseqs, ["lndb", base.joinpath("qdb_h"), base.joinpath("prof_res_h")])
    run_mmseqs(mmseqs, ["expandaln", base.joinpath("qdb"), dbbase.joinpath(f"{db}{dbSuffix1}"), base.joinpath("res"), dbbase.joinpath(f"{db}{dbSuffix2}"), base.joinpath("res_exp"), "--db-load-mode", str(db_load_mode), "--threads", str(threads),] + expand_param,)
    run_mmseqs(mmseqs, ["align", base.joinpath("prof_res"), dbbase.joinpath(f"{db}{dbSuffix1}"), base.joinpath("res_exp"), base.joinpath("res_exp_realign"), "--db-load-mode", str(db_load_mode), "--alignment-mode", "1", "-e", "0.001", "--max-accept", "1000000", "--threads", str(threads),],)
    run_mmseqs(mmseqs, ["pairaln", base.joinpath("qdb"), dbbase.joinpath(f"{db}"), base.joinpath("res_exp_realign"), base.joinpath("res_exp_realign_pair"), "--db-load-mode", str(db_load_mode), "--pairing-mode", str(pairing_strategy), "--pairing-dummy-mode", "0", "--threads", str(threads), ],)
    run_mmseqs(mmseqs, ["align", base.joinpath("prof_res"), dbbase.joinpath(f"{db}{dbSuffix1}"), base.joinpath("res_exp_realign_pair"), base.joinpath("res_exp_realign_pair_bt"), "--db-load-mode", str(db_load_mode), "-e", "inf", "-a", "--threads", str(threads), ],)
    run_mmseqs(mmseqs, ["pairaln", base.joinpath("qdb"), dbbase.joinpath(f"{db}"), base.joinpath("res_exp_realign_pair_bt"), base.joinpath("res_final"), "--db-load-mode", str(db_load_mode), "--pairing-mode", str(pairing_strategy), "--pairing-dummy-mode", "1", "--threads", str(threads),],)
    run_mmseqs(mmseqs, ["result2msa", base.joinpath("qdb"), dbbase.joinpath(f"{db}{dbSuffix1}"), base.joinpath("res_final"), base.joinpath("pair.a3m"), "--db-load-mode", str(db_load_mode),  "--msa-format-mode", "5", "--threads", str(threads),] + filter_param,)
    if unpack:
        run_mmseqs(mmseqs, ["unpackdb", base.joinpath("pair.a3m"), base.joinpath("."), "--unpack-name-mode", "0", "--unpack-suffix", output,],)
        run_mmseqs(mmseqs, ["rmdb", base.joinpath("pair.a3m")])
    run_mmseqs(mmseqs, ["rmdb", base.joinpath("res")])
    run_mmseqs(mmseqs, ["rmdb", base.joinpath("res_exp")])
    run_mmseqs(mmseqs, ["rmdb", base.joinpath("res_exp_realign")])
    run_mmseqs(mmseqs, ["rmdb", base.joinpath("res_exp_realign_pair")])
    run_mmseqs(mmseqs, ["rmdb", base.joinpath("res_exp_realign_pair_bt")])
    run_mmseqs(mmseqs, ["rmdb", base.joinpath("res_final")])
    run_mmseqs(mmseqs, ["rmdb", base.joinpath("prof_res")])
    run_mmseqs(mmseqs, ["rmdb", base.joinpath("prof_res_h")])
    shutil.rmtree(base.joinpath("tmp"))
    # @formatter:on
    # fmt: on

# ----------------------------------------------------------------------
# END vendored colabfold/mmseqs/search.py body
# ----------------------------------------------------------------------
