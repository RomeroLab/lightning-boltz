#!/bin/bash
# Copyright 2026 Romero Lab, Duke University
# MIT License (same as Boltz)
#
# Download and prepare MMseqs2 GPU-padded databases for Boltz.
# Based on ColabFold's setup_databases.sh with GPU-padded conversion.
#
# This script supports two database configurations:
#   1. ColabFold databases (default): UniRef30 + ColabFold env DB
#      - Smaller, faster to download (~150 GB total)
#      - Matches what the ColabFold server uses
#      - Includes taxonomy files for paired MSA generation
#   2. AlphaFold3 databases: UniRef90 + MGnify + Small BFD + UniProt
#      - Larger, more comprehensive (~800 GB total)
#      - Same databases used by AlphaFast
#
# Usage:
#   ./scripts/setup_boltz_mmseqs_dbs.sh <target_dir> [OPTIONS]
#
# Options:
#   --mode colabfold    Use ColabFold databases (default, recommended)
#   --mode alphafold3   Use AlphaFold3 databases
#   --source SOURCE     Download source: 'mmseqs' (default) or 'huggingface'
#   --hf-repo REPO      HuggingFace dataset repo (default: boltz-community/mmseqs-databases)
#   --keep-raw          Keep raw files after conversion (default: remove)
#   --threads N         Number of threads for conversion (default: all cores)
#   --skip-download     Skip download, only convert existing files
#   --with-uniprot      Also download UniProt for paired MSA (colabfold mode)
#   --verify            Validate existing databases and exit (no download)
#
# Output directory structure:
#   <target_dir>/
#     uniref30_2302_db*                # (colabfold mode) profile DB
#     uniref30_2302_db_mapping         # (colabfold mode) taxonomy mapping
#     uniref30_2302_db_taxonomy        # (colabfold mode) taxonomy
#     colabfold_envdb_202108_db*       # (colabfold mode)
#     uniprot_padded*                  # (colabfold mode, optional for pairing)
#     --- OR ---
#     uniref90_padded*                 # (alphafold3 mode)
#     mgnify_padded*                   # (alphafold3 mode)
#     small_bfd_padded*                # (alphafold3 mode)
#     uniprot_padded*                  # (alphafold3 mode)
#
# Requirements:
#   - mmseqs (GPU version) in PATH
#   - wget, curl, or aria2c
#   - tar, zstd (for alphafold3 mode)
#   - ~150 GB free (colabfold) or ~800 GB free (alphafold3)

set -euo pipefail

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
usage() {
    echo "Usage: $0 <target_dir> [OPTIONS]"
    echo ""
    echo "Download and prepare MMseqs2 GPU-padded databases for Boltz."
    echo ""
    echo "Arguments:"
    echo "  target_dir           Directory where databases will be stored"
    echo ""
    echo "Options:"
    echo "  --mode MODE          Database mode: 'colabfold' (default) or 'alphafold3'"
    echo "  --source SOURCE      Download source: 'mmseqs' (default) or 'huggingface'"
    echo "  --hf-repo REPO       HuggingFace dataset repo (default: boltz-community/mmseqs-databases)"
    echo "  --keep-raw           Keep raw files after conversion"
    echo "  --threads N          Number of threads for conversion (default: all cores)"
    echo "  --skip-download      Skip download, only convert existing files"
    echo "  --with-uniprot       Also download UniProt for paired MSA (colabfold mode)"
    echo "  --verify             Validate existing databases and exit"
    echo ""
    echo "Examples:"
    echo "  # Default: ColabFold databases (from mmseqs.org)"
    echo "  $0 /data/boltz_dbs"
    echo ""
    echo "  # ColabFold databases from HuggingFace (pre-indexed, no conversion)"
    echo "  $0 /data/boltz_dbs --source huggingface"
    echo ""
    echo "  # ColabFold databases with UniProt for multi-chain pairing"
    echo "  $0 /data/boltz_dbs --with-uniprot"
    echo ""
    echo "  # AlphaFold3 databases (reuse existing AlphaFast DBs)"
    echo "  $0 /data/boltz_dbs --mode alphafold3"
    exit 1
}

if [ "$#" -lt 1 ]; then
    usage
fi

TARGET_DIR="$1"
shift

MODE="colabfold"
SOURCE="mmseqs"
HF_REPO="boltz-community/mmseqs-databases"
KEEP_RAW=false
THREADS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 8)
SKIP_DOWNLOAD=false
WITH_UNIPROT=false
VERIFY_ONLY=false

while [ "$#" -gt 0 ]; do
    case "$1" in
        --mode)        MODE="$2"; shift 2 ;;
        --source)      SOURCE="$2"; shift 2 ;;
        --hf-repo)     HF_REPO="$2"; shift 2 ;;
        --keep-raw)    KEEP_RAW=true; shift ;;
        --threads)     THREADS="$2"; shift 2 ;;
        --skip-download) SKIP_DOWNLOAD=true; shift ;;
        --with-uniprot)  WITH_UNIPROT=true; shift ;;
        --verify)        VERIFY_ONLY=true; shift ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

if [ "$MODE" != "colabfold" ] && [ "$MODE" != "alphafold3" ]; then
    echo "ERROR: Invalid mode '$MODE'. Use 'colabfold' or 'alphafold3'."
    exit 1
fi

if [ "$SOURCE" != "mmseqs" ] && [ "$SOURCE" != "huggingface" ]; then
    echo "ERROR: Invalid source '$SOURCE'. Use 'mmseqs' or 'huggingface'."
    exit 1
fi

# ---------------------------------------------------------------------------
# Check prerequisites
# ---------------------------------------------------------------------------
if ! command -v mmseqs &> /dev/null; then
    echo "ERROR: mmseqs is not installed or not in PATH."
    echo ""
    echo "Install MMseqs2 with GPU support:"
    echo "  wget https://mmseqs.com/latest/mmseqs-linux-gpu.tar.gz"
    echo "  tar xzf mmseqs-linux-gpu.tar.gz"
    echo "  sudo cp mmseqs/bin/mmseqs /usr/local/bin/"
    exit 1
fi

echo "Using MMseqs2 version: $(mmseqs version 2>/dev/null || echo 'unknown')"

# ---------------------------------------------------------------------------
# Verify mode: check existing databases and exit
# ---------------------------------------------------------------------------
if [ "$VERIFY_ONLY" = true ]; then
    echo ""
    echo "=========================================="
    echo "Verifying databases in: $TARGET_DIR"
    echo "=========================================="
    VERIFY_OK=true

    verify_db() {
        local db_path="$1"
        local db_name="$2"
        if [ -f "${db_path}.dbtype" ]; then
            # Check that all required companion files exist
            local has_data=false
            if [ -f "${db_path}" ] || [ -f "${db_path}.0" ]; then
                has_data=true
            fi
            if [ "$has_data" = true ] && [ -f "${db_path}.index" ]; then
                local entry_count
                entry_count=$(wc -l < "${db_path}.index" | tr -d ' ')
                echo "  OK: ${db_name} (${entry_count} entries)"
            else
                echo "  INCOMPLETE: ${db_name} - missing data or index files"
                VERIFY_OK=false
            fi
        else
            echo "  MISSING: ${db_name}"
            VERIFY_OK=false
        fi
    }

    if [ "$MODE" = "colabfold" ]; then
        verify_db "${TARGET_DIR}/uniref30_2302_db" "uniref30_2302_db"
        verify_db "${TARGET_DIR}/colabfold_envdb_202108_db" "colabfold_envdb_202108_db"
        if [ -f "${TARGET_DIR}/uniref30_2302_db_mapping" ]; then
            echo "  OK: taxonomy mapping"
        else
            echo "  MISSING: taxonomy mapping (needed for paired MSA)"
        fi
        if [ -f "${TARGET_DIR}/uniref30_2302_db.idx" ]; then
            echo "  OK: uniref30 index"
        else
            echo "  MISSING: uniref30 index (will be slower without it)"
        fi
        if [ -f "${TARGET_DIR}/uniprot_padded.dbtype" ]; then
            verify_db "${TARGET_DIR}/uniprot_padded" "uniprot_padded"
        else
            echo "  INFO: uniprot_padded not found (optional, for multi-chain pairing)"
        fi
    elif [ "$MODE" = "alphafold3" ]; then
        for db_name in uniref90 mgnify small_bfd uniprot; do
            verify_db "${TARGET_DIR}/${db_name}_padded" "${db_name}_padded"
        done
    fi

    echo ""
    if [ "$VERIFY_OK" = true ]; then
        echo "All required databases are present."
        echo ""
        echo "Test with:"
        echo "  boltz predict input.yaml --use_mmseqs_gpu --mmseqs_db_dir $TARGET_DIR"
        exit 0
    else
        echo "Some databases are missing or incomplete."
        echo "Run without --verify to download/convert them."
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# Download helper (matches ColabFold's strategy: aria2c > curl > wget)
# ---------------------------------------------------------------------------
downloadFile() {
    local URL="$1"
    local OUTPUT="$2"
    local FILENAME
    FILENAME=$(basename "$OUTPUT")
    local DIR
    DIR=$(dirname "$OUTPUT")

    # All tools support resume (-c / -C - / --continue) for interrupted downloads
    if command -v aria2c &> /dev/null; then
        aria2c --max-connection-per-server=8 --allow-overwrite=true \
            --continue=true \
            -o "$FILENAME" -d "$DIR" "$URL" && return 0
    fi
    if command -v curl &> /dev/null; then
        curl -L -C - -o "$OUTPUT" "$URL" && return 0
    fi
    if command -v wget &> /dev/null; then
        wget -c --progress=bar:force:noscroll -O "$OUTPUT" "$URL" && return 0
    fi

    echo "ERROR: No download tool found (tried aria2c, curl, wget)"
    exit 1
}

mkdir -p "$TARGET_DIR"

echo "=========================================="
echo "Boltz MMseqs2-GPU Database Setup"
echo "=========================================="
echo "Target directory: $TARGET_DIR"
echo "Mode:            $MODE"
echo "Source:          $SOURCE"
echo "Threads:         $THREADS"
echo "Keep raw files:  $KEEP_RAW"
echo "Start time:      $(date)"
echo "=========================================="
echo ""

# Make MMseqs2 merge databases to avoid file spam (from ColabFold)
export MMSEQS_FORCE_MERGE=1

# GPU index parameters (skip large k-mer index, not needed for GPU)
GPU_INDEX_PAR="--split 1 --index-subset 2"

# ---------------------------------------------------------------------------
# HuggingFace download helper
# ---------------------------------------------------------------------------
downloadFromHF() {
    local FILENAME="$1"
    local OUTPUT="$2"
    local HF_URL="https://huggingface.co/datasets/${HF_REPO}/resolve/main/${FILENAME}"

    echo "  Downloading from HuggingFace: ${FILENAME}..."
    downloadFile "$HF_URL" "$OUTPUT"
}

# ---------------------------------------------------------------------------
# HuggingFace source: download pre-indexed tarballs (no conversion needed)
# ---------------------------------------------------------------------------
if [ "$SOURCE" = "huggingface" ] && [ "$VERIFY_ONLY" = false ] && [ "$SKIP_DOWNLOAD" = false ]; then
    echo "=== Downloading pre-indexed databases from HuggingFace ==="
    echo "  Repo: ${HF_REPO}"
    echo ""

    if [ "$MODE" = "colabfold" ]; then
        # ColabFold databases tarball (UniRef30 + envDB + taxonomy, pre-indexed)
        if [ -f "${TARGET_DIR}/uniref30_2302_db.dbtype" ] && [ -f "${TARGET_DIR}/colabfold_envdb_202108_db.dbtype" ]; then
            echo "SKIP: ColabFold databases already exist"
        else
            downloadFromHF "colabfold_dbs.tar.gz" "${TARGET_DIR}/colabfold_dbs.tar.gz"
            echo "Extracting ColabFold databases..."
            tar xzf "${TARGET_DIR}/colabfold_dbs.tar.gz" -C "$TARGET_DIR"
            rm -f "${TARGET_DIR}/colabfold_dbs.tar.gz"
            echo "Done: ColabFold databases extracted"
        fi

        # UniProt (optional)
        if [ "$WITH_UNIPROT" = true ]; then
            if [ -f "${TARGET_DIR}/uniprot_padded.dbtype" ]; then
                echo "SKIP: UniProt padded database already exists"
            else
                downloadFromHF "uniprot_padded.tar.gz" "${TARGET_DIR}/uniprot_padded.tar.gz"
                echo "Extracting UniProt padded database..."
                tar xzf "${TARGET_DIR}/uniprot_padded.tar.gz" -C "$TARGET_DIR"
                rm -f "${TARGET_DIR}/uniprot_padded.tar.gz"
                echo "Done: UniProt padded database extracted"
            fi
        fi
    elif [ "$MODE" = "alphafold3" ]; then
        echo "ERROR: HuggingFace source is only supported for colabfold mode."
        echo "       Use --source mmseqs for alphafold3 mode."
        exit 1
    fi

    echo ""
    echo "HuggingFace download complete. Databases are pre-indexed — no conversion needed."
    echo ""

    # Skip the rest of the download/convert logic — jump to summary
    # (We use a flag to skip the mmseqs-based download sections below)
    SKIP_DOWNLOAD=true
fi

# ---------------------------------------------------------------------------
# Helper: convert to padded database
# ---------------------------------------------------------------------------
convert_to_padded() {
    local db_name="$1"
    local source_db="$2"
    local target_padded="${TARGET_DIR}/${db_name}_padded"

    if [ -f "${target_padded}.dbtype" ]; then
        echo "  SKIP: ${db_name}_padded already exists"
        return 0
    fi

    if [ ! -f "${source_db}.dbtype" ]; then
        echo "  ERROR: Source database not found: ${source_db}.dbtype"
        return 1
    fi

    echo "  Creating padded database for GPU acceleration..."
    time mmseqs makepaddedseqdb "$source_db" "$target_padded" --threads "$THREADS"

    if [ -f "${target_padded}.dbtype" ]; then
        echo "  SUCCESS: Created ${target_padded}"
    else
        echo "  ERROR: Failed to create padded database"
        return 1
    fi
}

# ---------------------------------------------------------------------------
# ColabFold mode
# ---------------------------------------------------------------------------
if [ "$MODE" = "colabfold" ]; then
    echo "=== Downloading ColabFold databases ==="
    echo ""

    UNIREF30DB="uniref30_2302"
    CFDB="colabfold_envdb_202108"

    # ---- UniRef30 ----
    if [ -f "${TARGET_DIR}/${UNIREF30DB}_db.dbtype" ] || [ -f "${TARGET_DIR}/${UNIREF30DB}_padded.dbtype" ]; then
        echo "SKIP: UniRef30 database already exists"
    elif [ "$SKIP_DOWNLOAD" = true ]; then
        echo "SKIP: Download skipped (--skip-download)"
    else
        echo "Downloading UniRef30 2302 (prebuilt GPU+CPU database)..."
        downloadFile \
            "https://opendata.mmseqs.org/colabfold/${UNIREF30DB}.db.tar.gz" \
            "${TARGET_DIR}/${UNIREF30DB}.tar.gz"
        echo "Extracting UniRef30..."
        tar xzf "${TARGET_DIR}/${UNIREF30DB}.tar.gz" -C "$TARGET_DIR"
        if [ "$KEEP_RAW" = false ]; then
            rm -f "${TARGET_DIR}/${UNIREF30DB}.tar.gz"
        fi
    fi

    # Download updated taxonomy/pairing files (needed for pairaln-based pairing)
    if [ ! -f "${TARGET_DIR}/${UNIREF30DB}_db_mapping" ] || [ ! -f "${TARGET_DIR}/${UNIREF30DB}_db_taxonomy" ]; then
        if [ "$SKIP_DOWNLOAD" = false ]; then
            echo "Downloading updated taxonomy files for pairing..."
            downloadFile \
                "https://opendata.mmseqs.org/colabfold/${UNIREF30DB}_newtaxonomy.tar.gz" \
                "${TARGET_DIR}/${UNIREF30DB}_newtaxonomy.tar.gz"
            tar xzf "${TARGET_DIR}/${UNIREF30DB}_newtaxonomy.tar.gz" -C "$TARGET_DIR"
            rm -f "${TARGET_DIR}/${UNIREF30DB}_newtaxonomy.tar.gz"
        fi
    fi

    # Create binary taxonomy mapping if needed (faster loading during pairing)
    if [ -f "${TARGET_DIR}/${UNIREF30DB}_db_mapping" ]; then
        TAXHEADER=$(od -An -N4 -t x4 "${TARGET_DIR}/${UNIREF30DB}_db_mapping" | tr -d ' ')
        if [ "${TAXHEADER}" != "0c170013" ]; then
            echo "Creating binary taxonomy mapping..."
            mmseqs createbintaxmapping \
                "${TARGET_DIR}/${UNIREF30DB}_db_mapping" \
                "${TARGET_DIR}/${UNIREF30DB}_db_mapping.bin"
            mv -f -- "${TARGET_DIR}/${UNIREF30DB}_db_mapping.bin" "${TARGET_DIR}/${UNIREF30DB}_db_mapping"
        fi
        # Symlink for index-based access
        ln -sf "${UNIREF30DB}_db_mapping" "${TARGET_DIR}/${UNIREF30DB}_db.idx_mapping" 2>/dev/null || true
    fi
    if [ -f "${TARGET_DIR}/${UNIREF30DB}_db_taxonomy" ]; then
        ln -sf "${UNIREF30DB}_db_taxonomy" "${TARGET_DIR}/${UNIREF30DB}_db.idx_taxonomy" 2>/dev/null || true
    fi

    # Create index for UniRef30
    if [ -f "${TARGET_DIR}/${UNIREF30DB}_db.dbtype" ] && [ ! -f "${TARGET_DIR}/${UNIREF30DB}_db.idx" ]; then
        echo "Creating index for UniRef30..."
        mmseqs createindex "${TARGET_DIR}/${UNIREF30DB}_db" "${TARGET_DIR}/tmp_uniref" \
            --remove-tmp-files 1 ${GPU_INDEX_PAR}
    fi
    echo ""

    # ---- ColabFold environmental database ----
    if [ -f "${TARGET_DIR}/${CFDB}_db.dbtype" ] || [ -f "${TARGET_DIR}/${CFDB}_padded.dbtype" ]; then
        echo "SKIP: ColabFold env database already exists"
    elif [ "$SKIP_DOWNLOAD" = true ]; then
        echo "SKIP: Download skipped (--skip-download)"
    else
        echo "Downloading ColabFold environmental DB (prebuilt GPU+CPU database)..."
        downloadFile \
            "https://opendata.mmseqs.org/colabfold/${CFDB}.db.tar.gz" \
            "${TARGET_DIR}/${CFDB}.tar.gz"
        echo "Extracting ColabFold env DB..."
        tar xzf "${TARGET_DIR}/${CFDB}.tar.gz" -C "$TARGET_DIR"
        if [ "$KEEP_RAW" = false ]; then
            rm -f "${TARGET_DIR}/${CFDB}.tar.gz"
        fi
    fi

    # Create index for env DB
    if [ -f "${TARGET_DIR}/${CFDB}_db.dbtype" ] && [ ! -f "${TARGET_DIR}/${CFDB}_db.idx" ]; then
        echo "Creating index for ColabFold env DB..."
        mmseqs createindex "${TARGET_DIR}/${CFDB}_db" "${TARGET_DIR}/tmp_envdb" \
            --remove-tmp-files 1 ${GPU_INDEX_PAR}
    fi
    echo ""

    # ---- UniProt (optional, for multi-chain pairing) ----
    if [ "$WITH_UNIPROT" = true ]; then
        UNIPROT_FASTA="uniprot_all_2021_04.fa"
        UNIPROT_DB="${TARGET_DIR}/uniprot"
        AF3_SOURCE="https://storage.googleapis.com/alphafold-databases/v3.0"

        if [ -f "${TARGET_DIR}/uniprot_padded.dbtype" ]; then
            echo "SKIP: UniProt padded database already exists"
        elif [ -f "${UNIPROT_DB}.dbtype" ]; then
            echo "UniProt MMseqs2 DB found, will convert to padded..."
        elif [ "$SKIP_DOWNLOAD" = true ]; then
            echo "SKIP: Download skipped (--skip-download)"
        else
            echo "Downloading UniProt (~65 GB compressed)..."
            wget --progress=bar:force:noscroll -O - \
                "${AF3_SOURCE}/${UNIPROT_FASTA}.zst" | \
                zstd --decompress > "${TARGET_DIR}/${UNIPROT_FASTA}"

            echo "Creating MMseqs2 database from UniProt FASTA..."
            mmseqs createdb "${TARGET_DIR}/${UNIPROT_FASTA}" "$UNIPROT_DB" --threads "$THREADS"

            if [ "$KEEP_RAW" = false ]; then
                rm -f "${TARGET_DIR}/${UNIPROT_FASTA}"
            fi
        fi

        # Convert UniProt to padded
        if [ -f "${UNIPROT_DB}.dbtype" ]; then
            convert_to_padded "uniprot" "$UNIPROT_DB"
            if [ "$KEEP_RAW" = false ] && [ -f "${TARGET_DIR}/uniprot_padded.dbtype" ]; then
                echo "  Cleaning up raw UniProt database..."
                rm -f "${UNIPROT_DB}" "${UNIPROT_DB}".dbtype "${UNIPROT_DB}".index \
                      "${UNIPROT_DB}".lookup "${UNIPROT_DB}".source \
                      "${UNIPROT_DB}_h" "${UNIPROT_DB}_h".dbtype "${UNIPROT_DB}_h".index
            fi
        fi
        echo ""
    fi

fi

# ---------------------------------------------------------------------------
# AlphaFold3 mode
# ---------------------------------------------------------------------------
if [ "$MODE" = "alphafold3" ]; then
    echo "=== Downloading AlphaFold3 databases ==="
    echo ""

    AF3_SOURCE="https://storage.googleapis.com/alphafold-databases/v3.0"

    # Database mapping: db_name -> fasta_filename
    declare -A DATABASES=(
        ["uniref90"]="uniref90_2022_05.fa"
        ["mgnify"]="mgy_clusters_2022_05.fa"
        ["small_bfd"]="bfd-first_non_consensus_sequences.fasta"
        ["uniprot"]="uniprot_all_2021_04.fa"
    )

    # Download FASTA files
    for db_name in "${!DATABASES[@]}"; do
        fasta_file="${DATABASES[$db_name]}"
        target_path="${TARGET_DIR}/${fasta_file}"

        if [ -f "${TARGET_DIR}/${db_name}_padded.dbtype" ]; then
            echo "SKIP: ${db_name}_padded already exists"
        elif [ -f "$target_path" ]; then
            echo "SKIP: $fasta_file already downloaded"
        elif [ "$SKIP_DOWNLOAD" = true ]; then
            echo "SKIP: Download skipped (--skip-download)"
        else
            echo "Downloading $db_name ($fasta_file)..."
            wget --progress=bar:force:noscroll -O - \
                "${AF3_SOURCE}/${fasta_file}.zst" | \
                zstd --decompress > "$target_path"
            echo "Done: $db_name"
        fi
    done
    echo ""

    # Convert to MMseqs2 padded format
    echo "=== Converting to GPU-padded format ==="
    echo ""

    total_dbs=${#DATABASES[@]}
    current_db=0

    for db_name in "${!DATABASES[@]}"; do
        current_db=$((current_db + 1))
        fasta_file="${DATABASES[$db_name]}"
        source_fasta="${TARGET_DIR}/${fasta_file}"
        intermediate_db="${TARGET_DIR}/${db_name}"

        echo "[$current_db/$total_dbs] Converting: $db_name"

        if [ -f "${TARGET_DIR}/${db_name}_padded.dbtype" ]; then
            echo "  SKIP: Padded database already exists"
            echo ""
            continue
        fi

        if [ ! -f "$source_fasta" ]; then
            echo "  WARNING: Source FASTA not found: $source_fasta"
            echo "  Skipping."
            echo ""
            continue
        fi

        # Step 1: createdb
        if [ -f "${intermediate_db}.dbtype" ]; then
            echo "  Found intermediate database, skipping createdb..."
        else
            echo "  Creating MMseqs2 database..."
            time mmseqs createdb "$source_fasta" "$intermediate_db" --threads "$THREADS"
        fi

        # Step 2: makepaddedseqdb
        convert_to_padded "$db_name" "$intermediate_db"

        # Clean up intermediate database
        if [ "$KEEP_RAW" = false ] && [ -f "${TARGET_DIR}/${db_name}_padded.dbtype" ]; then
            echo "  Cleaning up intermediate database..."
            rm -f "${intermediate_db}" "${intermediate_db}".dbtype "${intermediate_db}".index \
                  "${intermediate_db}".lookup "${intermediate_db}".source \
                  "${intermediate_db}_h" "${intermediate_db}_h".dbtype "${intermediate_db}_h".index

            echo "  Cleaning up raw FASTA..."
            rm -f "$source_fasta"
        fi

        echo ""
    done
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo "End time: $(date)"
echo ""
echo "Database directory: $TARGET_DIR"
echo ""
echo "Available databases:"
for f in "${TARGET_DIR}"/*_padded.dbtype "${TARGET_DIR}"/*_db.dbtype; do
    if [ -f "$f" ]; then
        db=$(basename "$f" .dbtype)
        echo "  - $db"
    fi
done
echo ""
echo "Usage with Boltz:"
echo "  boltz predict input.yaml --use_mmseqs_gpu --mmseqs_db_dir $TARGET_DIR"
echo ""
echo "Or set the environment variable:"
echo "  export BOLTZ_MMSEQS_DB_DIR=$TARGET_DIR"
echo "  boltz predict input.yaml --use_mmseqs_gpu"
echo "=========================================="
