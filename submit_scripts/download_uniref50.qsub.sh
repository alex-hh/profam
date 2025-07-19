#!/bin/bash
#$ -P cath
#$ -l tmem=8G
#$ -l h_vmem=8G
#$ -l h_rt=72:55:30
#$ -S /bin/bash
#$ -N downloadUR19
#$ -o /SAN/orengolab/cath_plm/ProFam/qsub_logs/
#$ -wd /SAN/orengolab/cath_plm/ProFam/profam
#$ -j y

uniref50_2019_11="https://ftp.uniprot.org/pub/databases/uniprot/previous_releases/release-2019_11/uniref/uniref2019_11.tar.gz"
save_location="/SAN/orengolab/cath_plm/ProFam/data/uniref"

# Make sure the save directory exists
mkdir -p "$save_location"

# Determine output file path
outfile="$save_location/$(basename "$uniref50_2019_11")"

# Download the file (supports resume)
echo "Downloading $uniref50_2019_11 to $outfile"
wget --continue --output-document="$outfile" "$uniref50_2019_11"

echo "Download finished: $outfile"

# Decompress the archive into the save location
echo "Decompressing $outfile into $save_location"

tar -xvzf "$outfile" -C "$save_location"

echo "Decompression finished."