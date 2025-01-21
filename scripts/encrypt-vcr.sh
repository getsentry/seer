#!/bin/bash

# Exit on any error
set -e

# Source common functions
source "$(dirname "$0")/vcr-common.sh"

# Check GPG key
check_gpg_key

# Get passphrase
passphrase=$(get_passphrase)

# Initialize a counter for the number of encrypted files
encrypted_count=0

# Find all cassette directories under tests/
while IFS= read -r -d '' cassette_dir; do
    # Create corresponding _encrypted_cassettes directory
    encrypted_dir="${cassette_dir%/*}/_encrypted_cassettes"
    mkdir -p "$encrypted_dir"

    # Find all yaml files in the cassette directory
    while IFS= read -r -d '' yaml_file; do
        # Get the base filename
        filename=$(basename "$yaml_file")
        # Create encrypted file path
        encrypted_file="$encrypted_dir/${filename}.gpg"

        echo "Encrypting $yaml_file"
        # Encrypt the file using GPG with symmetric encryption
        gpg --default-key $GPG_KEY_ID --sign-with $GPG_KEY_ID --yes --batch --passphrase "$passphrase" --symmetric --output "$encrypted_file" "$yaml_file"

        # Increment the counter
        ((encrypted_count++))
    done < <(find "$cassette_dir" -type f -name "*.yaml" -print0)
done < <(find tests -type d -name "cassettes" -print0)

echo "Encryption complete! Total cassettes encrypted: $encrypted_count"
