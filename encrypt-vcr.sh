#!/bin/bash

# Exit on any error
set -e

key_id="7360DA708B34E5473FA3133EDFE3D3229D824D99"

# Check if GPG key exists
if ! gpg --list-keys "$key_id" >/dev/null 2>&1; then
    echo "Error: GPG key $key_id not found, you should follow the setup instructions to setup your GPG keys."
    exit 1
fi
echo "Using GPG key: $key_id"

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
        gpg --default-key $key_id --sign-with $key_id --yes --batch --passphrase-file .env --symmetric --output "$encrypted_file" "$yaml_file"

        # Increment the counter
        ((encrypted_count++))
    done < <(find "$cassette_dir" -type f -name "*.yaml" -print0)
done < <(find tests -type d -name "cassettes" -print0)

echo "Encryption complete! Total cassettes encrypted: $encrypted_count"
