#!/bin/bash

# Exit on any error
set -e

key_id="7360DA708B34E5473FA3133EDFE3D3229D824D99"

# Create a temporary file to store counts
temp_file=$(mktemp)
echo "0" > "$temp_file"
declare -a created_dirs

# Check if GPG key exists
if ! gpg --list-keys "$key_id" >/dev/null 2>&1; then
    echo "Error: GPG key $key_id not found, you should follow the setup instructions to setup your GPG keys."
    exit 1
fi
echo "Using GPG key: $key_id"

# Find all _encrypted_cassettes directories under tests/
find tests -type d -name "_encrypted_cassettes" | while read encrypted_dir; do
    # Get the corresponding cassettes directory
    cassette_dir="${encrypted_dir%/*}/cassettes"
    mkdir -p "$cassette_dir"
    created_dirs+=("$cassette_dir")

    # Find all gpg files in the encrypted directory
    find "$encrypted_dir" -type f -name "*.yaml.gpg" | while read gpg_file; do
        # Get the base filename without .gpg extension
        filename=$(basename "$gpg_file" .gpg)
        # Create decrypted file path
        decrypted_file="$cassette_dir/${filename}"

        echo "Decrypting $decrypted_file"
        # Decrypt the file using GPG
        gpg --default-key $key_id --sign-with $key_id --yes --batch --passphrase-file .env --decrypt --output "$decrypted_file" "$gpg_file" 2>/dev/null

        # Increment count in temp file
        count=$(cat "$temp_file")
        echo $((count + 1)) > "$temp_file"
    done
done

# Read final count
decrypted_count=$(cat "$temp_file")
rm "$temp_file"

echo "Decryption complete! Total cassettes decrypted: $decrypted_count"
echo "Created cassette folders:"
find tests -type d -name "cassettes" | while read dir; do
    echo "  $dir"
done
