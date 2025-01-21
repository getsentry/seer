#!/bin/bash

# Exit on any error
set -e

# Source common functions
source "$(dirname "$0")/vcr-common.sh"

# Check GPG key
check_gpg_key

# Get passphrase
passphrase=$(get_passphrase)

# Create a temporary file to store counts
temp_file=$(mktemp)
echo "0" > "$temp_file"
declare -a created_dirs

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
        gpg --default-key $GPG_KEY_ID --sign-with $GPG_KEY_ID --yes --batch --passphrase "$passphrase" --decrypt --output "$decrypted_file" "$gpg_file" 2>/dev/null

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
