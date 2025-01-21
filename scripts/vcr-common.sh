#!/bin/bash

# GPG key ID used for encryption/decryption
GPG_KEY_ID="7360DA708B34E5473FA3133EDFE3D3229D824D99"

# Function to check if GPG key exists
check_gpg_key() {
    if ! gpg --list-keys "$GPG_KEY_ID" >/dev/null 2>&1; then
        echo "Error: GPG key $GPG_KEY_ID not found, you should follow the setup instructions to setup your GPG keys."
        exit 1
    fi
    echo "Using GPG key: $GPG_KEY_ID"
}

# Function to get passphrase from 1Password or environment variable
get_passphrase() {
    if [ "$CI" = "1" ]; then
        if [ -z "$GPG_PASSPHRASE" ]; then
            echo "Error: GPG_PASSPHRASE environment variable not set in CI"
            exit 1
        fi
        echo "$GPG_PASSPHRASE"
    else
        # Check if op (1Password CLI) is installed
        if ! command -v op &> /dev/null; then
            echo "Error: 1Password CLI (op) is not installed. Please install it first."
            exit 1
        fi

        echo "Reading passphrase from 1Password..."
        passphrase=$(op read "op://AI ML Team/GPG VCR Passphrase/add more/password")

        if [ -z "$passphrase" ]; then
            echo "Error: Failed to read passphrase from 1Password"
            exit 1
        fi

        echo "$passphrase"
    fi
}
