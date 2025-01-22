import hashlib
import os
from pathlib import Path

import tink
from absl import app, flags, logging
from tink import aead
from tink.integration import gcpkms

FLAGS = flags.FLAGS

flags.DEFINE_enum("mode", None, ["encrypt", "decrypt"], "The operation to perform.")
flags.DEFINE_string("kek_uri", None, "The Cloud KMS URI of the key encryption key.")
flags.DEFINE_boolean(
    "clean",
    False,
    "Whether to remove files in the destination directory that don't exist in the source directory.",
)


def main(argv):
    del argv  # Unused.

    # Initialise Tink
    aead.register()
    base_directory = os.getcwd()

    try:
        # `None` here is to using the default gcloud credentials
        # For local, it will be coming from gcloud auth application-default login
        # For CI, it will be coming from 'google-github-actions/auth@v1'
        client = gcpkms.GcpKmsClient(FLAGS.kek_uri, None)
    except tink.TinkError as e:
        logging.exception("Error creating GCP KMS client: %s", e)
        return 1

    # Create envelope AEAD primitive using AES256 GCM for encrypting the data
    try:
        remote_aead = client.get_aead(FLAGS.kek_uri)
        env_aead = aead.KmsEnvelopeAead(aead.aead_key_templates.AES256_GCM, remote_aead)
    except tink.TinkError as e:
        logging.exception("Error creating primitive: %s", e)
        return 1

    print(f"Mode: {FLAGS.mode}")
    if FLAGS.clean:
        print(
            "CLEAN is true. Will clean out the destination directory if files are not present in the source directory."
        )

    if FLAGS.mode == "encrypt":
        total_files = 0
        skipped_files = 0
        # /tests/**/**/cassettes
        for cassette_dir in Path("tests").rglob("cassettes"):
            # Create corresponding _encrypted_cassettes directory if not exists
            encrypted_dir = cassette_dir.parent / "_encrypted_cassettes"
            encrypted_dir.mkdir(exist_ok=True)

            # Keep track of processed files for cleanup
            processed_files = set()

            # Find all yaml files in the cassette directory
            for yaml_file in cassette_dir.glob("*.yaml"):
                encrypted_file = encrypted_dir / f"{yaml_file.name}.encrypted"
                processed_files.add(encrypted_file.name)

                # Read the new content and calculate its hash
                with open(yaml_file, "rb") as input_file:
                    new_content = input_file.read()
                    new_hash = hashlib.sha256(new_content).digest()
                    associated_data = os.path.relpath(yaml_file, base_directory).encode("utf-8")

                # Check if we need to encrypt by comparing hashes
                needs_encryption = True
                if encrypted_file.exists():
                    try:
                        # Try to decrypt existing file and compare hashes
                        with open(encrypted_file, "rb") as existing_file:
                            existing_decrypted = env_aead.decrypt(
                                existing_file.read(), associated_data
                            )
                            existing_hash = hashlib.sha256(existing_decrypted).digest()
                            if existing_hash == new_hash:
                                print(f"Skipping {yaml_file} - content unchanged")
                                needs_encryption = False
                                skipped_files += 1
                    except tink.TinkError:
                        # If decryption fails, we'll re-encrypt
                        pass

                if needs_encryption:
                    print(f"Encrypting {yaml_file}")
                    # Encrypt the content
                    encrypted_data = env_aead.encrypt(new_content, associated_data)
                    # Write the encrypted data to the output file
                    with open(encrypted_file, "wb") as output_file:
                        output_file.write(encrypted_data)
                    total_files += 1

            # Clean up orphaned files if --clean is set
            if FLAGS.clean:
                for encrypted_file in encrypted_dir.glob("*.encrypted"):
                    if encrypted_file.name not in processed_files:
                        print(f"Removing orphaned file {encrypted_file}")
                        encrypted_file.unlink()

        print(f"\nTotal files encrypted: {total_files}")
        print(f"Total files skipped: {skipped_files}")

    elif FLAGS.mode == "decrypt":
        total_files = 0
        skipped_files = 0
        affected_folders = set()
        # /tests/**/**/_encrypted_cassettes
        for encrypted_dir in Path("tests").rglob("_encrypted_cassettes"):
            affected_folders.add(str(encrypted_dir.parent))
            # Create corresponding decrypted directory if not exists
            decrypted_dir = encrypted_dir.parent / "cassettes"
            decrypted_dir.mkdir(exist_ok=True)

            # Keep track of processed files for cleanup
            processed_files = set()

            # Find all encrypted files in the encrypted directory
            for encrypted_file in encrypted_dir.glob("*.encrypted"):
                decrypted_file = decrypted_dir / encrypted_file.name.replace(".encrypted", "")
                processed_files.add(decrypted_file.name)

                print(f"Decrypting {encrypted_file}")

                # Decrypt the files
                with open(encrypted_file, "rb") as input_file:
                    # using the relative path of the unencrypted file as associated data
                    associated_data = (
                        os.path.relpath(encrypted_file, base_directory)
                        .replace(".encrypted", "")
                        .replace("_encrypted_cassettes", "cassettes")
                        .encode("utf-8")
                    )
                    decrypted_data = env_aead.decrypt(input_file.read(), associated_data)

                # Write the decrypted data to the output file
                with open(decrypted_file, "wb") as output_file:
                    output_file.write(decrypted_data)
                total_files += 1

            # Clean up orphaned files if --clean is set
            if FLAGS.clean:
                for decrypted_file in decrypted_dir.glob("*.yaml"):
                    if decrypted_file.name not in processed_files:
                        print(f"Removing orphaned file {decrypted_file}")
                        decrypted_file.unlink()

        print(f"\nTotal files decrypted: {total_files}")
        print("\nAffected folders:")
        for folder in sorted(affected_folders):
            print(f"  - {folder}")

    else:
        logging.error(
            'Unsupported mode %s. Please choose "encrypt" or "decrypt".',
            FLAGS.mode,
        )
        return 1


if __name__ == "__main__":
    flags.mark_flags_as_required(["mode", "kek_uri"])
    app.run(main)
