import os
from pathlib import Path

import tink
from absl import app, flags, logging
from tink import aead
from tink.integration import gcpkms

FLAGS = flags.FLAGS

flags.DEFINE_enum("mode", None, ["encrypt", "decrypt"], "The operation to perform.")
flags.DEFINE_string("kek_uri", None, "The Cloud KMS URI of the key encryption key.")


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

    if FLAGS.mode == "encrypt":
        # /tests/**/**/cassettes
        for cassette_dir in Path("tests").rglob("cassettes"):
            # Create corresponding _encrypted_cassettes directory if not exists
            encrypted_dir = cassette_dir.parent / "_encrypted_cassettes"
            encrypted_dir.mkdir(exist_ok=True)

            # Find all yaml files in the cassette directory
            for yaml_file in cassette_dir.glob("*.yaml"):
                print(f"Encrypting {yaml_file}")
                encrypted_file = encrypted_dir / f"{yaml_file.name}.encrypted"

                # Encrypt the files
                with open(yaml_file, "rb") as input_file:
                    # using the relative path as associated data
                    associated_data = os.path.relpath(yaml_file, base_directory).encode("utf-8")
                    encrypted_data = env_aead.encrypt(input_file.read(), associated_data)

                # Write the encrypted data to the output file
                with open(encrypted_file, "wb") as output_file:
                    output_file.write(encrypted_data)

    elif FLAGS.mode == "decrypt":
        # /tests/**/**/_encrypted_cassettes
        for encrypted_dir in Path("tests").rglob("_encrypted_cassettes"):
            # Create corresponding _decrypted_cassettes directory if not exists
            decrypted_dir = encrypted_dir.parent / "cassettes"
            decrypted_dir.mkdir(exist_ok=True)

            # Find all encrypted files in the encrypted directory
            for encrypted_file in encrypted_dir.glob("*.encrypted"):
                print(f"Decrypting {encrypted_file}")
                decrypted_file = decrypted_dir / encrypted_file.name.replace(".encrypted", "")

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
    else:
        logging.error(
            'Unsupported mode %s. Please choose "encrypt" or "decrypt".',
            FLAGS.mode,
        )
        return 1


if __name__ == "__main__":
    flags.mark_flags_as_required(["mode", "kek_uri"])
    app.run(main)
