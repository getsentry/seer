from __future__ import annotations

import configparser

from devenv import constants
from devenv.lib import colima, config, limactl, proc, venv


def main(context: dict[str, str]) -> int:
    repo = context["repo"]
    reporoot = context["reporoot"]

    venv_dir, python_version, requirements, editable_paths, bins = venv.get(reporoot, repo)
    url, sha256 = config.get_python(reporoot, python_version)
    print(f"ensuring {repo} venv at {venv_dir}...")
    venv.ensure(venv_dir, python_version, url, sha256)
    venv.sync(reporoot, venv_dir, requirements)

    repo_config = configparser.ConfigParser()
    repo_config.read(f"{reporoot}/devenv/config.ini")

    if constants.DARWIN:
        colima.install(
            repo_config["colima"]["version"],
            repo_config["colima"][constants.SYSTEM_MACHINE],
            repo_config["colima"][f"{constants.SYSTEM_MACHINE}_sha256"],
            reporoot,
        )
        limactl.install(
            repo_config["lima"]["version"],
            repo_config["lima"][constants.SYSTEM_MACHINE],
            repo_config["lima"][f"{constants.SYSTEM_MACHINE}_sha256"],
            reporoot,
        )
        colima.start(reporoot)

    print("Executing update tasks in Makefile...")
    proc.run(("make", "-C", reporoot, "update"), exit=True)

    return 0
