from __future__ import annotations

import configparser

from devenv.lib import colima, limactl, config, proc, venv
from devenv import constants


def main(context: dict[str, str]) -> int:
    repo = context["repo"]
    reporoot = context["reporoot"]

    venv_dir, python_version, requirements, editable_paths, bins = venv.get(reporoot, repo)
    url, sha256 = config.get_python(reporoot, python_version)
    print(f"ensuring {repo} venv at {venv_dir}...")
    venv.ensure(venv_dir, python_version, url, sha256)
    venv.sync(reporoot, venv_dir, requirements)

    # install colima
    repo_config = configparser.ConfigParser()
    repo_config.read(f"{reporoot}/devenv/config.ini")
    colima.install(
        repo_config["colima"]["version"],
        repo_config["colima"][constants.SYSTEM_MACHINE],
        repo_config["colima"][f"{constants.SYSTEM_MACHINE}_sha256"],
        reporoot,
    )
    limactl.install(reporoot)

    # start colima if it's not already running
    colima.start(reporoot)

    print("Executing update tasks in Makefile...")
    proc.run(("make", "-C", reporoot, "update"), exit=True)

    return 0
