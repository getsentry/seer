import os
import shutil
import tarfile

import httpx
import requests
from github import Auth, Github


class GithubClient:
    def __init__(self, auth):
        self._client = Github(auth=auth)

    def load_repo(self, repo_name, output_dir, ref="main"):
        # Check if output directory exists, if not create it
        os.makedirs(output_dir, exist_ok=True)
        for root, dirs, files in os.walk(output_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))

        repo = self._client.get_repo(repo_name)
        tarball_url = repo.get_archive_link("tarball", ref=ref)

        response = requests.get(tarball_url, stream=True)
        if response.status_code == 200:
            with open(f"{output_dir}/repo.tar.gz", "wb") as f:
                f.write(response.content)
        else:
            print(
                f"Failed to get tarball url for {tarball_url}. Please check if the repository exists and the provided token is valid."
            )
            print(f"Response status code: {response.status_code}, response text: {response.text}")
            raise Exception(
                f"Failed to get tarball url for {tarball_url}. Please check if the repository exists and the provided token is valid."
            )

        # Extract tarball into the output directory
        with tarfile.open(f"{output_dir}/repo.tar.gz", "r:gz") as tar:
            tar.extractall(path=output_dir)  # extract all members normally
            extracted_folders = [
                name
                for name in os.listdir(output_dir)
                if os.path.isdir(os.path.join(output_dir, name))
            ]
            if extracted_folders:
                root_folder = extracted_folders[0]  # assuming the first folder is the root folder
                root_folder_path = os.path.join(output_dir, root_folder)
                for item in os.listdir(root_folder_path):
                    s = os.path.join(root_folder_path, item)
                    d = os.path.join(output_dir, item)
                    if os.path.isdir(s):
                        shutil.move(
                            s, d
                        )  # move all directories from the root folder to the output directory
                    else:
                        shutil.copy2(
                            s, d
                        )  # copy all files from the root folder to the output directory
                shutil.rmtree(root_folder_path)  # remove the root folder

        # Delete the tar file
        try:
            os.remove(f"{output_dir}/repo.tar.gz")
        except OSError as e:
            print(f"Failed to delete tar file: {e}")

        return output_dir

    def create_pull_request(self, repo_name, title, body, base, head):
        repo = self._client.get_repo(repo_name)
        pull_request = repo.create_pull(title=title, body=body, base=base, head=head)
        return pull_request

    def create_branch(self, repo_name, branch_name, base_branch):
        repo = self._client.get_repo(repo_name)

        base_branch_sha = repo.get_branch(base_branch).commit.sha

        repo.create_git_ref(ref=f"refs/heads/{branch_name}", sha=base_branch_sha)
        return branch_name

    def commit_file_change(self, repo_name, path, message, content, branch):
        repo = self._client.get_repo(repo_name)
        contents = repo.get_contents(path, ref=branch)
        repo.update_file(contents.path, message, content, contents.sha, branch=branch)
        return contents.sha

    def get_file_contents(self, repo_name, path, branch):
        print(f"Getting file contents for {path} in {repo_name} on branch {branch}")
        repo = self._client.get_repo(repo_name)
        contents = repo.get_contents(path, ref=branch)
        return contents.decoded_content.decode()
