from contextlib import contextmanager


@contextmanager
def clone_repo(repo_url):
    """
    A function that clones a repository from a remote source into a specified folder.
    This function returns the path to the cloned repository and the hash of the commit
    that was cloned.
    """
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdirname:
        print(f"Cloning {repo_url} into {tmpdirname}")
        subprocess.run(["git", "clone", "-q", "--single-branch", "--depth", "1", repo_url, tmpdirname])
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmpdirname).decode("utf-8").strip()
        print(f"Done cloning repo {repo_url} with commit hash {commit_hash}")
        yield tmpdirname, commit_hash
