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
        subprocess.run(["git", "clone", "--single-branch", "--depth", "1", repo_url, tmpdirname])
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmpdirname).decode("utf-8").strip()
        yield tmpdirname, commit_hash
