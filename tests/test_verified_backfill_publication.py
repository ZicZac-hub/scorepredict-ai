"""Verify clean worktree publishing against a local bare Git remote."""
import subprocess
from pathlib import Path

SCRIPT=Path(__file__).resolve().parents[1]/"scripts"/"publish_verified_backfill.sh"

def git(cwd,*args):
    return subprocess.check_output(["git","-C",str(cwd),*args],text=True).strip()

def setup(tmp_path):
    remote=tmp_path/"remote.git"
    subprocess.run(["git","init","--bare","--initial-branch=main",str(remote)],check=True)
    repo=tmp_path/"repo"
    subprocess.run(["git","clone",str(remote),str(repo)],check=True)
    git(repo,"config","user.email","test@example.invalid")
    git(repo,"config","user.name","Test")
    (repo/"result.json").write_text("original")
    git(repo,"add","result.json")
    git(repo,"commit","-m","base")
    git(repo,"push","origin","main")
    other=tmp_path/"other"
    subprocess.run(["git","clone",str(remote),str(other)],check=True)
    git(other,"config","user.email","test@example.invalid")
    git(other,"config","user.name","Test")
    return repo,other

def publish(repo):
    return subprocess.run(["bash",str(SCRIPT)],cwd=repo,capture_output=True,text=True)

def test_untracked_collision_does_not_block_publish(tmp_path):
    repo,other=setup(tmp_path)
    (repo/"result.json").write_text("new verified score")
    git(repo,"commit","-am","verified")
    (repo/"derived.json").write_text("local untracked")
    (repo/"result.json").write_text("local unstaged")
    (other/"derived.json").write_text("independently published")
    git(other,"add","derived.json")
    git(other,"commit","-m","independent publication")
    git(other,"push","origin","main")
    result=publish(repo)
    assert result.returncode==0,result.stdout+result.stderr
    assert (repo/"derived.json").read_text()=="local untracked"
    assert (repo/"result.json").read_text()=="local unstaged"
    git(other,"pull","--ff-only")
    assert (other/"result.json").read_text()=="new verified score"
    assert (other/"derived.json").read_text()=="independently published"

def test_committed_conflict_fails_closed(tmp_path):
    repo,other=setup(tmp_path)
    (repo/"result.json").write_text("local evidence")
    git(repo,"commit","-am","local")
    (other/"result.json").write_text("other evidence")
    git(other,"commit","-am","other")
    git(other,"push","origin","main")
    result=publish(repo)
    assert result.returncode!=0
    assert (repo/"result.json").read_text()=="local evidence"
    git(repo,"fetch","origin","main")
    assert git(repo,"show","origin/main:result.json")=="other evidence"
