import subprocess, sys, os, tarfile
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "pyyaml", "omegaconf", "lightgbm", "pyarrow", "scipy", "scikit-learn", "duckdb", "torch", "hmmlearn", "ripser", "giotto-ph", "persim"])
src_tar = "/opt/ml/processing/input/source/source_pkg.tar.gz"
src_dir = "/opt/ml/processing/source"
os.makedirs(src_dir, exist_ok=True)
with tarfile.open(src_tar, "r:gz") as tar:
    tar.extractall(src_dir)
import glob
# Unpack any model.tar.gz in teacher input
teacher_dir = "/opt/ml/processing/input/teacher"
if os.path.isdir(teacher_dir):
    for tgz in glob.glob(os.path.join(teacher_dir, "*.tar.gz")):
        with tarfile.open(tgz, "r:gz") as tar:
            tar.extractall(teacher_dir)
        print(f"Unpacked teacher: {os.listdir(teacher_dir)}")
sys.path.insert(0, src_dir)
os.chdir(src_dir)
sys.argv = ["adapters/santander_adapter.py"] + ['--pipeline', 'configs/santander/pipeline.yaml', '--input-dir', '/opt/ml/processing/input/raw', '--output-dir', '/opt/ml/processing/output', '--stages', '1-6']
exec(open("adapters/santander_adapter.py").read())
