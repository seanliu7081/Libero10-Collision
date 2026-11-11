"""Bundle frozen data in wheels while retaining the readable source layout."""
from pathlib import Path
import shutil

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildWithBenchmarkData(build_py):
    def run(self):
        super().run()
        source = Path(__file__).resolve().parent
        target = Path(self.build_lib) / 'libero10_collision' / 'data'
        target.mkdir(parents=True, exist_ok=True)
        for name in ('benchmark.json', 'runtime_provenance.json', 'GENERATION_PROTOCOL.md',
                     'README.md', 'LICENSE', 'THIRD_PARTY_NOTICES.md',
                     'requirements.txt', 'requirements-lock-py310-linux.txt'):
            shutil.copy2(source / name, target / name)
        for name in ('tasks', 'generation', 'docs'):
            # Rebuilds must not keep stale episode or recipe data.
            if (target / name).exists():
                shutil.rmtree(target / name)
            shutil.copytree(source / name, target / name)


setup(cmdclass={'build_py': BuildWithBenchmarkData})
