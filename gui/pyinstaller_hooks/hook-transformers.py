"""Transformers 4.57 discovers lazy model exports by reading package source files."""
from PyInstaller.utils.hooks import collect_data_files, copy_metadata

datas = collect_data_files("transformers", include_py_files=True)
datas += copy_metadata("transformers", recursive=True)
