# How to Install `btf-extractor` (Python 3.10+ Environments)

## The Issue
When trying to install `btf-extractor` via a simple `pip install btf-extractor` in modern environments (such as Python 3.10+), the installation frequently fails due to a hardcoded build constraint. 

In the source code's `pyproject.toml`, the author explicitly limited the Numpy version to `numpy<1.20` for older backward-compatibility reasons during compilation.
However, because Numpy 1.19's C-extensions rely on internal Python C-API macros (specifically `_Py_HashDouble`) which changed signatures in Python 3.10, attempting to compile Numpy 1.19 on Python 3.10 throws breaking C-compiler errors:

```text
error: too few arguments to function ‘_Py_HashDouble’
```
Because the build environment cannot supply Numpy < 1.20, `btf-extractor` fails fully.

## The Solution Pipeline
To quickly bypass this issue without having to downgrade your conda environment to Python 3.9, you simply need to download the package source code, remove the Numpy version limit, and install it locally so that it compiles against your modern Numpy API.

Run the following pipeline in your terminal:

```bash
# 1. Download the source package from PyPI to a temporary folder
curl -L 'https://files.pythonhosted.org/packages/source/b/btf-extractor/btf_extractor-1.7.0.tar.gz' -o /tmp/btf_extractor-1.7.0.tar.gz

# 2. Extract the archive
tar -xzf /tmp/btf_extractor-1.7.0.tar.gz -C /tmp
cd /tmp/btf_extractor-1.7.0

# 3. Modify the pyproject.toml to remove the strict `numpy<1.20` build requirement
# (This uses sed to directly modify the file and remove the constraint)
sed -i 's/"numpy<1.20"/"numpy"/g' pyproject.toml

# 4. Build and install into your active currently configured Python (conda) environment
pip install .

# 5. (Clean up)
rm -rf /tmp/btf_extractor-1.7.0 /tmp/btf_extractor-1.7.0.tar.gz
```

After executing these steps, `btf-extractor` will be successfully compiled against your environment's modern C++ tools and current Numpy headers. You can then resume executing models or loaders that import the package.
