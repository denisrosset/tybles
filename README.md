# Tybles: simple schemas for Pandas dataframes

See the website https://denisrosset.github.io/tybles 


## How to compile the documentation

```bash
poetry install -E docs -E beartype -E typeguard # important, install the documentation extras
poetry run make -C docs clean html
```
