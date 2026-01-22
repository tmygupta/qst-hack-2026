
### Getting started

1. Install the [uv](https://docs.astral.sh/uv/) package and project manager.
2. Fork the _qst-hack-2026_ repository and clone it to your local machine.
3. In the terminal, `cd` to the `Case 7` folder and run `uv sync` to set up a virtual environment according to the prescriptions in the `pyproject.toml` file.
4. If you want to work with notebooks: Run `uv run python -m ipykernel install --user --name qst-hack-2026-case7` to install that environment as a Jupyter kernel that can be selected in VS Code and Jupyter Lab.



Alternative setup, if you want to start from a clean folder:

2. Create an environment in a new folder:  
   ```
   uv init <projectname>
   cd <projectname>
   ```
3. Add [MrMustard (development version)](https://github.com/XanaduAI/MrMustard) to the environment:
   ```
   uv add "mrmustard @ git+https://github.com/XanaduAI/MrMustard.git"
4. Add plotly and jupyter:
   ```
   uv add plotly jupyter
   ```
5. Start Jupyter Lab from within the `projectname` environment,
   ```
   uv run jupyter lab
   ```
   or open the folder in VS Code.  
   You probably want to install the kernel as in point 4 above.

Start hacking!


### Contacts for Case 7
- Luis González Hofmann, lcgho@dtu.dk
- Timour Javar Magnier, tkjma@dtu.dk
- Jonas Neergaard-Nielsen, jsne@fysik.dtu.dk
