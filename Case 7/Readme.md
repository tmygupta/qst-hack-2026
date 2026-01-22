#### Contacts for Case 7
- xxx
- yyy

### Getting started

1. Install [uv](https://docs.astral.sh/uv/)
2. Create an environment in a new folder:  
   ```
   uv init <projectname>
   cd <projectname>
   ```
3. Add [MrMustard (development version)](https://github.com/XanaduAI/MrMustard) to the environment:
   ```
   uv add "mrmustard @ git+https://github.com/XanaduAI/MrMustard.git"
   ```
4. Add plotly and jupyter:
   ```
   uv add jupyter plotly
   ```
5. Start Jupyter Lab:
   ```
   uv run jupyter lab
   ```