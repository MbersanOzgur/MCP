# Backend

This backend is built with **FastAPI** and **Python**.

## How to Run from Scratch

```bash

# 1. Create a virtual environment
python -m venv .venv

# 2. Activate the virtual environment
# On Linux/Mac:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the server
uvicorn app.mcp_server:app --reload

