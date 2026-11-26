In-Progress

## Requirements
- Python 3.9 or newer (tested on 3.11)
- pip

## Run locally

1. Clone or download the repo, then open a terminal inside the folder:
   ```
   git clone https://github.com/OOrangEmon/water.marker.git
   cd water.marker
   ```

2. Create and activate a virtual environment:
   - Linux / Mac:
     ```
     python3 -m venv .venv
     source .venv/bin/activate
     ```
   - Windows:
     ```
     python -m venv .venv
     .venv\Scripts\activate
     ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   - Windows: if `gunicorn` fails to install, delete that line from `requirements.txt`. It is only needed for hosting.

4. Start the app:
   ```
   python app.py
   ```

5. Open `http://127.0.0.1:5000` in your browser. Use a desktop for the best layout.

Stop the server with `Ctrl + C`.

## How to use
1. Upload an image.
2. Enter text (max 6 characters) or upload a logo. If both are given, the logo is used.
3. Set the strength (delta). Higher is more robust but slightly more visible.
4. Pick an attack and a level, then press Run.
5. Read the results:
   - PSNR / SSIM: how invisible the watermark is
   - NC / BER: how well the watermark was recovered (NC closer to 1 and BER closer to 0 is better)
   - Robustness table: results for 14 preset attacks
