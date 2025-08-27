# debugging-neural-networks
Numerical experiments for a better understanding of neural networks.

Cél: közelebb hozni a gyakorlati viselkedést az elméleti modellekhez - kis autoencoder példa + részletes logolás.

Telepítés:
pip install -r requirements.txt

Konfiguráció:
- Módosítsd az `experiments/config.yaml`-t igényeid szerint (input_dim, hidden_dims, lr, epochs, stb).

Futtatás:
python src/train.py

Eredmények:
A runs/ mappába kerül egy időbélyegzett alkönyvtár, abban step_000000.npz fájlok.
Ezek tartalmazzák:
- param__<name> : súlyok / biasok
- grad__<name> : gradiensek
- act__<name> : aktivációk
- loss : a mentéskori loss

Analízis:
python src/analysis.py  # szerkeszd, add meg a run_dir-t és paramneveket

