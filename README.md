# Early Warning for DeFi Exploits - dashboard

Level 1: replay of the stored test results (no key needed). Level 2: live check of any wallet with the FROZEN model.

    pip install -r requirements.txt
    cp .streamlit/secrets.toml.example .streamlit/secrets.toml     # then put your rotated Etherscan key in it
    streamlit run app.py

Level 2 modes
- Any wallet: address + reference time (now, or any past date/time in UTC). Only history before that time is used.
- Replay an exploit case: pick a case and "N hours before". With "at the exploit" the live score must match the stored study score.

Frozen files (frozen/): autoencoder.pt (sha256 starts c04fe92a526fa31a), frozen_settings.json (d218f0cd08fa265b),
autoencoder_weights.npz (the same weights as numpy, so torch is not needed).
Tests: python tests/test_engine.py  (features identical to the notebook; numpy model equals the torch model; mock pipeline).
The Etherscan key is read from secrets or the environment and is never written to any file or output.
