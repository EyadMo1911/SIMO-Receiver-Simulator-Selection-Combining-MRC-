PROJECT 6 – SIMO Receiver Simulator (SC & MRC)

This project simulates a Single-Input Multiple-Output (SIMO)
wireless communication system using Selection Combining (SC)
and Maximum Ratio Combining (MRC).

Requirements:
- Python 3.12
- NumPy
- Matplotlib
- Tkinter

How to Run:
1. Install required libraries:
   pip install numpy matplotlib

2. Run the simulation:
   python simo_sc_mrc.py

Inputs:
- Number of receive antennas (2–8)
- Average SNR per antenna (dB)
- Fading distribution      

Outputs:
- Combined SNR for SC and MRC
- Bit Error Rate (BER) for SC and MRC
- SNR improvement vs number of antennas plot

Channel Model:
- Rayleigh block fading
- AWGN noise
- BPSK modulation
