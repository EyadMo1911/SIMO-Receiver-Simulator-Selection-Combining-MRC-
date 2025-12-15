import numpy as np
from tkinter import *
from tkinter import messagebox,ttk
import matplotlib.pyplot as plt

# -----------------------------
# BPSK Modulation
# -----------------------------
def bpsk_mod(bits):
    return 2 * bits - 1

# -----------------------------
# AWGN Channel
# -----------------------------
def add_awgn(signal, snr_db):
    snr_linear = 10 ** (snr_db / 10)
    noise_var = 1 / (2 * snr_linear)
    noise = np.sqrt(noise_var) * (np.random.randn(*signal.shape) +
                                  1j * np.random.randn(*signal.shape))
    return signal + noise

# -----------------------------
# BER Calculation
# -----------------------------
def calculate_ber(original, detected):
    return np.mean(original != detected)

# -----------------------------
# Main Simulation
# -----------------------------
def run_simulation():
    try:
        N = int(entry_N.get())
        snr_db = float(entry_SNR.get())

        if N < 2 or N > 8:
            raise ValueError
    except:
        messagebox.showerror("Invalid Input", "Enter valid N (2–8) and SNR (dB)")
        return

    fading_type = fading_var.get()
    num_bits = 100000

    # Generate bits and BPSK symbols
    bits = np.random.randint(0, 2, num_bits)
    x = bpsk_mod(bits)

    # Fading coefficients
    if fading_type == "Rayleigh":
        h = (1/np.sqrt(2)) * (np.random.randn(N) + 1j*np.random.randn(N))
    elif fading_type == "Rician":
        K = 3  # Rician factor
        h = np.sqrt(K/(K+1)) + np.sqrt(1/(K+1)) * (np.random.randn(N) + 1j*np.random.randn(N))
    else:  # No Fading
        h = np.ones(N)

    # Received signals
    y = np.zeros((N, num_bits), dtype=complex)
    for i in range(N):
        y[i, :] = add_awgn(h[i] * x, snr_db)

    # ---------------- SC ----------------
    snr_per_branch = np.abs(h)**2
    best = np.argmax(snr_per_branch)

    y_sc = y[best, :]
    x_hat_sc = np.real(y_sc / h[best]) > 0

    ber_sc = calculate_ber(bits, x_hat_sc)
    snr_sc = snr_per_branch[best]

    # ---------------- MRC ----------------
    y_mrc = np.sum(np.conj(h[:, None]) * y, axis=0)
    h_mrc = np.sum(np.abs(h)**2)

    x_hat_mrc = np.real(y_mrc / h_mrc) > 0

    ber_mrc = calculate_ber(bits, x_hat_mrc)
    snr_mrc = h_mrc

    # Display Results
    label_sc_snr.config(text=f"SC Combined SNR: {snr_sc:.3f}")
    label_mrc_snr.config(text=f"MRC Combined SNR: {snr_mrc:.3f}")
    label_sc_ber.config(text=f"SC BER: {ber_sc:.6f}")
    label_mrc_ber.config(text=f"MRC BER: {ber_mrc:.6f}")

# -----------------------------
# Plot SNR vs N
# -----------------------------
def plot_graph():
    snr_db = float(entry_SNR.get())
    Ns = range(2, 9)

    sc_avg = []
    mrc_avg = []

    trials = 500  

    fading_type = fading_var.get()

    for N in Ns:
        sc_sum = 0
        mrc_sum = 0

        for _ in range(trials):
            if fading_type == "Rayleigh":
                h = (1/np.sqrt(2)) * (np.random.randn(N) + 1j*np.random.randn(N))
            elif fading_type == "Rician":
                K = 3
                h = np.sqrt(K/(K+1)) + np.sqrt(1/(K+1)) * (np.random.randn(N) + 1j*np.random.randn(N))
            else:
                h = np.ones(N)

            sc_sum += np.max(np.abs(h)**2)
            mrc_sum += np.sum(np.abs(h)**2)

        sc_avg.append(sc_sum / trials)
        mrc_avg.append(mrc_sum / trials)

    plt.figure()
    plt.plot(Ns, sc_avg, marker='o', label='SC SNR')
    plt.plot(Ns, mrc_avg, marker='s', label='MRC SNR')
    plt.xlabel("Number of Receive Antennas (N)")
    plt.ylabel("Average Combined SNR")
    plt.title("SNR Improvement vs Number of Antennas")
    plt.grid(True)
    plt.legend()
    plt.show()

# -----------------------------
# GUI
# -----------------------------
app = Tk()
app.title("SIMO Receiver Simulator")
app.geometry("480x540")
app.resizable(True, True)
app.configure(bg="#0f172a")

style = ttk.Style()
style.theme_use("clam")

style.configure("TLabel", background="#1e293b", foreground="#e5e7eb", font=("Segoe UI", 11))
style.configure("Title.TLabel", background="#0f172a", foreground="#38bdf8", font=("Segoe UI", 18, "bold"))
style.configure("TButton", font=("Segoe UI", 11, "bold"), padding=8, background="#38bdf8", foreground="black")
style.map("TButton", background=[("active", "#0ea5e9")])

ttk.Label(app, text="SIMO Receiver Simulator", style="Title.TLabel").pack(pady=15)

card = Frame(app, bg="#1e293b")
card.pack(padx=20, pady=10, fill="both", expand=True)

def field(label):
    ttk.Label(card, text=label).pack(anchor="w", padx=20, pady=(12, 3))
    e = ttk.Entry(card, font=("Segoe UI", 11))
    e.pack(fill="x", padx=20)
    return e

entry_N = field("Number of Antennas (2–8)")
entry_SNR = field("Average SNR per Antenna (dB)")

ttk.Label(card, text="Fading Type").pack(anchor="w", padx=20, pady=(12, 3))
fading_var = StringVar(value="Rayleigh")
ttk.Combobox(card, textvariable=fading_var,
             values=["Rayleigh", "Rician", "No Fading"],
             state="readonly").pack(fill="x", padx=20)

btn_frame = Frame(card, bg="#1e293b")
btn_frame.pack(pady=15)

ttk.Button(btn_frame, text="▶ Run Simulation", command=run_simulation, width=22).pack(pady=5)
ttk.Button(btn_frame, text="📊 Plot SNR Improvement", command=plot_graph, width=22).pack()

result_frame = Frame(card, bg="#020617")
result_frame.pack(fill="x", padx=15, pady=10)

def result_label(text):
    l = Label(result_frame, text=text, font=("Segoe UI", 11),
              bg="#020617", fg="#e5e7eb")
    l.pack(pady=4)
    return l

label_sc_snr = result_label("SC Combined SNR : -")
label_mrc_snr = result_label("MRC Combined SNR : -")
label_sc_ber = result_label("SC BER : -")
label_mrc_ber = result_label("MRC BER : -")

app.mainloop()