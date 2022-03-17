import matplotlib.pyplot as plt 
import pandas as pd

df = pd.read_csv("a.csv")

plt.scatter(df["chi"], df["tree"], label="TTN")
plt.plot(df["chi"], df["tree"], label="_nolegend_")
plt.scatter(df["chi"], df["mps"], label ="MPS")
plt.plot(df["chi"], df["mps"], label="_nolegend_")
plt.xlabel(r"$\chi$")
plt.ylabel(r"$ | \langle \psi | \tilde{\psi} \rangle | $")
plt.title(r"$N = 12$")
plt.legend()
plt.savefig("N_12_comp_ttn_mps.png")
plt.show()