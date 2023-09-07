# -*- coding: utf-8 -*-
"""
Created on Tue May 23 22:05:39 2023

@author: tmade
"""

import yfinance as yf

name="^FTSE"

df_brent=yf.download(name,end="2023-09-05")
df_brent.to_csv(f"./{name}.csv")


name_l=["MNG.L",
        "RTO.L",
        "TSCO.L",
        "CCH.L",
        "SSE.L",
        "VOD.L",
        "ANTO.L",
        "RMV.L",
        "SPX.L",
        "HLN.L",
        "RS1.L",
        "AUTO.L",
        "CNA.L",
        "SMT.L",
        "FRAS.L",
        "BATS.L",
        "AHT.L",
        "SHEL.L",
        "SMIN.L",
        "EXPN.L",
        "BA.L",
        "RR.L",
        "AAF.L",
        "CPG.L",
        "SDR.L",
        "ENT.L",
        "STJ.L",
        "PSN.L",
        "PRU.L",
        "ABDN.L"]

for name in name_l:
    df_brent=yf.download(name,end="2023-09-05")
    df_brent.to_csv(f"FTSE_data/{name}.csv")