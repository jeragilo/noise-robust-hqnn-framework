#!/usr/bin/env python
"""
Demo 12 — HQNN Explainability & Trustworthiness Analysis

This demo:
- Trains an HQNN on a 4-feature synthetic dataset.
- Computes:
    1. Parameter importance (sensitivity of predictions to variational params)
    2. Feature importance (sensitivity to input perturbations)
    3. Stability curve (prediction variance under parameter noise)
    4. Classical baseline explainability (Logistic Regression coefficients)
- Saves:
    - JSON summary
    - Heatmaps + sensitivity plots

Outputs:
- results/demo12/results_demo12_explainability.json
- results/demo12/explainability_heatmap.png
- results/demo12/feature_sensitivity_demo12.png
- results/demo12/parameter_sensitivity_demo12.png
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


# ============================================================
# HQNN Components (same architecture from Demo 01/08)
# ============================================================

def build_feature_map(num_qubits, x):
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.ry(float(x[i]), i)
    return qc

def build_var_layer(num_qubits, w):
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.rx(w[i], i)
        qc.rz(w[num_qubits+i], i)
    for i in range(num_qubits):
        qc.cz(i, (i+1)%num_qubits)
    return qc

def build_hqnn(num_qubits, x, w):
    fm = build_feature_map(num_qubits, x)
    var = build_var_layer(num_qubits, w)
    qc = fm.compose(var)
    qc.measure_all()
    return qc

def parity_expval(counts):
    shots=sum(counts.values())
    exp=0
    for bits,c in counts.items():
        parity=bits.count("1")%2
        sign=1 if parity==0 else -1
        exp+=sign*c/shots
    return exp

def predict_prob(sim, num_qubits, w, x):
    x_pad=np.zeros(num_qubits)
    x_pad[:len(x)] = x
    qc=build_hqnn(num_qubits,x_pad,w)
    result=sim.run(qc,shots=1024).result()
    counts=result.get_counts()
    exp=parity_expval(counts)
    return (1-exp)/2.0


# ============================================================
# SPSA (simple training)
# ============================================================

def loss_fn(sim,num_qubits,w,X,y):
    eps=1e-10
    preds=np.array([predict_prob(sim,num_qubits,w,x) for x in X])
    return float(-np.mean(y*np.log(preds+eps)+(1-y)*np.log(1-preds+eps)))

def spsa_step(sim,num_qubits,w,X,y,alpha=0.15,c=0.15):
    dim=len(w)
    delta=2*np.random.randint(0,2,dim)-1
    wplus=w+c*delta
    wminus=w-c*delta
    loss_p=loss_fn(sim,num_qubits,wplus,X,y)
    loss_m=loss_fn(sim,num_qubits,wminus,X,y)
    ghat=(loss_p-loss_m)/(2*c*delta)
    return w-alpha*ghat


# ============================================================
# Explainability: Parameter Importance
# ============================================================

def parameter_importance(sim, num_qubits, w, X_sample):
    """
    Computes importance for each parameter:
    importance[i] = average |prediction_change| over dataset
                    when perturbing parameter w_i by epsilon.
    """
    eps=0.1
    base_preds=np.array([predict_prob(sim,num_qubits,w,x) for x in X_sample])

    importances=[]
    for i in range(len(w)):
        w_perturbed=w.copy()
        w_perturbed[i]+=eps
        new_preds=np.array([predict_prob(sim,num_qubits,w_perturbed,x) for x in X_sample])
        delta=np.abs(new_preds-base_preds).mean()
        importances.append(float(delta))

    return importances


# ============================================================
# Explainability: Feature Importance
# ============================================================

def feature_importance(sim,num_qubits,w,X_sample):
    """
    Perturb each input feature by small noise and measure
    resulting prediction change.
    """
    eps=0.1
    base_preds=np.array([predict_prob(sim,num_qubits,w,x) for x in X_sample])
    importances=[]

    for j in range(X_sample.shape[1]):
        Xp=X_sample.copy()
        Xp[:,j]+=eps
        preds_pert=np.array([predict_prob(sim,num_qubits,w,x) for x in Xp])
        delta=np.abs(preds_pert-base_preds).mean()
        importances.append(float(delta))

    return importances


# ============================================================
# Stability Curve (Parameter Noise Injection)
# ============================================================

def stability_curve(sim,num_qubits,w,x_sample,noise_levels=[0,0.05,0.1,0.2]):
    """
    For one sample, evaluate prediction stability under parameter perturbation.
    """
    stabilities=[]
    for nl in noise_levels:
        preds=[]
        for _ in range(25):  # 25 random runs
            w_pert=w+nl*np.random.randn(len(w))
            preds.append(predict_prob(sim,num_qubits,w_pert,x_sample))
        stabilities.append(float(np.std(preds)))
    return noise_levels, stabilities


# ============================================================
# MAIN DEMO
# ============================================================

def run_demo(output_dir, epochs=10):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Dataset
    X,y=make_classification(
        n_samples=200,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        class_sep=1.5,
        random_state=42
    )

    X_train,X_test,y_train,y_test=train_test_split(
        X,y,test_size=0.3,random_state=42
    )
    scaler=StandardScaler()
    X_train=scaler.fit_transform(X_train)
    X_test=scaler.transform(X_test)

    sim=AerSimulator()
    num_qubits=4
    w=np.random.uniform(-np.pi,np.pi,2*num_qubits)

    # 2. Train HQNN
    for ep in range(epochs):
        w=spsa_step(sim,num_qubits,w,X_train,y_train)
        print(f"[HQNN][Epoch {ep}] Loss={loss_fn(sim,num_qubits,w,X_train,y_train):.4f}")

    # 3. Explainability calculations
    X_sample=X_test[:20]     # small subset for speed
    param_imp=parameter_importance(sim,num_qubits,w,X_sample)
    feat_imp=feature_importance(sim,num_qubits,w,X_sample)

    # Stability curve
    noise_levels, stabilities = stability_curve(sim,num_qubits,w,X_test[0])

    # Classical baseline
    clf=LogisticRegression(max_iter=2000)
    clf.fit(X_train,y_train)
    classical_coefs=clf.coef_[0].tolist()

    # 4. Save JSON
    summary={
        "parameter_importance": param_imp,
        "feature_importance": feat_imp,
        "stability_curve": {
            "noise_levels": noise_levels,
            "std_pred": stabilities
        },
        "classical_coefficients": classical_coefs
    }

    json_path=os.path.join(output_dir,"results_demo12_explainability.json")
    with open(json_path,"w") as f: json.dump(summary,f,indent=2)

    # 5. Plots
    # Parameter importance
    plt.figure(figsize=(7,5))
    plt.bar(range(len(param_imp)),param_imp,color="purple")
    plt.title("HQNN Parameter Importance")
    plt.xlabel("Parameter Index")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,"parameter_sensitivity_demo12.png"))
    plt.close()

    # Feature importance
    plt.figure(figsize=(7,5))
    plt.bar(range(len(feat_imp)),feat_imp,color="green")
    plt.title("HQNN Feature Importance")
    plt.xlabel("Feature Index")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,"feature_sensitivity_demo12.png"))
    plt.close()

    # Stability curve
    plt.figure(figsize=(7,5))
    plt.plot(noise_levels,stabilities,marker="o")
    plt.title("HQNN Stability Under Parameter Noise")
    plt.xlabel("Noise Level")
    plt.ylabel("Prediction StdDev")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,"stability_demo12.png"))
    plt.close()

    print("\n===== DEMO 12 SUMMARY =====")
    print(summary)
    print(f"\nSaved JSON: {json_path}")
    print("Saved plots in:", output_dir)


# CLI
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output_dir",default="results/demo12")
    parser.add_argument("--epochs",type=int,default=10)
    args=parser.parse_args()

    run_demo(args.output_dir,epochs=args.epochs)

