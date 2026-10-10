import streamlit as st
import matplotlib.pyplot as plt
import io
import torch
from core.model import MenuUpsellModel
from core.federated_engine import run_federated_orchestration

st.set_page_config(page_title="Federated POS Network", layout="wide", page_icon="🛡️")
st.title("🛡️ Privacy-Preserving POS Recommendation Coalition")
st.markdown("Collaborative cross-domain recommendation training using **Flower** and **Opacus Differential Privacy**.")

st.sidebar.header("Coalition Controls")
uploaded_files = st.sidebar.file_uploader("Upload Clover POS CSVs", type=["csv"], accept_multiple_files=True)
rounds = st.sidebar.slider("Federated Communication Rounds", min_value=1, max_value=20, value=10)

if not uploaded_files:
    st.info("👋 Upload at least two merchant CSV files on the left sidebar to initialize the coalition network.")
else:
    st.success(f"Network Initialized: {len(uploaded_files)} independent storefronts connected.")
    
    if st.button("🚀 Launch Federated Training Simulation"):
        with st.spinner(f"Running {rounds} rounds of distributed private training..."):
            results = run_federated_orchestration(uploaded_files, num_rounds=rounds)

        st.success("✅ Federated Training Coalition Completed!")
        rounds_x = list(range(1, len(results["acc"]) + 1))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=120)

        ax1.plot(rounds_x, [a * 100 for a in results["acc"]], marker='s', color='#1b9e77', linewidth=2.5)
        ax1.set_title("Cross-Domain Recommendation Accuracy")
        ax1.set_xlabel("Communication Round")
        ax1.set_ylabel("Global Accuracy (%)")
        ax1.grid(True, linestyle=':', alpha=0.6)

        ax2.plot(rounds_x, results["epsilon"], marker='^', color='#7570b3', linewidth=2.5)
        ax2.axhline(y=3.0, color='red', linestyle='--', alpha=0.7)
        ax2.set_title("Opacus Differential Privacy Expenditure")
        ax2.set_xlabel("Communication Round")
        ax2.set_ylabel("Privacy Budget")
        ax2.grid(True, linestyle=':', alpha=0.6)

        st.pyplot(fig)

        buffer = io.BytesIO()
        torch.save(MenuUpsellModel().state_dict(), buffer)
        buffer.seek(0)
        st.download_button(
            label="💾 Download Secured Global Checkpoint",
            data=buffer,
            file_name="clover_federated_dp_model.pt",
            mime="application/octet-stream"
        )