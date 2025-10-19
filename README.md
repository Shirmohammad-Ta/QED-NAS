# QED-NAS: Quantum Entanglement-Driven Neural Architecture Search

## 📖 Overview
This repository implements **QED-NAS**, a novel framework that automatically discovers optimal neural network architectures for representing quantum states based on their **entanglement properties**. Unlike traditional approaches that analyze entanglement patterns emerging from a given architecture, QED-NAS starts with desired entanglement properties and searches for architectures that efficiently reproduce them.

---

## 🔄 Core Innovation
- **Traditional Approach:** Architecture → Entanglement  
- **Our Approach:** Entanglement → Architecture  

This represents a **paradigm shift** from theoretical analysis to practical automation in quantum-inspired machine learning.

---

## 🏗️ Framework Architecture

### Phase 0: Preparation & Predictor Training (Offline)
1. **Generate Training Dataset** - Diverse quantum systems (gapped, gapless, critical, topological) of manageable size (10–16 qubits).  
2. **Calculate Exact E_target** - Compute full entanglement spectrum using exact diagonalization.  
3. **Extract Proxy Features** - Derive efficient measurements (entropy, mean, variance, skewness) capturing essential entanglement properties.  
4. **Train Predictor Model** - Develop a neural network that predicts entanglement features from Hamiltonian parameters.

### Phase 1: Efficient Mapping for Target System
1. **Receive Target System** - Input Hamiltonian of large quantum system (50+ qubits).  
2. **Predict E_proxy_target** - Use trained predictor to estimate entanglement properties efficiently.

### Phase 2: Architecture Optimization with Supernet
1. **Define Search Space** - Building blocks (RBM, MPS, Attention, FC) corresponding to different entanglement patterns.  
2. **Build Differentiable Supernet** - Comprehensive network containing all possible architecture combinations.  
3. **Train Supernet** - Optimize model weights and architecture parameters simultaneously using a combined loss function (task performance + entanglement matching).  
4. **Sample Final Architecture** - Select the optimal architecture from the trained supernet.

### Phase 3: Evaluation & Interpretation
1. **Evaluate Optimal Architecture** - Test the discovered architecture on the target system.  
2. **Compare with Baselines** - Benchmark against established methods.  
3. **Extract Design Rules** - Identify patterns connecting physical system properties to optimal architectural choices.  
4. **Physical Interpretation** - Develop theoretical understanding of architecture–entanglement relationships.

---

## 🎯 Key Features
- **Automated Architecture Discovery** – No manual design needed.  
- **Entanglement-Driven** – Uses quantum entanglement as primary optimization criterion.  
- **Scalable** – Handles large quantum systems (50+ qubits) through efficient proxies.  
- **Interpretable** – Provides insights into quantum physics and ML architecture relationships.  
- **Differentiable Search** – Efficient optimization through supernet training.

---

## 📚 Theoretical Foundation
- Short-range RBMs capture **area-law entangled states**.  
- Long-range RBMs represent **volume-law entangled states**.  
- Neural network architecture determines **entanglement capabilities**.

---

## 🚀 Applications
- Quantum state representation and simulation.  
- Quantum machine learning model design.  
- Quantum-inspired classical algorithms.  
- Cross-disciplinary research bridging quantum physics and neural architecture search.

---

## 📝 Citation
Proposed Idea Based on the Paper of Quantum Entanglement in Neural Network States  
**Shirmohammad Tavangari**, Advisor: Prof. Jayakumar Rajadas  
Stanford University School of Medicine, August 20, 2025  

---

## 📧 Contact
For questions and collaborations, please open an issue or contact the maintainers.
