# 🧬 Tumor Microenvironment (TME) Modeling Plan  
**Handover Document**

## 1. Objective

Build a hybrid off-lattice computational model of the tumor microenvironment that simulates:

- Cancer cells, immune cells, and healthy/stromal cells  
- Oxygen and glucose concentration fields  
- Vasculature layout  
- Cell states: proliferative, quiescent, dead  

---

## 2. Modeling Strategy

Hybrid discrete–continuum framework:

### Discrete (cells)
- Off-lattice agents
- Position, volume, type, internal state

### Continuum (fields)

Oxygen:
∂O/∂t = D_O ∇²O − Σ c_i(O) + vessel source

Glucose:
∂G/∂t = D_G ∇²G − Σ c_i(G) + vessel source

---

## 3. Framework Choice

Primary: PhysiCell + BioFVM  
Optional: Microvessel Chaste (for vasculature)

---

## 4. Architecture

Cells → Fields → Vasculature → Outputs

---

## 5. Cell State Rules

Energy proxy:
E = w_O * O + w_G * G

- High energy → proliferative  
- Medium → quiescent  
- Low → necrotic  
- Immune attack → apoptotic  

---

## 6. Vasculature

Option A: static  
Option B: angiogenesis (VEGF-driven)

---

## 7. Implementation Plan

Phase 1: tumor + oxygen  
Phase 2: add glucose  
Phase 3: immune cells  
Phase 4: vasculature  

---

## 8. Outputs

- Oxygen/glucose maps  
- Cell states  
- Tumor size  
- Necrotic fraction  

---

## 9. References

- PhysiCell (2018)  
- BioFVM (2015)  
- Microvessel Chaste (2017)  
- Nikmaneshi et al. (2020)  
- Phillips et al. (2020)  

---

## 10. Next Step

Start with PhysiCell → add glucose → add state rules → extend to vasculature
