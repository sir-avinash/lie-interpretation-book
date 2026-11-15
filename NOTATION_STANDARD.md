# Master Notation, Terminology & Consistency Standard

1. **Equations:** `(C.N)` where `C` is chapter number, `N` is equation number (e.g., `Eq. (2.4)`).
2. **Figures & Tables:** `Fig. C.N`, `Table C.N`.
3. **Formal Statements:** `Def. C.N`, `Thm. C.N`, `Prop. C.N`, `Lemma C.N`, `Assumption C.N`, `Remark C.N`.
4. **Hat & Vee Maps:** `(·)∧ : ℝ³ → so(3)` such that `x∧ y = x × y`; `(·)∨ : so(3) → ℝ³`.
5. **Angular Velocities:** `Ω ∈ ℝ³` for `SO(3)` (`Ṙ = R Ω∧`); `ω ∈ ℝ³` for `S²` (`q̇ = ω × q`, `ω · q = 0`).
6. **Errors & Transport Maps:**
   - `SO(3)`: `Ψ_R = ½ Tr(I₃ − R_dᵀ R)`, `e_R = ½ (R_dᵀ R − Rᵀ R_d)∨`, `T(R, R_d) Ω_d = Rᵀ R_d Ω_d`, `e_Ω = Ω − Rᵀ R_d Ω_d`.
   - `S²`: `Ψ_q = 1 − q · q_d`, `e_q = (q_d∧)² q`, `T(q, q_d) q̇_d = (q_d × q̇_d) × q`, `e_q̇ = q̇ − (q_d × q̇_d) × q`, `e_ω = ω − (−q∧²) ω_d`.
7. **Infinitesimal Variations:** `δR = R_d η∧` (`η ∈ ℝ³`), `δΩ = Ω_d∧ η + η̇`; `δq = ξ × q_d` (`ξ · q_d = 0`), `(ξ × q_d) · ω_d + q_d · δω = 0`.
8. **Symmetry & Local Connection:** `ξ_body = A(r) ṙ` on principal fiber bundle `Q = G × B`.
