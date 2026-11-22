// Keenan Crane-inspired SVG illustrations (beautiful-diagrams skill):
// 1. 3D-projected geometry with thick silhouettes (#1e242b), thin contours, dashed hidden lines
// 2. Muted pastel matte radial/linear shading + 20% opacity cast shadows
// 3. #1b1f8a (indigo), #c4512d (terracotta), #2a6f73 (teal) accents + white-haloed math labels

window.KC_DIAGRAMS = {
  s3_double_cover: `
    <figure class="kc-figure">
      <svg viewBox="0 0 680 280" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="S3 Double Cover and Unwinding">
        <defs>
          <radialGradient id="sphereGrad" cx="38%" cy="32%" r="65%">
            <stop offset="0%" stop-color="#f7f5ef"/>
            <stop offset="55%" stop-color="#e4e7f2"/>
            <stop offset="100%" stop-color="#c8cfe2"/>
          </radialGradient>
          <radialGradient id="shadowGrad" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stop-color="#1e242b" stop-opacity="0.20"/>
            <stop offset="100%" stop-color="#1e242b" stop-opacity="0"/>
          </radialGradient>
          <marker id="arrowIndigo" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 1 L 10 5 L 0 9 z" fill="#1b1f8a"/>
          </marker>
          <marker id="arrowTerra" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 1 L 10 5 L 0 9 z" fill="#c4512d"/>
          </marker>
        </defs>

        <ellipse cx="195" cy="248" rx="92" ry="16" fill="url(#shadowGrad)"/>

        <circle cx="190" cy="135" r="96" fill="url(#sphereGrad)" stroke="#1e242b" stroke-width="2.4" stroke-linecap="round"/>
        <path d="M 94 135 A 96 28 0 0 1 286 135" fill="none" stroke="#525d6a" stroke-width="1.0" stroke-dasharray="4 3" opacity="0.65"/>
        <path d="M 94 135 A 96 28 0 0 0 286 135" fill="none" stroke="#1e242b" stroke-width="1.2" opacity="0.8"/>
        <ellipse cx="190" cy="135" rx="34" ry="96" fill="none" stroke="#525d6a" stroke-width="0.9" opacity="0.6"/>

        <circle cx="152" cy="76" r="4.5" fill="#1b1f8a" stroke="#fdfcf8" stroke-width="1.5"/>
        <circle cx="228" cy="194" r="4.5" fill="#c4512d" stroke="#fdfcf8" stroke-width="1.5"/>
        <circle cx="235" cy="82" r="4.5" fill="#2a6f73" stroke="#fdfcf8" stroke-width="1.5"/>

        <path d="M 152 76 Q 194 58 235 82" fill="none" stroke="#1b1f8a" stroke-width="2.2" stroke-linecap="round" marker-end="url(#arrowIndigo)"/>
        <path d="M 228 194 C 130 218, 96 120, 152 76" fill="none" stroke="#c4512d" stroke-width="2.0" stroke-dasharray="5 3" stroke-linecap="round" marker-end="url(#arrowTerra)"/>

        <text x="132" y="64" font-family="EB Garamond, serif" font-style="italic" font-size="16" fill="#1b1f8a" stroke="#fdfcf8" stroke-width="4" stroke-linejoin="round" paint-order="stroke fill">+q</text>
        <text x="242" y="79" font-family="EB Garamond, serif" font-style="italic" font-size="16" fill="#2a6f73" stroke="#fdfcf8" stroke-width="4" stroke-linejoin="round" paint-order="stroke fill">q_d</text>
        <text x="236" y="206" font-family="EB Garamond, serif" font-style="italic" font-size="16" fill="#c4512d" stroke="#fdfcf8" stroke-width="4" stroke-linejoin="round" paint-order="stroke fill">-q</text>
        <text x="168" y="50" font-family="Source Sans 3, sans-serif" font-size="12" font-weight="600" fill="#1b1f8a" stroke="#fdfcf8" stroke-width="3.5" paint-order="stroke fill">Short Geodesic</text>
        <text x="52" y="205" font-family="Source Sans 3, sans-serif" font-size="12" font-weight="600" fill="#c4512d" stroke="#fdfcf8" stroke-width="3.5" paint-order="stroke fill">Unwinding (350°)</text>

        <path d="M 315 135 L 395 135" fill="none" stroke="#1e242b" stroke-width="1.8" stroke-linecap="round" marker-end="url(#arrowIndigo)"/>
        <text x="324" y="123" font-family="Source Sans 3, sans-serif" font-size="12" fill="#525d6a">2-to-1 cover</text>

        <rect x="420" y="62" width="230" height="148" rx="6" fill="#f4f6fb" stroke="#1b1f8a" stroke-width="2.0"/>
        <text x="440" y="92" font-family="Source Sans 3, sans-serif" font-weight="700" font-size="14" fill="#1b1f8a">On SO(3) Directly:</text>
        <text x="440" y="120" font-family="EB Garamond, serif" font-size="16" fill="#1e242b">R(+q) = R(-q) ∈ SO(3)</text>
        <text x="440" y="148" font-family="EB Garamond, serif" font-size="15" fill="#1e242b">Ψ_R = ½ Tr(I₃ − R_dᵀ R)</text>
        <text x="440" y="176" font-family="Source Sans 3, sans-serif" font-size="13" fill="#2a6f73">✓ Zero unwinding</text>
        <text x="440" y="196" font-family="Source Sans 3, sans-serif" font-size="13" fill="#2a6f73">✓ Almost-global stability</text>
      </svg>
      <figcaption class="kc-caption"><strong>Fig. 1.1:</strong> Double cover $S^3 \\to SO(3)$ and the unwinding phenomenon. Antipodal quaternions $+\\mathfrak{q}$ and $-\\mathfrak{q}$ represent the same physical rotation $R \\in SO(3)$, yet continuous $S^3$ feedback from $-\\mathfrak{q}$ traverses the long $350^\\circ$ path instead of the $10^\\circ$ geodesic. Formulating $\\Psi_R$ directly on $SO(3)$ eliminates unwinding.</figcaption>
    </figure>`,

  se3_s2_manifold: `
    <figure class="kc-figure">
      <svg viewBox="0 0 680 300" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Quadrotor and Cable-Suspended Load on SE(3) x S2">
        <defs>
          <radialGradient id="s2Sphere" cx="35%" cy="30%" r="70%">
            <stop offset="0%" stop-color="#f7f5ef"/>
            <stop offset="55%" stop-color="#dbe6f2"/>
            <stop offset="100%" stop-color="#b4c8df"/>
          </radialGradient>
          <radialGradient id="s2Shadow" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stop-color="#1b1f8a" stop-opacity="0.20"/>
            <stop offset="100%" stop-color="#1b1f8a" stop-opacity="0"/>
          </radialGradient>
          <marker id="arr2Indigo" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 1 L 10 5 L 0 9 z" fill="#1b1f8a"/>
          </marker>
          <marker id="arr2Terra" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 1 L 10 5 L 0 9 z" fill="#c4512d"/>
          </marker>
          <marker id="arr2Teal" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 1 L 10 5 L 0 9 z" fill="#2a6f73"/>
          </marker>
        </defs>

        <!-- Ground cast shadow -->
        <ellipse cx="225" cy="272" rx="115" ry="18" fill="url(#s2Shadow)"/>

        <!-- Inertial frame E -->
        <line x1="65" y1="245" x2="120" y2="245" stroke="#525d6a" stroke-width="1.4" marker-end="url(#arr2Teal)"/>
        <line x1="65" y1="245" x2="65" y2="195" stroke="#525d6a" stroke-width="1.4" marker-end="url(#arr2Teal)"/>
        <line x1="65" y1="245" x2="40" y2="266" stroke="#525d6a" stroke-width="1.4" marker-end="url(#arr2Teal)"/>
        <text x="52" y="188" font-family="EB Garamond, serif" font-style="italic" font-size="14" fill="#525d6a">e₃</text>

        <!-- S^2 Configuration Sphere centered at quadrotor x_Q -->
        <circle cx="225" cy="110" r="82" fill="url(#s2Sphere)" fill-opacity="0.45" stroke="#1e242b" stroke-width="1.8"/>
        <path d="M 143 110 A 82 22 0 0 1 307 110" fill="none" stroke="#525d6a" stroke-width="0.9" stroke-dasharray="4 3" opacity="0.65"/>
        <path d="M 143 110 A 82 22 0 0 0 307 110" fill="none" stroke="#1e242b" stroke-width="1.0" opacity="0.75"/>

        <!-- Quadrotor Rigid Body at center (225, 110) -->
        <line x1="188" y1="98" x2="262" y2="122" stroke="#1e242b" stroke-width="3.0" stroke-linecap="round"/>
        <line x1="198" y1="124" x2="252" y2="96" stroke="#1e242b" stroke-width="2.6" stroke-linecap="round"/>
        <ellipse cx="188" cy="98" rx="14" ry="5" fill="#f2efe6" stroke="#1e242b" stroke-width="1.6"/>
        <ellipse cx="262" cy="122" rx="14" ry="5" fill="#f2efe6" stroke="#1e242b" stroke-width="1.6"/>
        <ellipse cx="198" cy="124" rx="12" ry="4.5" fill="#f2efe6" stroke="#1e242b" stroke-width="1.4"/>
        <ellipse cx="252" cy="96" rx="12" ry="4.5" fill="#f2efe6" stroke="#1e242b" stroke-width="1.4"/>

        <!-- Body thrust vector f R e_3 -->
        <line x1="225" y1="110" x2="245" y2="54" stroke="#1b1f8a" stroke-width="2.2" stroke-linecap="round" marker-end="url(#arr2Indigo)"/>
        <text x="252" y="56" font-family="EB Garamond, serif" font-style="italic" font-size="15" fill="#1b1f8a" stroke="#fdfcf8" stroke-width="4" paint-order="stroke fill">f R e₃</text>

        <!-- Cable vector q on S^2 down to Load -->
        <line x1="225" y1="110" x2="262" y2="184" stroke="#1e242b" stroke-width="2.4" stroke-linecap="round"/>
        <!-- Load mass m_L -->
        <circle cx="262" cy="184" r="7.5" fill="#c4512d" stroke="#1e242b" stroke-width="2.0"/>

        <!-- Tangent plane T_q S^2 at load -->
        <polygon points="222,190 292,162 316,182 246,210" fill="#ffffff" fill-opacity="0.78" stroke="#2a6f73" stroke-width="1.3"/>
        <!-- Velocity q_dot = omega x q in T_q S^2 -->
        <line x1="262" y1="184" x2="298" y2="170" stroke="#2a6f73" stroke-width="2.1" stroke-linecap="round" marker-end="url(#arr2Teal)"/>

        <!-- Haloed labels -->
        <text x="206" y="102" font-family="EB Garamond, serif" font-style="italic" font-size="15" fill="#1e242b" stroke="#fdfcf8" stroke-width="4" paint-order="stroke fill">x_Q, R</text>
        <text x="248" y="152" font-family="EB Garamond, serif" font-style="italic" font-size="16" fill="#c4512d" stroke="#fdfcf8" stroke-width="4" paint-order="stroke fill">l q</text>
        <text x="268" y="205" font-family="EB Garamond, serif" font-style="italic" font-size="15" fill="#c4512d" stroke="#fdfcf8" stroke-width="4" paint-order="stroke fill">x_L = x_Q + l q</text>
        <text x="303" y="168" font-family="EB Garamond, serif" font-style="italic" font-size="14" fill="#2a6f73" stroke="#fdfcf8" stroke-width="4" paint-order="stroke fill">q̇ = ω × q</text>

        <!-- Summary Product Manifold Card -->
        <rect x="390" y="55" width="260" height="186" rx="6" fill="#f4f6fb" stroke="#1b1f8a" stroke-width="2.0"/>
        <text x="410" y="85" font-family="Source Sans 3, sans-serif" font-weight="700" font-size="14" fill="#1b1f8a">Product Configuration Space</text>
        <text x="410" y="114" font-family="EB Garamond, serif" font-size="17" fill="#1e242b">Q = SE(3) × S²</text>
        <text x="410" y="142" font-family="Source Sans 3, sans-serif" font-size="13" fill="#525d6a">• Quadrotor pose: (x_Q, R) ∈ ℝ³ × SO(3)</text>
        <text x="410" y="164" font-family="Source Sans 3, sans-serif" font-size="13" fill="#525d6a">• Cable direction: q ∈ S² (‖q‖ = 1)</text>
        <text x="410" y="186" font-family="Source Sans 3, sans-serif" font-size="13" fill="#525d6a">• Tangent space: ω · q = 0, Ω ∈ ℝ³</text>
        <text x="410" y="214" font-family="Source Sans 3, sans-serif" font-size="13" font-weight="600" fill="#2a6f73">Dimension: 6 + 2 = 8 DOF</text>
      </svg>
      <figcaption class="kc-caption"><strong>Fig. 2.1:</strong> Product configuration manifold $Q = SE(3) \\times S^2$ for a quadrotor with a cable-suspended load. The unit cable vector $q \\in S^2$ evolves on the sphere centered at $x_Q$, with velocity $\\dot{q} = \\omega \\times q \\in T_q S^2$ orthogonal to $q$.</figcaption>
    </figure>`,

  transport_map: `
    <figure class="kc-figure">
      <svg viewBox="0 0 680 270" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Compatible Transport Map on Manifold">
        <defs>
          <linearGradient id="manifoldPatch" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#f2efe6"/>
            <stop offset="55%" stop-color="#dfe4f0"/>
            <stop offset="100%" stop-color="#c6cee0"/>
          </linearGradient>
          <radialGradient id="tmShadow" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stop-color="#1b1f8a" stop-opacity="0.18"/>
            <stop offset="100%" stop-color="#1b1f8a" stop-opacity="0"/>
          </radialGradient>
          <marker id="vecBlue" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 1 L 10 5 L 0 9 z" fill="#1b1f8a"/>
          </marker>
          <marker id="vecRed" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 1 L 10 5 L 0 9 z" fill="#c4512d"/>
          </marker>
        </defs>

        <ellipse cx="340" cy="252" rx="220" ry="14" fill="url(#tmShadow)"/>

        <path d="M 70 195 Q 240 125 360 155 T 610 130 L 570 235 Q 410 250 280 230 T 95 245 Z"
              fill="url(#manifoldPatch)" stroke="#1e242b" stroke-width="2.4" stroke-linejoin="round"/>
        <path d="M 140 185 Q 320 148 540 165" fill="none" stroke="#525d6a" stroke-width="0.9" opacity="0.5"/>
        <path d="M 110 215 Q 310 188 565 198" fill="none" stroke="#525d6a" stroke-width="0.9" opacity="0.5"/>

        <polygon points="145,115 275,92 310,148 180,171" fill="#ffffff" fill-opacity="0.76" stroke="#1b1f8a" stroke-width="1.4"/>
        <polygon points="395,95 525,78 558,136 428,153" fill="#ffffff" fill-opacity="0.76" stroke="#c4512d" stroke-width="1.4"/>

        <circle cx="228" cy="132" r="4.5" fill="#1b1f8a" stroke="#fdfcf8" stroke-width="1.5"/>
        <circle cx="476" cy="116" r="4.5" fill="#c4512d" stroke="#fdfcf8" stroke-width="1.5"/>

        <path d="M 228 132 Q 350 158 476 116" fill="none" stroke="#1e242b" stroke-width="1.8" stroke-dasharray="5 3"/>

        <line x1="228" y1="132" x2="268" y2="104" stroke="#1b1f8a" stroke-width="2.2" stroke-linecap="round" marker-end="url(#vecBlue)"/>
        <line x1="476" y1="116" x2="518" y2="92" stroke="#1b1f8a" stroke-width="2.0" stroke-dasharray="4 2" stroke-linecap="round" marker-end="url(#vecBlue)"/>
        <line x1="476" y1="116" x2="495" y2="146" stroke="#c4512d" stroke-width="2.2" stroke-linecap="round" marker-end="url(#vecRed)"/>

        <path d="M 272 95 Q 375 35 488 85" fill="none" stroke="#2a6f73" stroke-width="2.0" stroke-linecap="round" marker-end="url(#vecBlue)"/>

        <text x="155" y="106" font-family="Source Sans 3, sans-serif" font-size="13" font-weight="600" fill="#1b1f8a" stroke="#fdfcf8" stroke-width="3.5" paint-order="stroke fill">T_{q_d} M</text>
        <text x="405" y="88" font-family="Source Sans 3, sans-serif" font-size="13" font-weight="600" fill="#c4512d" stroke="#fdfcf8" stroke-width="3.5" paint-order="stroke fill">T_q M</text>
        <text x="216" y="152" font-family="EB Garamond, serif" font-style="italic" font-size="16" fill="#1b1f8a" stroke="#fdfcf8" stroke-width="4" paint-order="stroke fill">q_d</text>
        <text x="470" y="138" font-family="EB Garamond, serif" font-style="italic" font-size="16" fill="#c4512d" stroke="#fdfcf8" stroke-width="4" paint-order="stroke fill">q</text>
        <text x="322" y="48" font-family="Source Sans 3, sans-serif" font-size="13" font-weight="700" fill="#2a6f73" stroke="#fdfcf8" stroke-width="4" paint-order="stroke fill">Transport Map T(q, q_d)</text>
        <text x="322" y="170" font-family="EB Garamond, serif" font-size="15" fill="#1e242b" stroke="#fdfcf8" stroke-width="4" paint-order="stroke fill">Ψ(q, q_d)</text>
      </svg>
      <figcaption class="kc-caption"><strong>Fig. 3.1:</strong> Compatible transport map $T(q, q_d) : T_{q_d} M \\to T_q M$ satisfying $d_2\\Psi = -T^*(q, q_d) d_1\\Psi$, enabling coordinate-free velocity error $e_{\\dot{q}} = \\dot{q} - T(q, q_d)\\dot{q}_d$ across distinct tangent spaces.</figcaption>
    </figure>`,

  lagrange_dalembert_variation: `
    <figure class="kc-figure">
      <svg viewBox="0 0 680 270" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Coordinate-Free Variations on Manifold">
        <defs>
          <linearGradient id="ldPatch" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#f6efe2"/>
            <stop offset="60%" stop-color="#e3ecf6"/>
            <stop offset="100%" stop-color="#c3d4e8"/>
          </linearGradient>
          <radialGradient id="ldShadow" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stop-color="#1e242b" stop-opacity="0.18"/>
            <stop offset="100%" stop-color="#1e242b" stop-opacity="0"/>
          </radialGradient>
          <marker id="arr4Indigo" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 1 L 10 5 L 0 9 z" fill="#1b1f8a"/>
          </marker>
          <marker id="arr4Terra" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 1 L 10 5 L 0 9 z" fill="#c4512d"/>
          </marker>
        </defs>

        <ellipse cx="330" cy="248" rx="220" ry="14" fill="url(#ldShadow)"/>

        <!-- Manifold M -->
        <path d="M 55 180 Q 210 110 355 135 T 625 115 L 590 225 Q 410 245 265 220 T 80 235 Z"
              fill="url(#ldPatch)" stroke="#1e242b" stroke-width="2.4" stroke-linejoin="round"/>

        <!-- Nominal Trajectory R(t), q(t) -->
        <path d="M 145 190 Q 320 142 520 170" fill="none" stroke="#1b1f8a" stroke-width="2.4" stroke-linecap="round"/>
        <!-- Perturbed Trajectory R_eps(t) -->
        <path d="M 145 190 Q 320 95 520 170" fill="none" stroke="#c4512d" stroke-width="2.0" stroke-dasharray="5 3" stroke-linecap="round"/>

        <!-- Endpoints fixed at t0 and tf -->
        <circle cx="145" cy="190" r="4.5" fill="#1e242b" stroke="#fdfcf8" stroke-width="1.5"/>
        <circle cx="520" cy="170" r="4.5" fill="#1e242b" stroke="#fdfcf8" stroke-width="1.5"/>
        <circle cx="325" cy="156" r="4" fill="#1b1f8a"/>

        <!-- Variation Vector delta R = R eta^ -->
        <line x1="325" y1="156" x2="325" y2="108" stroke="#c4512d" stroke-width="2.2" stroke-linecap="round" marker-end="url(#arr4Terra)"/>

        <!-- Labels with white halos -->
        <text x="128" y="212" font-family="EB Garamond, serif" font-style="italic" font-size="15" fill="#1e242b" stroke="#fdfcf8" stroke-width="4" paint-order="stroke fill">t₀</text>
        <text x="526" y="190" font-family="EB Garamond, serif" font-style="italic" font-size="15" fill="#1e242b" stroke="#fdfcf8" stroke-width="4" paint-order="stroke fill">t_f</text>
        <text x="336" y="128" font-family="EB Garamond, serif" font-style="italic" font-size="15" fill="#c4512d" stroke="#fdfcf8" stroke-width="4" paint-order="stroke fill">δR = R η∧,  δq = ξ × q</text>
        <text x="200" y="185" font-family="Source Sans 3, sans-serif" font-size="12" font-weight="600" fill="#1b1f8a" stroke="#fdfcf8" stroke-width="3.5" paint-order="stroke fill">Nominal Trajectory (R(t), q(t))</text>
        <text x="220" y="92" font-family="Source Sans 3, sans-serif" font-size="12" font-weight="600" fill="#c4512d" stroke="#fdfcf8" stroke-width="3.5" paint-order="stroke fill">Perturbed Curve exp(ε η∧)</text>
      </svg>
      <figcaption class="kc-caption"><strong>Fig. 4.1:</strong> Coordinate-free exponential variations on $SO(3) \\times S^2$ with fixed endpoints at $t_0, t_f$. The Lagrange-d'Alembert principle $\\delta \\int L\\,dt + \\int \\delta W\\,dt = 0$ yields Euler-Poincaré equations directly on the manifold without Christoffel symbols or Euler-angle charts.</figcaption>
    </figure>`,

  variation_lqr_tube: `
    <figure class="kc-figure">
      <svg viewBox="0 0 680 270" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Variation-Based Linearization and LTV-LQR Tube">
        <defs>
          <linearGradient id="tubeGrad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stop-color="#c7d0e8" stop-opacity="0.55"/>
            <stop offset="100%" stop-color="#d8ebe8" stop-opacity="0.55"/>
          </linearGradient>
          <marker id="arr7Indigo" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 1 L 10 5 L 0 9 z" fill="#1b1f8a"/>
          </marker>
          <marker id="arr7Teal" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 1 L 10 5 L 0 9 z" fill="#2a6f73"/>
          </marker>
        </defs>

        <!-- LTV-LQR Contraction Tube -->
        <path d="M 75 95 C 210 105, 420 128, 600 132 L 600 156 C 420 160, 210 185, 75 205 Z"
              fill="url(#tubeGrad)" stroke="#525d6a" stroke-width="1.0" stroke-dasharray="4 3"/>

        <!-- Nominal Desired Trajectory x_d(t) -->
        <path d="M 75 150 C 230 145, 420 144, 600 144" fill="none" stroke="#1b1f8a" stroke-width="2.4" stroke-linecap="round" marker-end="url(#arr7Indigo)"/>

        <!-- Perturbed Trajectory converging via LTV-LQR -->
        <path d="M 85 104 C 190 112, 330 134, 575 142" fill="none" stroke="#c4512d" stroke-width="2.0" stroke-linecap="round"/>
        <path d="M 85 196 C 200 184, 340 156, 575 146" fill="none" stroke="#2a6f73" stroke-width="2.0" stroke-linecap="round"/>

        <!-- Tangent-space error vectors delta x(t) -->
        <line x1="175" y1="147" x2="175" y2="117" stroke="#1e242b" stroke-width="1.6" marker-end="url(#arr7Teal)"/>
        <circle cx="175" cy="147" r="4" fill="#1b1f8a"/>
        <circle cx="175" cy="115" r="4" fill="#c4512d"/>

        <text x="184" y="128" font-family="EB Garamond, serif" font-style="italic" font-size="15" fill="#1e242b" stroke="#fdfcf8" stroke-width="4" paint-order="stroke fill">δx = (η, δΩ, ξ, δω) ∈ T_{x_d} M</text>
        <text x="415" y="120" font-family="Source Sans 3, sans-serif" font-size="12" font-weight="700" fill="#1b1f8a" stroke="#fdfcf8" stroke-width="3.5" paint-order="stroke fill">Nominal x_d(t)</text>
        <text x="360" y="182" font-family="Source Sans 3, sans-serif" font-size="12" font-weight="600" fill="#2a6f73" stroke="#fdfcf8" stroke-width="3.5" paint-order="stroke fill">LTV-LQR: δu = −K(t) δx</text>
        <text x="90" y="238" font-family="Source Sans 3, sans-serif" font-size="12" fill="#525d6a">Tangent-space Riccati P(t) projected onto constraint manifold ξ · q_d = 0</text>
      </svg>
      <figcaption class="kc-caption"><strong>Fig. 7.1:</strong> Symbolic variation-based linearization and LTV-LQR feedback along a nominal trajectory $x_d(t)$ on $SO(3) \\times S^2$. State variations $\\delta x(t) \\in T_{x_d(t)}M$ obey linear time-varying dynamics $\\delta\\dot{x} = A(t)\\delta x + B(t)\\delta u$ while respecting the orthogonal constraint $\\xi \\cdot q_d = 0$.</figcaption>
    </figure>`,

  geomech_symbolic_tree: `
    <figure class="kc-figure">
      <svg viewBox="0 0 680 280" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Frozen Expression Tree and 6-Stage geomech Pipeline">
        <defs>
          <radialGradient id="astShadow" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stop-color="#1b1f8a" stop-opacity="0.16"/>
            <stop offset="100%" stop-color="#1b1f8a" stop-opacity="0"/>
          </radialGradient>
          <marker id="arr5Indigo" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 1 L 10 5 L 0 9 z" fill="#1b1f8a"/>
          </marker>
        </defs>

        <ellipse cx="165" cy="252" rx="135" ry="14" fill="url(#astShadow)"/>

        <text x="50" y="34" font-family="Source Sans 3, sans-serif" font-weight="700" font-size="13" fill="#1b1f8a">1. Frozen Expression Tree (Expr)</text>

        <line x1="165" y1="68" x2="105" y2="118" stroke="#1e242b" stroke-width="1.8"/>
        <line x1="165" y1="68" x2="225" y2="118" stroke="#1e242b" stroke-width="1.8"/>
        <line x1="225" y1="118" x2="175" y2="174" stroke="#1e242b" stroke-width="1.8"/>
        <line x1="225" y1="118" x2="275" y2="174" stroke="#1e242b" stroke-width="1.8"/>
        <line x1="275" y1="174" x2="242" y2="224" stroke="#525d6a" stroke-width="1.4"/>
        <line x1="275" y1="174" x2="306" y2="224" stroke="#525d6a" stroke-width="1.4"/>

        <rect x="135" y="52" width="60" height="28" rx="5" fill="#f4f6fb" stroke="#1b1f8a" stroke-width="2.0"/>
        <text x="165" y="71" text-anchor="middle" font-family="JetBrains Mono, monospace" font-size="12" fill="#1b1f8a">Mul</text>

        <rect x="75" y="104" width="60" height="28" rx="5" fill="#fbf7ef" stroke="#b07322" stroke-width="1.6"/>
        <text x="105" y="123" text-anchor="middle" font-family="JetBrains Mono, monospace" font-size="12" fill="#1e242b">0.5</text>

        <rect x="195" y="104" width="60" height="28" rx="5" fill="#f4f6fb" stroke="#1b1f8a" stroke-width="2.0"/>
        <text x="225" y="123" text-anchor="middle" font-family="JetBrains Mono, monospace" font-size="12" fill="#1b1f8a">Dot</text>

        <rect x="145" y="160" width="60" height="28" rx="5" fill="#f3f8f8" stroke="#2a6f73" stroke-width="1.8"/>
        <text x="175" y="179" text-anchor="middle" font-family="EB Garamond, serif" font-style="italic" font-size="15" fill="#2a6f73">Ω</text>

        <rect x="242" y="160" width="66" height="28" rx="5" fill="#f4f6fb" stroke="#1b1f8a" stroke-width="1.8"/>
        <text x="275" y="179" text-anchor="middle" font-family="JetBrains Mono, monospace" font-size="11.5" fill="#1b1f8a">MVMul</text>

        <rect x="218" y="210" width="46" height="26" rx="4" fill="#fdfcf8" stroke="#1e242b" stroke-width="1.5"/>
        <text x="241" y="228" text-anchor="middle" font-family="EB Garamond, serif" font-style="italic" font-size="14" fill="#1e242b">J*</text>

        <rect x="282" y="210" width="46" height="26" rx="4" fill="#f3f8f8" stroke="#2a6f73" stroke-width="1.5"/>
        <text x="305" y="228" text-anchor="middle" font-family="EB Garamond, serif" font-style="italic" font-size="14" fill="#2a6f73">Ω</text>

        <line x1="336" y1="142" x2="386" y2="142" stroke="#1b1f8a" stroke-width="2.2" marker-end="url(#arr5Indigo)"/>
        <text x="361" y="130" text-anchor="middle" font-family="Source Sans 3, sans-serif" font-size="11" font-weight="700" fill="#1b1f8a">δ, IBP</text>

        <rect x="402" y="34" width="252" height="218" rx="6" fill="#f4f6fb" stroke="#1b1f8a" stroke-width="2.0"/>
        <text x="420" y="60" font-family="Source Sans 3, sans-serif" font-weight="700" font-size="13.5" fill="#1b1f8a">2. Pure Tree-Rewrite Pipeline</text>
        <text x="420" y="88" font-family="Source Sans 3, sans-serif" font-size="12.5" fill="#1e242b">① δS = δL + δW (product rule)</text>
        <text x="420" y="112" font-family="Source Sans 3, sans-serif" font-size="12.5" fill="#c4512d">② Manifold rules: δω, δΩ</text>
        <text x="420" y="136" font-family="Source Sans 3, sans-serif" font-size="12.5" fill="#1e242b">③ full_simplify (pull, vector, elim)</text>
        <text x="420" y="160" font-family="Source Sans 3, sans-serif" font-size="12.5" fill="#2a6f73">④ collect + integrate_by_parts</text>
        <text x="420" y="184" font-family="Source Sans 3, sans-serif" font-size="12.5" fill="#1e242b">⑤ expand + extract_linear_coeff</text>
        <text x="420" y="212" font-family="EB Garamond, serif" font-size="15" fill="#1b1f8a">⇒ −J Ω̇ − Ω × JΩ + M = 0</text>
        <text x="420" y="236" font-family="Source Sans 3, sans-serif" font-size="12" font-weight="600" fill="#2a6f73">✓ check_eom residual &lt; 10⁻⁷</text>
      </svg>
      <figcaption class="kc-caption"><strong>Fig. 5.1:</strong> Frozen symbolic expression tree in <code>geomech</code> for rigid-body kinetic energy $\\frac{1}{2}\\Omega \\cdot J\\Omega$ and the 6-stage pure tree-rewrite pipeline (<code>compute_eom</code>) yielding coordinate-free Euler–Poincaré equations verified by <code>check_eom</code>.</figcaption>
    </figure>`
};
