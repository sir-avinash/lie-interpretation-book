// Keenan Crane-inspired SVG illustrations:
// 1. 3D-projected geometry with thick silhouettes (#1e242b), thin contours, dashed hidden lines
// 2. Muted pastel matte radial/linear shading + 20% opacity cast shadows
// 3. #1b1f8a (indigo) & #c4512d (terracotta) vector accents + white-haloed math labels

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

        <!-- Cast Shadow (20% opacity) -->
        <ellipse cx="195" cy="248" rx="92" ry="16" fill="url(#shadowGrad)"/>

        <!-- S^3 Hemisphere / Sphere Projection -->
        <circle cx="190" cy="135" r="96" fill="url(#sphereGrad)" stroke="#1e242b" stroke-width="2.4"/>
        <!-- Dashed hidden back equator -->
        <path d="M 94 135 A 96 28 0 0 1 286 135" fill="none" stroke="#525d6a" stroke-width="1.2" stroke-dasharray="4,4"/>
        <!-- Solid front equator -->
        <path d="M 94 135 A 96 28 0 0 0 286 135" fill="none" stroke="#1e242b" stroke-width="1.4"/>
        <!-- Thin meridian contour -->
        <ellipse cx="190" cy="135" rx="34" ry="96" fill="none" stroke="#525d6a" stroke-width="0.9" opacity="0.65"/>

        <!-- Antipodal points q and -q -->
        <circle cx="152" cy="76" r="4.5" fill="#1b1f8a" stroke="#fdfcf8" stroke-width="1.5"/>
        <circle cx="228" cy="194" r="4.5" fill="#c4512d" stroke="#fdfcf8" stroke-width="1.5"/>
        <circle cx="235" cy="82" r="4.5" fill="#2d6a4f" stroke="#fdfcf8" stroke-width="1.5"/>

        <!-- Short Geodesic Arc (q -> q_d) -->
        <path d="M 152 76 Q 194 58 235 82" fill="none" stroke="#1b1f8a" stroke-width="2.4" marker-end="url(#arrowIndigo)"/>
        <!-- Long Unwinding Arc (-q -> q_d) -->
        <path d="M 228 194 C 130 218, 96 120, 152 76" fill="none" stroke="#c4512d" stroke-width="2.2" stroke-dasharray="6,3" marker-end="url(#arrowTerra)"/>

        <!-- White-haloed Labels -->
        <text x="132" y="64" font-family="EB Garamond, serif" font-style="italic" font-size="16" fill="#1b1f8a" stroke="#fdfcf8" stroke-width="3" paint-order="stroke">+q</text>
        <text x="242" y="79" font-family="EB Garamond, serif" font-style="italic" font-size="16" fill="#2d6a4f" stroke="#fdfcf8" stroke-width="3" paint-order="stroke">q_d</text>
        <text x="236" y="206" font-family="EB Garamond, serif" font-style="italic" font-size="16" fill="#c4512d" stroke="#fdfcf8" stroke-width="3" paint-order="stroke">-q</text>
        <text x="168" y="54" font-family="Source Sans 3, sans-serif" font-size="12" font-weight="600" fill="#1b1f8a">Short Geodesic</text>
        <text x="58" y="205" font-family="Source Sans 3, sans-serif" font-size="12" font-weight="600" fill="#c4512d">Unwinding (350°)</text>

        <!-- Right side: Quotient Map to SO(3) -->
        <path d="M 315 135 L 395 135" fill="none" stroke="#1e242b" stroke-width="1.8" marker-end="url(#arrowIndigo)"/>
        <text x="326" y="124" font-family="Source Sans 3, sans-serif" font-size="12" fill="#525d6a">2-to-1 cover</text>

        <!-- SO(3) Geodesic Error Ψ_R Box -->
        <rect x="420" y="62" width="230" height="148" rx="8" fill="#f6f4ee" stroke="#1e242b" stroke-width="2"/>
        <text x="440" y="92" font-family="Source Sans 3, sans-serif" font-weight="700" font-size="14" fill="#1b1f8a">On SO(3) Directly:</text>
        <text x="440" y="120" font-family="EB Garamond, serif" font-size="16" fill="#1e242b">R(+q) = R(-q) ∈ SO(3)</text>
        <text x="440" y="148" font-family="EB Garamond, serif" font-size="15" fill="#1e242b">Ψ_R = ½ Tr(I₃ − R_dᵀ R)</text>
        <text x="440" y="176" font-family="Source Sans 3, sans-serif" font-size="13" fill="#2d6a4f">✓ Zero unwinding</text>
        <text x="440" y="196" font-family="Source Sans 3, sans-serif" font-size="13" fill="#2d6a4f">✓ Almost-global stability</text>
      </svg>
      <figcaption class="kc-caption"><strong>Fig. 1.1:</strong> Double cover $S^3 \\to SO(3)$ and the unwinding phenomenon. Antipodal quaternions $+\\mathfrak{q}$ and $-\\mathfrak{q}$ represent the same physical rotation $R \\in SO(3)$, yet continuous $S^3$ feedback from $-\\mathfrak{q}$ traverses the long $350^\\circ$ path instead of the $10^\\circ$ geodesic. Formulating $\\Psi_R$ directly on $SO(3)$ eliminates unwinding.</figcaption>
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
          <marker id="vecBlue" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 1 L 10 5 L 0 9 z" fill="#1b1f8a"/>
          </marker>
          <marker id="vecRed" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 1 L 10 5 L 0 9 z" fill="#c4512d"/>
          </marker>
        </defs>

        <!-- Curved Manifold Patch M -->
        <path d="M 70 195 Q 240 125 360 155 T 610 130 L 570 235 Q 410 250 280 230 T 95 245 Z"
              fill="url(#manifoldPatch)" stroke="#1e242b" stroke-width="2.2"/>
        <!-- Thin coordinate grid lines on M -->
        <path d="M 140 185 Q 320 148 540 165" fill="none" stroke="#525d6a" stroke-width="0.9" opacity="0.5"/>
        <path d="M 110 215 Q 310 188 565 198" fill="none" stroke="#525d6a" stroke-width="0.9" opacity="0.5"/>

        <!-- Tangent Plane at q_d -->
        <polygon points="145,115 275,92 310,148 180,171" fill="#ffffff" fill-opacity="0.72" stroke="#1b1f8a" stroke-width="1.4"/>
        <!-- Tangent Plane at q -->
        <polygon points="395,95 525,78 558,136 428,153" fill="#ffffff" fill-opacity="0.72" stroke="#c4512d" stroke-width="1.4"/>

        <!-- Points q_d and q on M -->
        <circle cx="228" cy="132" r="4.5" fill="#1b1f8a" stroke="#fdfcf8" stroke-width="1.5"/>
        <circle cx="476" cy="116" r="4.5" fill="#c4512d" stroke="#fdfcf8" stroke-width="1.5"/>

        <!-- Geodesic connecting q_d and q -->
        <path d="M 228 132 Q 350 158 476 116" fill="none" stroke="#1e242b" stroke-width="1.8" stroke-dasharray="5,3"/>

        <!-- Tangent vectors -->
        <line x1="228" y1="132" x2="268" y2="104" stroke="#1b1f8a" stroke-width="2.4" marker-end="url(#vecBlue)"/>
        <line x1="476" y1="116" x2="518" y2="92" stroke="#1b1f8a" stroke-width="2.0" stroke-dasharray="4,2" marker-end="url(#vecBlue)"/>
        <line x1="476" y1="116" x2="495" y2="146" stroke="#c4512d" stroke-width="2.4" marker-end="url(#vecRed)"/>

        <!-- Transport Map Arc -->
        <path d="M 272 95 Q 375 35 488 85" fill="none" stroke="#2d6a4f" stroke-width="2.0" marker-end="url(#vecBlue)"/>

        <!-- Labels -->
        <text x="155" y="106" font-family="Source Sans 3, sans-serif" font-size="13" font-weight="600" fill="#1b1f8a">T_{q_d} M</text>
        <text x="405" y="88" font-family="Source Sans 3, sans-serif" font-size="13" font-weight="600" fill="#c4512d">T_q M</text>
        <text x="216" y="152" font-family="EB Garamond, serif" font-style="italic" font-size="16" fill="#1b1f8a">q_d</text>
        <text x="470" y="138" font-family="EB Garamond, serif" font-style="italic" font-size="16" fill="#c4512d">q</text>
        <text x="330" y="48" font-family="Source Sans 3, sans-serif" font-size="13" font-weight="700" fill="#2d6a4f">Transport Map T(q, q_d)</text>
        <text x="322" y="170" font-family="EB Garamond, serif" font-size="15" fill="#1e242b">Ψ(q, q_d)</text>
      </svg>
      <figcaption class="kc-caption"><strong>Fig. 3.1:</strong> Compatible transport map $T(q, q_d) : T_{q_d} M \\to T_q M$ satisfying $d_2\\Psi = -T^*(q, q_d) d_1\\Psi$, enabling coordinate-free velocity error $e_{\\dot{q}} = \\dot{q} - T(q, q_d)\\dot{q}_d$ across distinct tangent spaces.</figcaption>
    </figure>`
};
