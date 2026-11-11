---
name: frontend-design
description: Create distinctive, production-grade frontend interfaces with high design quality. Use this skill when the user asks to build web components, pages, artifacts, posters, or applications (examples include websites, landing pages, dashboards, React components, HTML/CSS layouts, or when styling/beautifying any web UI). Generates creative, polished code and UI design that avoids generic AI aesthetics.
license: Complete terms in LICENSE.txt (https://github.com/anthropics/skills/tree/main/skills/frontend-design)
---

# Frontend Design Skill (`anthropics/skills`)

This skill guides creation of distinctive, production-grade frontend interfaces that avoid generic "AI slop" aesthetics. Implement real working code with exceptional attention to aesthetic details and creative choices.

## 1. Design Thinking & Grounding

Before coding, understand the context and commit to a **BOLD aesthetic direction**:
- **Purpose**: What problem does this interface solve? Who uses it?
- **Tone**: Pick an intentional, domain-authentic aesthetic: brutally minimal, tactical telemetry/command-center, editorial/magazine, retro-futuristic, Swiss grid precision, industrial/utilitarian, etc.
- **Constraints**: Technical requirements (framework, performance, accessibility, bilingual Thai/English legibility).
- **Differentiation**: What makes this unforgettable? What is the one signature visual element someone will remember?

## 2. Explicit Design Plan (4–6 Named Hex Tokens & Typography Pairing)

Always establish:
1. **Color & Atmospheric Polarity**: Commit to a cohesive palette using 4–6 named hex tokens as CSS variables. Dominant surfaces with sharp, high-contrast functional accents outperform timid, evenly distributed gradients.
2. **Typography**: Choose fonts that are characterful and domain-appropriate. Pair a distinctive display font (e.g., `Chakra Petch`, `Space Grotesk`, `Syne`) with a refined body font (`IBM Plex Sans Thai`, `Plus Jakarta Sans`) and a true monospace font (`JetBrains Mono` with `tabular-nums`) for data/telemetry. Never default to plain Arial/Inter everywhere.
3. **Spatial Composition & Layout**: Use architectural grids, exposed hairlines, asymmetric split-panes, generous negative space or intentional high-density telemetry panels rather than uniform rounded card grids.
4. **Motion & Micro-interactions**: Use purposeful CSS/SVG animations — staggered reveals, pulsing alert beacons, smooth state transitions, keyboard navigation shortcuts (`1-4`), and `prefers-reduced-motion` support.

## 3. Anti-Default Self-Review

Never use:
- Generic overused font stacks without character
- Clichéd purple-to-blue SaaS gradients on generic rounded cards
- Predictable 4-equal-rounded-box Bootstrap templates without visual hierarchy
- Unaligned numbers in tables (always enforce `font-variant-numeric: tabular-nums`)

