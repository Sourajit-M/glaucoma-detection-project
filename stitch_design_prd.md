# GlaucomaDetect — UI Design PRD for Stitch AI

## Overview

**App Name:** GlaucomaDetect  
**Type:** Medical AI research web application  
**Purpose:** Upload a retinal fundus image → get glaucoma prediction, probability score, Grad-CAM heatmap, CDR value, and segmentation overlays. A separate dashboard shows model research results.  
**Pages:** 3 (Predict, Dashboard, About)  
**Device:** Desktop-first, mobile-responsive

---

## Visual Design Identity

### Aesthetic
- **Theme:** Modern Health-Tech — A premium, trustworthy, and calming aesthetic often used by top-tier biotech and digital health platforms. It uses soft clinical blues, slate grays, and warm off-whites instead of stark hospital white or aggressive black.
- **Primary / Brand colour:** Royal Slate Blue — `#2563EB` (for primary actions) and Deep Navy `#0F172A` (for branding/headers)
- **Background:** Soft Slate-tinted Off-White — `#F8FAFC`
- **Surface cards:** Pure White — `#FFFFFF`. Cards use an extremely subtle, expansive drop shadow (`box-shadow: 0 10px 40px rgba(15, 23, 42, 0.04)`) to appear as if they are gently floating off the background.
- **Text primary:** Deep Slate Gray — `#0F172A` (softer and more premium than pure black, reducing eye strain)
- **Text secondary:** Muted Slate — `#64748B`
- **Semantic Colours (Refined & Softened):**
  - **Danger / Glaucoma:** Soft Crimson — `#E11D48` (serious, but not a harsh neon red)
  - **Safe / Normal:** Emerald — `#059669` (calming, natural green)
  - **Warning / Borderline:** Amber — `#D97706`

### Typography
- **Font:** Inter (or optionally 'Plus Jakarta Sans' if available) — clean, modern sans-serif.
- **Headings:** Bold (700 weight), deep navy color, slightly tight letter-spacing.
- **Body:** Regular (400 weight), 16px, very readable 1.6 line-height.
- **Metric numbers:** Semi-bold, large (36–48px), using tabular lining (monospace numbers) for data to align perfectly.

### Visual style details
- **Soft UI:** Focus on gentle curves and soft visual separation. No harsh borders unless necessary.
- **Hover effects:** Buttons smoothly elevate (y-axis shift up 2px) and shadows intensify slightly.
- **Corners:** Medium-soft rounded corners: `border-radius: 12px` for cards and image panels, `8px` for buttons.
- **Dividers:** Very light slate lines (`#E2E8F0`).
- **Loaders:** Smooth pulsing animations or elegant circular spinners in the primary Royal Slate Blue.

---

## Page 1 — Predict (`/`)

**Purpose:** The core feature. Upload a fundus image, analyse it, see results.

### Layout
Two-column grid on desktop (50/50 split), generous padding (40px). Single column on mobile (upload → results stacked).

### Left Panel — Upload
- Solid white card (`#FFFFFF`), `12px` border radius, soft shadow.
- Large centered drag-and-drop zone:
  - Inside a very light blue container (`#EFF6FF`).
  - Dashed border using the primary blue (`#60A5FA`).
  - Eye/Image upload icon (SVG) centred — Royal Slate Blue.
  - Primary text: **"Upload retinal fundus image"** (18px, semi-bold, deep navy).
  - Secondary text: *"Drag and drop, or click to browse"* (muted slate, 14px).
  - Supported formats: Small, subtle gray text `JPG, PNG, BMP (Max 10 MB)`.
- When a file is selected:
  - Beautiful fade-in of the image thumbnail with a subtle 4px border radius.
  - Minimalist `×` button to discard image.
- Primary CTA button: **"Analyse Scan"**
  - Full width, solid Royal Slate Blue (`#2563EB`), white text, 48px height.
  - Hover state: Deepens to `#1D4ED8` and lifts slightly.
  - Disabled state: Pale blue (`#BFDBFE`) with white text.
  - Loading state: A subtle, smooth spinner appears inside the button.

### Right Panel — Results
Initially shows an empty state: A light slate eye/scan illustration with the text *"Upload a scan to view diagnostic results"*.

After successful response (results fade in smoothly):

**1. Verdict Card** (top, full width)
- White card, gently emphasized with a top-border of 4px in the verdict color (Crimson for Glaucoma, Emerald for Normal).
- Verdict text: **"Signs of Glaucoma Detected"** (Crimson) or **"No Signs Detected"** (Emerald) in 28px Semi-bold.
- Probability: Rendered nicely to the side (e.g., `Confidence: 87.3%`).
- Pill badge: Soft background with darker text (e.g., Pale red background `#FFE4E6` with dark red text `#BE123C`).

**2. Biomarker Card (CDR)** (below verdict)
- Clean, minimal card.
- Label: "Cup-to-Disc Ratio (CDR)"
- Numeric value (e.g., `0.71`) in 32px.
- Small indicator bar showing a gradient from green to red, with a pin marking the current value (0.71) to visually demonstrate where it falls relative to the `0.65` clinical threshold.

**3. Image Viewer** (full width)
- Elegantly tabbed interface: `Original Scan` | `Grad-CAM Overlay` | `U-Net Segmentation`
- Active tab has a bold blue underline and navy text; inactive tabs are muted slate.
- The image presentation area has a very soft gray background (`#F1F5F9`) so dark or differently-sized fundus images look naturally framed.

**4. Clinical Disclaimer Banner**
- Pale amber background (`#FEF3C7`) with warm amber text (`#92400E`).
- Small warning icon.
- Text: *"For research and demonstration purposes only. Not a substitute for clinical diagnosis."*

---

## Page 2 — Research Dashboard (`/dashboard`)

**Purpose:** Display the model's performance metrics with academic rigor but modern SaaS styling.

### Layout
Full-width single column. Soft background (`#F8FAFC`), content max-width ~1200px centred.

### Page Header
- Title: **"Clinical Evaluation Metrics"** (Deep Navy)
- Subtitle: *"Performance validated on 1,050 held-out test scans across three distinct medical datasets."*

### Section 1 — Key Metrics Row
Row of 4 metric cards (responsive 2×2 on mobile):
| Metric | Value |
|---|---|
| Best AUC | **0.947** |
| Sensitivity | **93.2%** |
| Test Scans | **1,050** |
| Datasets | **3** |

Each card: White background, soft shadow. A top accent line in Royal Blue. Huge Navy numbers, Slate gray labels.

### Section 2 — ROC Curve Chart
- Title: "Receiver Operating Characteristic (ROC)"
- Encased in a white card.
- Proposed model (`CNN + CDR`): Bold Royal Blue line (`#2563EB`).
- Other models: Muted slate lines.
- The area under the proposed model's curve has a very gentle, transparent blue gradient fill down to the x-axis to make it visually pop.
- Subtle grid lines (`#F1F5F9`).

### Section 3 — Model Performance Table
- Encased in a white card with rounded corners.
- Header row has a very light blue background (`#F8FAFC`).
- Type badges: `Classical ML` (gray pill), `Deep Learning` (purple pill), `Hybrid Ensemble` (blue pill).
- The row for the proposed model has a very faint blue background highlight (`#EFF6FF`).

### Section 4 — Ablation Study
- Horizontal bar chart.
- Proposed variant: Solid Royal Blue bar.
- Other variants: Slate gray bars.
- Values printed neatly inside or just outside the tip of the bars.

---

## Page 3 — About (`/about`)

**Purpose:** Explain the project architecture clearly for recruiters and reviewers.

### Layout
Centred editorial layout, max-width ~800px.

### Sections

**Hero Section**
- Title: **"GlaucomaDetect"** (Deep Navy)
- Description: *"A multi-source feature fusion system for glaucoma detection, combining deep visual features (ResNet18) with clinical structural biomarkers (U-Net)."*
- A beautiful CTA button to view the GitHub repository.

**Architecture Flow (The Pipeline)**
- Designed like a modern flowchart. Connecting lines between steps.
- Each step is a small white card:
  1. **Preprocessing** (Image standardisation)
  2. **CNN Inference** (Global features)
  3. **Segmentation** (Disc/Cup mask generation)
  4. **Meta-Learner** (Logistic Regression fusion)
- Use small, elegant outline icons for each step.

**Tech Stack**
- Extremely clean pills with soft backgrounds: 
  - e.g., `PyTorch` (pale orange background, dark orange text)
  - `FastAPI` (pale teal background, dark teal text)
  - `React` (pale blue background, dark blue text)

---

## Navigation Bar (Shared)

Fixed top navbar, full width, white background (`#FFFFFF`), with a delicate bottom shadow (`0 1px 2px rgba(15, 23, 42, 0.05)`).
- **Left:** App name "GlaucomaDetect" in Deep Navy, optionally accompanied by a small blue medical cross or eye icon.
- **Centre:** Nav links — `Predict` · `Dashboard` · `About`.
  - Active link is Royal Blue. Inactive is Slate Gray.
- **Right:** GitHub icon button.

---

## Design Configuration Summary

| Token | Value |
|---|---|
| Primary Royal Blue | `#2563EB` |
| Deep Navy (Headers) | `#0F172A` |
| App Background | `#F8FAFC` (Slate Off-White) |
| Surface / Cards | `#FFFFFF` (Pure White) |
| Text Body | `#475569` (Slate Gray) |
| Elements / Borders | `#E2E8F0` |
| Danger (Glaucoma) | `#E11D48` (Soft Crimson) |
| Success (Normal) | `#059669` (Emerald) |
| Warning | `#D97706` (Amber) |
| Primary Font | Inter (or Plus Jakarta Sans) |
| Corner Radius | 12px for cards, 8px for buttons |
| Card Shadow | `box-shadow: 0 10px 40px rgba(15, 23, 42, 0.04)` |
