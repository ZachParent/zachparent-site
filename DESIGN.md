---
name: Zach Parent Portfolio
description: A blueprint-polaroid portfolio for a software developer.
colors:
  ink-blue: "#071f3a"
  paper-cream: "#f8f3ea"
  night-blue: "#040e17"
  accent-blue: "#0a63e7"
  accent-blue-soft: "#7ba8f2"
  amber: "#a76d31"
  green: "#2d8d73"
  muted-blue: "#657890"
typography:
  display:
    fontFamily: "New York, NewYork, Bodoni 72, Didot, Iowan Old Style, Georgia, serif"
    fontSize: "clamp(3.6rem, 12vw, 8.25rem)"
    fontWeight: 500
    lineHeight: 0.92
  body:
    fontFamily: "Avenir, Avenir Next, ui-sans-serif, system-ui, sans-serif"
    fontSize: "clamp(1rem, 2vw, 1.8rem)"
    fontWeight: 400
    lineHeight: 1.45
  label:
    fontFamily: "SF Mono, SFMono-Regular, ui-monospace, Menlo, Consolas, monospace"
    fontSize: "clamp(0.75rem, 1.6vw, 1.125rem)"
    fontWeight: 700
    letterSpacing: "0.18em"
rounded:
  sm: "7px"
  pill: "999px"
spacing:
  xs: "8px"
  sm: "14px"
  md: "24px"
  lg: "42px"
  xl: "72px"
components:
  nav-link:
    textColor: "{colors.ink-blue}"
    padding: "8px 0"
  work-pill:
    textColor: "{colors.accent-blue}"
    rounded: "{rounded.pill}"
    padding: "9px 21px"
---

# Design System: Zach Parent Portfolio

## 1. Overview

**Creative North Star: "The Developer Field Note"**

The system blends a real portrait, paper-like surfaces, visible blueprint measurement, and code notation. It should feel precise and personal: a resume distilled into a technical artifact rather than a marketing template.

The page rejects generic SaaS polish. No glass panels, no purple-blue AI gradients, no repeated icon-card grids, and no motion that gets in the way of reading.

**Key Characteristics:**
- Visible grid and measuring marks carry the visual voice.
- Serif display type supplies humanity; mono labels supply technical precision.
- Light and dark themes are peers, not an afterthought.
- Motion is brief, purposeful, and easy to disable.

## 2. Colors

The palette is restrained blueprint ink on warm paper in light mode, inverted to night-blue drafting paper in dark mode.

### Primary
- **Blueprint Ink** (#071f3a): Main display type, logo, and structural text in light mode.
- **Signal Blue** (#0a63e7): Navigation accents, measurement marks, and primary pills.

### Secondary
- **Drafting Amber** (#a76d31): Education/location chip accent.
- **Build Green** (#2d8d73): Project chip accent.

### Neutral
- **Paper Cream** (#f8f3ea): Light theme page field.
- **Night Blueprint** (#040e17): Dark theme page field.
- **Muted Draft Blue** (#657890): Secondary labels, line work, and supporting text.

### Named Rules

**The Rare Signal Rule.** Blue should mark structure and action, not flood the whole page.

## 3. Typography

**Display Font:** New York stack with Didot, Iowan Old Style, Georgia fallbacks
**Body Font:** Avenir stack with system sans fallbacks
**Label/Mono Font:** SF Mono stack with Menlo and Consolas fallbacks

**Character:** The pairing is classical but technical. The serif reads like a personal letterhead; the mono reads like a build log.

### Hierarchy
- **Display** (500, `clamp(3.6rem, 12vw, 8.25rem)`, 0.92): Hero name and work titles.
- **Headline** (700, `clamp(1rem, 2vw, 1.25rem)`, 1): Section labels and eyebrow text.
- **Body** (400, `clamp(1rem, 2vw, 1.8rem)`, 1.45): Intro and contact text. Keep paragraphs under 65ch.
- **Label** (700, `clamp(0.75rem, 1.6vw, 1.125rem)`, 0.18em tracking): Technical labels, action links, and code annotations.

### Named Rules

**The One Serif Moment Rule.** Use the display serif for identity and resume entries. Do not spread it across body copy.

## 4. Elevation

Depth is mostly tonal and structural. The polaroid is the only strongly lifted object because it represents the human anchor. Other sections use line work, alternating row tints, and surface opacity.

### Shadow Vocabulary
- **Polaroid Lift** (`0 24px 34px rgba(0, 0, 0, 0.42)`): Portrait only.
- **Interactive Lift** (`0 10px 24px rgba(13, 45, 77, 0.16)`): Hover feedback on small controls only.

### Named Rules

**The Portrait Owns The Shadow Rule.** Do not give every container a shadow. Let the page stay drafted and flat.

## 5. Components

### Buttons
- **Shape:** Icon-first or text-plus-icon, never heavy rectangles.
- **Primary:** Resume link uses mono caps, signal blue, and a moving arrow on hover.
- **Hover / Focus:** Use translate and color transitions only. Focus rings should be visible and blue.

### Chips
- **Style:** Outlined pill, currentColor border, transparent or lightly tinted fill.
- **State:** Hover may add a faint fill and a 1px lift. Do not change size.

### Cards / Containers
- **Corner Style:** Small radius at mobile sizes; squared work-list edges on desktop.
- **Background:** Transparent or lightly tinted with visible line work.
- **Shadow Strategy:** Avoid shadows except on the polaroid.
- **Internal Padding:** Use the spacing scale; keep related data tight.

### Navigation
- **Style:** Icon plus label, smaller on mobile. Desktop uses the sidebar; mobile uses the top rail.
- **States:** Hover shifts icons subtly and reveals blue emphasis. Theme toggle rotates the icon but keeps the hit target stable.

### Signature Component

**Polaroid Portrait:** Always square image crop with a larger lower paper margin. It may settle on page load and respond gently to pointer movement on capable devices.

## 6. Do's and Don'ts

### Do:
- **Do** preserve the blueprint grid and measurement marks as the visual system.
- **Do** keep the portrait square at all breakpoints.
- **Do** use short, eased motion under 800ms for first-load reveal and under 250ms for interaction feedback.
- **Do** alternate work row backgrounds subtly so the list scans cleanly.

### Don't:
- **Don't** use purple-blue AI gradients, glass panels, or generic SaaS card grids.
- **Don't** animate layout properties such as width, height, top, or left.
- **Don't** hide core resume details on mobile.
- **Don't** add long explanatory copy inside the page.
