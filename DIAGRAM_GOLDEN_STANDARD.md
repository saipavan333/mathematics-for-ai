# Diagram Golden Standard

Rules for every hand-authored SVG diagram in this course. A diagram is a
mathematical claim drawn in pixels — it must be **provably faithful to the math**,
not merely tidy. This document exists because early diagrams passed a "no clipping"
check yet were geometrically wrong (a tangent that didn't touch its curve, a
"convex" function drawn concave, Big-O curves in the wrong order). Those are
defects. The rules below make them catchable and preventable.

---

## 0. The one principle

**Fidelity before beauty.** If the picture depicts a relationship
(tangency, perpendicularity, an ordering, a shape class), that relationship must
*actually hold in the coordinates*. A pretty diagram that lies is worse than no
diagram.

---

## 1. Construction rules (how to draw math correctly)

1. **Compute curves from the formula; never eyeball a Bézier.**
   Freehand `Q`/`C` Béziers with guessed control points do **not** trace the
   intended function, so any point, tangent, or secant placed on them by eye will
   be wrong. Instead: pick an explicit `y = f(x)`, sample it at ~10–13 points, and
   draw a `<polyline>`. Then every other element is placed with the *same* formula.

2. **Place related elements from shared math, not by eye.**
   - A point `P` on a curve must use `P = (x0, f(x0))`.
   - A tangent at `P` must pass through `P` and use slope `f'(x0)`
     (draw it as `P ± t·(1, f'(x0))`).
   - A secant must connect `P` to another *on-curve* point `(x1, f(x1))`.
   - A projection's foot must be the actual perpendicular foot; draw the residual
     with a right-angle marker and verify `residual · subspace_dir = 0`.
   - Eigenvector / principal-axis arrows must be **collinear** with the object they
     claim (image `Av` on the same ray as `v`; ellipse axis along the covariance
     eigenvector).

3. **Verify numerically in the render script**, not just visually. Before shipping,
   print the checks: tangent slope == `f'(x0)`; tangent `y` at `x0` == `f(x0)`;
   curve ordering at the right edge; dot products that should be zero; etc.
   (See the 6.1 fix: the script printed `tangent slope -0.6`, `curve slope -0.6`,
   `tangent y at x0 = 75 = P.y` before it was accepted.)

4. **Prefer exact primitives.** Circles/ellipses/lines/polylines with computed
   coordinates over Béziers. If a Bézier is unavoidable, verify it passes through
   the required points.

---

## 2. Per-type geometric checklist

Before shipping a diagram of each kind, confirm:

- **Tangent / derivative:** the tangent line *touches the curve at exactly the
  marked point* and has that point's slope. Secants connect the point to other
  on-curve points and visibly approach the tangent.
- **Convex vs concave:** convex = a **bowl** (U); the chord between two points lies
  **above** the curve. Concave = a **hill** (∩); chord below. Draw the shape that
  matches the label.
- **Ordered curves (e.g. Big-O):** the asymptotic ordering must hold at the right
  edge — `O(1) < O(log n) < O(n) < O(n log n) < O(n²)`. The fastest curve is
  steepest and highest (let it exit the top if needed). Labels sit next to the
  curve they name, on the correct side.
- **Projection / orthogonality:** residual meets the subspace at a right angle
  (right-angle marker), and the foot is the true perpendicular foot.
- **Eigenvector / PCA / SVD axes:** the arrow is collinear with the object's axis;
  principal axes align with the data spread / ellipse tilt; axes are mutually
  perpendicular.
- **Gradient vs contours:** the gradient is perpendicular to the level set at the
  point and points toward increasing value.
- **Distributions:** a Gaussian is symmetric with the mean at the peak; the
  balance/fulcrum sits at the mean; Beta shapes lean the correct way
  (`Beta(a,b)` peaks near `(a−1)/(a+b−2)`); shaded areas match the stated interval.
- **Transform pictures (circle→ellipse, A=QR/LU):** the "after" object reflects the
  actual transform (long axis = largest singular value; triangular factors shaded
  on the correct side).

---

## 3. Label & layout rules (the earlier "no clipping/overlap" rules — still apply)

- **Nothing clipped:** every text label's `x + estimated_width` must be `≤ viewBox
  width` (estimate ~0.55·fontSize per character; be generous). Widen the `viewBox`
  or shorten the label if it overflows.
- **No text over lines/curves:** labels go in empty space, not on top of an arrow,
  curve, or another label. Check each label against every stroked element.
- **No label–label overlap:** stack labels with ≥ (fontSize + 4)px vertical gap or
  separate them horizontally.
- **Readable contrast & size:** ≥ 9px font; label color matches the element it
  names.

---

## 4. Verification protocol (mandatory before "done")

1. Render **every** SVG to PNG (cairosvg or a headless browser) at ~2× scale.
2. **Look at each one** and run it against the §2 checklist for its type — this is a
   geometric review, not a glance for clipping.
3. For any diagram asserting a numeric relationship, print the relationship in the
   render script and confirm it holds.
4. Re-render after each fix and re-inspect. A diagram is done only when it renders
   with no clipping/overlap **and** passes its geometric checklist.

> Rule of thumb: if you cannot state, in one sentence, the mathematical fact the
> diagram proves and point to the coordinates that make it true, it is not ready.

---

## 5. Failure log (so these don't recur)

| Diagram | Defect | Root cause | Fix |
|---|---|---|---|
| 6.1 tangent/secant | Tangent didn't touch the curve at P; secants didn't start at P | Curve was a guessed Bézier; P and tangent placed by eye | Polyline from `f`, P=`(x0,f(x0))`, tangent slope=`f'(x0)`, secants to on-curve points; numerically verified |
| 8.2 convex | "Convex" drawn as a concave hill, chord below | Confused screen-y orientation; drew ∩ instead of U | Redrew as a bowl (U) with the chord above |
| 1.4 Big-O | O(n²) not steepest; ended below O(n log n); label floating | Curve endpoints not ordered at the right edge | Reordered so O(n²) is steepest and exits the top; labels beside curve ends |
| 6.5 Taylor | 2nd-order parabola didn't pass through `a`; tangent slope wrong | Béziers not tied to the function at `a` | Cosine curve as polyline; tangent through `a` with `f'(a)`; osculating parabola through `a` |
| 10.4 / 10.3 | Labels drawn over an arrow / over each other | Placed without an overlap check | Repositioned to clear space |

**Meta-lesson:** the missing instruction was "**verify geometry, not just layout**."
A clipping check is necessary but not sufficient. Every diagram now also passes a
type-specific geometric checklist and, where applicable, a printed numeric check.
