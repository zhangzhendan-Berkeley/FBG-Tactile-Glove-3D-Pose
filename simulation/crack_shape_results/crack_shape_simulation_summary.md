# Crack-shape ray-tracing simulation summary

Model: strain-aware 2D Monte Carlo geometric ray tracing in a 3 mm x 30 mm TPU slab waveguide.
The MCP slit layout is used: five 2 mm-deep features with 4 mm spacing.
The equivalent outer tensile strain is eps = t*theta/(2L); at 90 deg it is 7.85%.
A ray is counted as attenuated according to its intersections with air crack/notch regions.
Closed micro-slits have near-zero initial opening and nonlinear strain-dependent opening.
Material-removal notches have finite initial voids but still show bending-dependent loss.

| Shape | Loss at 0 deg (dB) | Loss at 90 deg (dB) | Dynamic change (dB) | Opening at 90 deg (mm) |
|---|---:|---:|---:|---:|
| Closed micro-slit | 0.076 | 4.828 | 4.752 | 0.399 |
| Rectangular removed notch | 3.798 | 6.139 | 2.341 | 0.538 |
| V-shaped removed notch | 4.786 | 6.798 | 2.012 | 0.847 |

Interpretation: the closed micro-slit preserves the total-internal-reflection path in the straight state.
Its loss-angle curve is monotonic but not perfectly linear because crack-mouth opening, ray escape saturation,
and curvature-induced loss are all nonlinear. Rectangular and V-shaped notches already interrupt the waveguide
path before bending, yielding high initial loss; bending still changes their air gaps and loss, but the usable
dynamic range is smaller than that of the closed micro-slit.