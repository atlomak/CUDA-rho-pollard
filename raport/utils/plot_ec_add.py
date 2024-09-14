from sage.all import *

E = EllipticCurve("389a1")
P = E(-1, 1)
Q = E(0, -1)
R = P + Q
R_inv = -R

# Plot elliptic curve
plot_curve = plot(E, rgbcolor="black", xmax=5)

# Plot points
plot_p = plot(P, marker="o", rgbcolor="red", size=50)
p_label = text(
    "P",
    P.dehomogenize(2),
    horizontal_alignment="right",
    vertical_alignment="bottom",
    color="black",
)
plot_q = plot(Q, marker="o", rgbcolor="red", size=50)
q_label = text(
    " Q",
    Q.dehomogenize(2),
    horizontal_alignment="left",
    vertical_alignment="top",
    color="black",
)
plot_r = plot(R, marker="o", rgbcolor="red", size=50)
r_label = text(
    "R",
    R.dehomogenize(2),
    horizontal_alignment="right",
    vertical_alignment="bottom",
    color="black",
)
plot_r_inv = plot(R_inv, marker="o", rgbcolor="red", size=50)
r_inv_label = text(
    "-R",
    R_inv.dehomogenize(2),
    horizontal_alignment="right",
    vertical_alignment="top",
    color="black",
)

p6 = line2d([P.dehomogenize(2), R_inv.dehomogenize(2)], linestyle="--", rgbcolor="blue")
p7 = line2d([R.dehomogenize(2), R_inv.dehomogenize(2)], linestyle="--", rgbcolor="blue")

final_plot = plot_curve + plot_p + plot_q + plot_r + plot_r_inv + p6 + p7 + p_label + q_label + r_label + r_inv_label

final_plot.save("ec_add.png")
