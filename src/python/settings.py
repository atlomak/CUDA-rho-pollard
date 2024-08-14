from sage.all import EllipticCurve, GF

# ECC79p settings

field_order = 0x62CE5177412ACA899CF5
r = 0x1CE4AF36EED8DE22B99D

curve_a = 0x39C95E6DDDB1BC45733C
curve_b = 0x1F16D880E89D5A1C0ED1

curve_order = 0x62CE5177407B7258DC31

P_x = 0x315D4B201C208475057D
P_y = 0x035F3DF5AB370252450A

Q_x = 0x0679834CEFB7215DC365
Q_y = 0x4084BC50388C4E6FDFAB

F = GF(field_order)
E = EllipticCurve(F, [curve_a, curve_b])

P = E(P_x, P_y)
Q = E(Q_x, Q_y)


# field_order = 0xD3915
# F = GF(field_order)
# curve_a = 738492
# curve_b = 694682
# r = 926251
# E = EllipticCurve(F, [curve_a, curve_b])
# curve_order = E.order()
# P = E(184224, 74658)
# Q = 42 * P
