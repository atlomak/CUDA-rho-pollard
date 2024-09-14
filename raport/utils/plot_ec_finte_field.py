from sage.all import *

G = GF(2**11-9)
Ef = EllipticCurve(G, [-4,2])

pl = plot(Ef, rgbcolor=(0,0,0))
pl.save('ec_2_11-9.png')