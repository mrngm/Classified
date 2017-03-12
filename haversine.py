#!/usr/bin/env python2

# adapted from https://gist.github.com/rochacbruno/2883505
# uses the first formula from
# <https://en.wikipedia.org/wiki/Great-circle_distance#Computational_formulas>
# restrict types to floats, round distance to 2 decimals (i.e. accuracy of
# about 10 meter)

from math import sin, cos, radians, atan2, sqrt

def haversine(origin, destination):
    lat1, lon1 = origin
    lat2, lon2 = destination
    if type(lat1) != float or type(lon1) != float:
        raise ValueError("Origin does not contain floats")
    if type(lat2) != float or type(lon2) != float:
        raise ValueError("Destination does not contain floats")
    radius = 6371 # radius of the earth in km, WGS84 ellipsoid

    dlat = radians(lat2-lat1)
    dlon = radians(lon2-lon1)
    a = sin(dlat/2) * sin(dlat/2) + cos(radians(lat1)) \
        * cos(radians(lat2)) * sin(dlon/2) * sin(dlon/2)
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    d = round(radius * c, 2)

    return d
