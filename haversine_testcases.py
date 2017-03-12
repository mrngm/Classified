from haversine import haversine

def print_testcases():
    ## testcase: 1 degree difference in latitude
    print haversine((1.0, 50.0), (2.0, 50.0))
    print haversine((10.0, 50.0), (11.0, 50.0))
    print haversine((110.0, 50.0), (111.0, 50.0))
    print haversine((-1.0, 50.0), (-2.0, 50.0))
    print haversine((-10.0, 50.0), (-11.0, 50.0))
    print haversine((-110.0, 50.0), (-111.0, 50.0))
    # should all print 111.19 (km)

    ## testcase: 1 degree difference in longitude
    print haversine((1.0, 0.0), (1.0, 1.0))
    print haversine((1.0, 45.0), (1.0, 46.0))
    print haversine((1.0, 90.0), (1.0, 89.0)) # fooled you!
    print haversine((-1.0, 0.0), (-1.0, -1.0))
    print haversine((-1.0, -45.0), (-1.0, -46.0))
    print haversine((-1.0, -90.0), (-1.0, -89.0))
    # should all print 111.18 (km)

    # some random tests
    print haversine((1.0, 0.0), (2.0, 1.0))
    print haversine((1.0, 0.0), (-1.0, 2.0))
    # opposite points, about 1/2 circumference of the earth
    print haversine((91.0, 45.0), (-91.0, -45.0))
    print haversine((-90.0, 0.0), (90.0, 0.0)) # 1/2 circumference of the earth

    print haversine((0.0, 0.0), (0.1, 0.0)) # 11.12 km
    print haversine((0.00, 0.0), (0.01, 0.0)) # 1.11 km
    print haversine((0.000, 0.0), (0.001, 0.00)) # 0.11 km
    print haversine((0.0000, 0.0), (0.0001, 0.00)) # 0.01 km
    print haversine((0.00000, 0.0), (0.00001, 0.00)) # 0.0 km

    ## testcases: no float input
    try:
        print haversine(("1.0", 0.0), (1.0, 1.0))
    except ValueError as e:
        print e
        pass

    try:
        print haversine((1.0, "0.0"), (1.0, 1.0))
    except ValueError as e:
        print e
        pass
    try:
        print haversine((1.0, 0.0), ("1.0", 1.0))
    except ValueError as e:
        print e
        pass
    try:
        print haversine((1.0, 0.0), (1.0, "1.0"))
    except ValueError as e:
        print e
        pass

print_testcases()
