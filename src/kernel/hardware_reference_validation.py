"""Shared field validation and reference interpolation for hardware loaders."""



def linear_interpolate(points, values, coordinate):
    if coordinate < points[0] or coordinate > points[-1]:
        raise ValueError(
            f"interpolation coordinate {coordinate} is outside the measured "
            f"range [{points[0]}, {points[-1]}]")
    upper = 1
    while upper < len(points) and coordinate > points[upper]:
        upper += 1
    if upper == len(points):
        upper = len(points) - 1
    lower = upper - 1
    span = points[upper] - points[lower]
    fraction = (coordinate - points[lower]) / span
    return values[lower] + fraction * (values[upper] - values[lower])


def require_exact_fields(section, values, expected):
    missing = expected - set(values)
    unknown = set(values) - expected
    if missing or unknown:
        raise ValueError(
            f"{section} schema mismatch: missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}")
