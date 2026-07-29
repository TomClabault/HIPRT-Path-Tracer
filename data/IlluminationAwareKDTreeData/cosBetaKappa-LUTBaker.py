import math

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import i0e

# tau in the paper
ANGLE_THRESHOLD = math.radians(3.0)
ALPHA = 1.0e-4
LUT_SIZE = 256


def probability_inside_threshold_cone(
    kappa: float,
    observed_angle: float,
) -> float:
    """
    One direction is fixed on the +Z axis.

    The other is vMF-distributed with:
      concentration = kappa
      mean direction separated from +Z by observed_angle

    Returns:
        P(angle(random_direction, +Z) <= ANGLE_THRESHOLD)

    We want this probability to equal ALPHA, because:

        P(angle > ANGLE_THRESHOLD) = 1 - ALPHA
    """

    if kappa < 1.0e-8:
        # Uniform spherical distribution.
        return 0.5 * (1.0 - math.cos(ANGLE_THRESHOLD))

    cos_beta = math.cos(observed_angle)
    sin_beta = math.sin(observed_angle)

    # 1 - exp(-2*kappa), evaluated accurately near zero.
    denominator = -math.expm1(-2.0 * kappa)

    def integrand(theta: float) -> float:
        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)

        x = kappa * sin_beta * sin_theta

        # i0e(x) = exp(-abs(x)) * I0(x).
        #
        # This form avoids overflowing while evaluating the
        # modified Bessel function I0.
        exponent = (
            kappa * (cos_beta * cos_theta - 1.0)
            + x
        )

        return (
            math.exp(exponent)
            * i0e(x)
            * sin_theta
        )

    integral, _ = quad(
        integrand,
        0.0,
        ANGLE_THRESHOLD,
        epsabs=1.0e-12,
        epsrel=1.0e-10,
        limit=100,
    )

    return kappa * integral / denominator


def minimum_observed_angle(kappa: float) -> float | None:
    """
    Find beta such that:

        P(actual angle > ANGLE_THRESHOLD) = 1 - ALPHA

    Equivalently:

        P(actual angle <= ANGLE_THRESHOLD) = ALPHA

    Returns None when even beta = pi is not sufficient.
    """

    probability_at_zero = probability_inside_threshold_cone(
        kappa,
        0.0,
    )

    probability_at_pi = probability_inside_threshold_cone(
        kappa,
        math.pi,
    )

    # Even identical observed directions would be enough.
    # This can happen for unusual threshold/alpha combinations.
    if probability_at_zero <= ALPHA:
        return 0.0

    # Even maximally opposite observed directions do not provide
    # enough confidence, so no split is possible.
    if probability_at_pi > ALPHA:
        return None

    return brentq(
        lambda beta:
            probability_inside_threshold_cone(kappa, beta) - ALPHA,
        0.0,
        math.pi,
        xtol=1.0e-11,
    )


def find_minimum_usable_kappa() -> float:
    """
    Find the smallest kappa for which some observed angle can trigger
    a split. At this boundary, beta = pi.
    """

    def endpoint_equation(log_kappa: float) -> float:
        kappa = math.exp(log_kappa)

        return (
            probability_inside_threshold_cone(kappa, math.pi)
            - ALPHA
        )

    log_low = math.log(1.0e-8)
    log_high = math.log(1.0)

    while endpoint_equation(log_high) > 0.0:
        log_high += math.log(2.0)

    return math.exp(
        brentq(endpoint_equation, log_low, log_high)
    )


def generate_lut() -> tuple[float, np.ndarray]:
    minimum_kappa = find_minimum_usable_kappa()

    # u = sqrt(1 / kappa)
    maximum_u = math.sqrt(1.0 / minimum_kappa)

    u_values = np.linspace(0.0, maximum_u, LUT_SIZE)
    cos_beta_values = np.empty(LUT_SIZE)

    # kappa -> infinity means no uncertainty, so beta -> tau.
    cos_beta_values[0] = math.cos(ANGLE_THRESHOLD)

    for index in range(1, LUT_SIZE - 1):
        u = u_values[index]
        kappa = 1.0 / (u * u)

        beta = minimum_observed_angle(kappa)

        # This should normally not happen before the final entry.
        cos_beta_values[index] = (
            -1.0 if beta is None else math.cos(beta)
        )

    # At u_max, beta = pi, so cos(beta) = -1.
    # The runtime condition dot < -1 can never pass.
    cos_beta_values[-1] = -1.0

    return maximum_u, cos_beta_values


def print_cpp_lut(maximum_u: float, values: np.ndarray) -> None:
    print(
        f"static constexpr float DIRECTION_LUT_MAX_U = "
        f"{maximum_u:.9g}f;"
    )

    print(
        f"static constexpr float DIRECTION_LUT[{LUT_SIZE}] ="
    )
    print("{")

    for start in range(0, LUT_SIZE, 8):
        row = values[start:start + 8]

        formatted = ", ".join(
            f"{value:.9g}f" for value in row
        )

        print(f"    {formatted},")

    print("};")


if __name__ == "__main__":
    maximum_u, lut = generate_lut()
    print_cpp_lut(maximum_u, lut)