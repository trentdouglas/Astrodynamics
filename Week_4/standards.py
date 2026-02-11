from dataclasses import dataclass
import math
import numpy as np

@dataclass(frozen=True)
class Vector3:
    x: float
    y: float
    z: float

    # Vector addition
    def __add__(self, other: "Vector3") -> "Vector3":
        return Vector3(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z
        )

    # Vector subtraction
    def __sub__(self, other: "Vector3") -> "Vector3":
        return Vector3(
            self.x - other.x,
            self.y - other.y,
            self.z - other.z
        )

    # Scalar multiplication
    def __mul__(self, scalar: float) -> "Vector3":
        return Vector3(
            self.x * scalar,
            self.y * scalar,
            self.z * scalar
        )
    
    # Scalar division
    def __truediv__(self, scalar: float) -> "Vector3":
        return Vector3(
            self.x / scalar,
            self.y / scalar,
            self.z / scalar
        )

    __rmul__ = __mul__
    __rtruediv__ = __truediv__

    # Dot product
    def dot(self, other: "Vector3") -> float:
        return (
            self.x * other.x +
            self.y * other.y +
            self.z * other.z
        )

    # Cross product
    def cross(self, other: "Vector3") -> "Vector3":
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    # Magnitude (length)
    def magnitude(self) -> float:
        return math.sqrt(self.dot(self))

    # Unit vector
    def unit_vector(self) -> "Vector3":
        mag = self.magnitude()
        if mag == 0:
            raise ValueError("Cannot normalize a zero vector")
        return self * (1 / mag)

    def get_np_vector(self):
        return np.array([
            [self.x],
            [self.y],
            [self.z]
        ])


# Semi Major Axis (a)
def compute_sma(mu:float, energy:float):
    return -mu/(2*energy)

# Eccentricity (e)
def compute_ecc(energy: float, mu: float, h:float):
    return math.sqrt(1+(2*math.pow(h, 2)*energy)/(math.pow(mu, 2)))

# Inclination (i)
def compute_inc(Z_: Vector3, h_: Vector3):
    Z_ = Z_.unit_vector()
    h_ = h_.unit_vector()
    return math.acos(Z_.dot(h_))

# RAAN (capital omega) Ω
def compute_raan(Nx: float, Ny: float):
    return math.atan2(Ny, Nx)

# Argument of Perigeee (omega sub p) ωp
def compute_argp(h_: Vector3, N_: Vector3, B_: Vector3):
    h_ = h_.unit_vector()
    N_ = N_.unit_vector()
    B_ = B_.unit_vector()
    return math.atan2(h_.dot(N_.cross(B_)), N_.dot(B_))

# True Anomaly (nu) ν
def compute_ta(v_: Vector3, r_: Vector3, B_: Vector3):
    cosTa = r_.dot(B_)/(r_.magnitude()*B_.magnitude())
    ta = math.acos(cosTa)
    if r_.dot(v_) < 0:
        ta = 2*math.pi - ta
    return ta

# Period (TP)
def compute_period(a: float, mu: float):
    return 2*math.pi*math.sqrt(math.pow(a, 3)/mu)

# Apogee (r sub a) ra
def compute_apogee(a: float, e: float):
    return a*(1+e)

# Perigee (r sub p) rp
def compute_perigee(a: float, e: float):
    return a*(1-e)

# Energy (xi) ξ
def compute_energy(v: float, r: float, mu: float):
    return math.pow(v, 2)/2 - mu/r

# Angular Momentum Vector (h_)
def compute_angular_momentum_vector(r: Vector3, v: Vector3):
    return r.cross(v)

# RAAN Vector (N_) Unitized
def compute_raan_vector(Z_: Vector3, h_: Vector3):
    Z_ = Z_.unit_vector()
    h_ = h_.unit_vector()
    return (Z_.cross(h_)/(Z_.cross(h_).magnitude())).unit_vector()

# Perigee Vector (B_) Unitized
def compute_perigee_vector(mu: float, r_: Vector3, v_: Vector3, h_: Vector3):
    return v_.cross(h_)-mu*(r_/r_.magnitude()).unit_vector()

class KeplerianElements:
    """
    Computes classical Keplerian orbital elements from
    position and velocity state vectors.
    """

    mu_earth = 3.986004418e14  # m^3 / s^2

    def __init__(self, position: Vector3, velocity: Vector3):
        """
        Parameters
        ----------
        position : Vector3
            Position vector (meters)
        velocity : Vector3
            Velocity vector (m/s)
        """

        mu = self.mu_earth
        if position is None or velocity is None:
            # Initialize attributes to None (or sensible defaults)
            self.xi = None
            self.a = None
            self.h_ = None
            self.ecc = None
            self.inc = None
            self.inc_deg = None
            self.N_Unit = None
            self.raan = None
            self.raan_deg = None
            self.B_Unit = None
            self.argp = None
            self.argp_deg = None
            self.ta = None
            self.ta_deg = None
            self.period = None
            self.apogee = None
            self.perigee = None
            self.mu = 3.986004418e14  # m^3 / s^2
            return
        
        k_hat = Vector3(0, 0, 1)

        # Specific orbital energy
        self.xi = compute_energy(velocity.magnitude(), position.magnitude(), mu)

        # Semi-major axis
        self.a = compute_sma(mu, self.xi)

        # Angular momentum vector
        self.h_ = compute_angular_momentum_vector(position, velocity)

        # Eccentricity
        self.ecc = compute_ecc(self.xi, mu, self.h_.magnitude())

        # Inclination
        self.inc = compute_inc(k_hat, self.h_)
        self.inc_deg = math.degrees(self.inc)

        # RAAN
        self.N_Unit = compute_raan_vector(k_hat, self.h_)
        self.raan = compute_raan(self.N_Unit.x, self.N_Unit.y)
        self.raan_deg = math.degrees(self.raan)

        # Argument of perigee
        self.B_Unit = compute_perigee_vector(
            mu,
            position,
            velocity,
            self.h_
        )
        self.argp = compute_argp(self.h_, self.N_Unit, self.B_Unit)
        self.argp_deg = math.degrees(self.argp)

        # True anomaly
        self.ta = compute_ta(velocity, position, self.B_Unit)
        self.ta_deg = math.degrees(self.ta)

        # Orbital period (elliptical only)
        self.period = compute_period(self.a, mu)

        # Apoapsis / periapsis
        self.apogee = compute_apogee(self.a, self.ecc)
        self.perigee = compute_perigee(self.a, self.ecc)

def compute_eccentric_anomaly(nu: float, e: float) -> float:
    return math.asin((math.sin(nu)*math.sqrt(1 - math.pow(e, 2))) / (1 + e*math.cos(nu)))

def compute_mean_anomaly(E: float, e: float) -> float:
    return E - e*math.sin(E)

def compute_mean_motion(mu: float, a: float):
    # note a is semi major axis in meters
    return math.sqrt(mu/math.pow(a,3))

def compute_time_delta_after_angle(rad_angle: float, kepler_elements: KeplerianElements) -> tuple[float, float]:
    n = compute_mean_motion(kepler_elements.mu_earth, kepler_elements.a)
    E_SV = compute_eccentric_anomaly(kepler_elements.ta, kepler_elements.ecc)
    E_angle = compute_eccentric_anomaly(rad_angle, kepler_elements.ecc)
    t_delta_nu_to_angle = (1/n)*(E_angle-kepler_elements.ecc*math.sin(E_angle)) - (1/n)*(E_SV-kepler_elements.ecc*math.sin(E_SV))
    #if t is negative, then we add the orbital period to it to get the next time it will hit the angle
    # if t_delta_nu_to_angle < 0:
    #     t_delta_nu_to_angle += 2*math.pi/n  # orbital period
    return E_angle, t_delta_nu_to_angle

def compute_propagate_nu_given_delta_t(kepler_elements: KeplerianElements, time_delta: float) -> tuple[float, int, float]:
    E0 = compute_eccentric_anomaly(kepler_elements.ta, kepler_elements.ecc)
    M0 = compute_mean_anomaly(E0, kepler_elements.ecc)
    n = compute_mean_motion(kepler_elements.mu_earth, kepler_elements.a)
    M_propagated = M0 + n*time_delta
    E_k = M_propagated
    E_k_1 = 100000000000 # just to make sure our diff is large enough for first iteration
    while (abs(E_k-E_k_1) != 0):
        E_k = E_k_1
        E_k_1=E_k + (M_propagated-(E_k-kepler_elements.ecc*math.sin(E_k)))/(1-kepler_elements.ecc*math.cos(E_k))
    E_propagated = E_k_1%(2*math.pi)
    k = math.floor((E_k_1-E0)/(2*math.pi))
    cos_nu = (kepler_elements.ecc-math.cos(E_propagated))/(kepler_elements.ecc*math.cos(E_propagated)-1)
    sin_nu = (math.sin(E_propagated)*math.sqrt(1-math.pow(kepler_elements.ecc, 2)))/(1-kepler_elements.ecc*math.cos(E_propagated))
    nu_propagated = math.atan2(sin_nu, cos_nu)
    # ensure we land in the correct quadrant:
    nu_propagated = math.atan2(math.sin(nu_propagated), math.cos(nu_propagated))
    if nu_propagated < 0:
        nu_propagated = nu_propagated + 2*math.pi
    return E_propagated, k, nu_propagated

def compute_perifocal_coordinates(kepler_elements: KeplerianElements) -> tuple[Vector3, Vector3]:
    p = kepler_elements.a*(1-math.pow(kepler_elements.ecc,2))
    r = p*(1/(1+kepler_elements.ecc*math.cos(kepler_elements.ta)))
    perifocal_pos_x = r*math.cos(kepler_elements.ta)
    perifocal_pos_y = r*math.sin(kepler_elements.ta)
    perifocal_vel_x = -math.sqrt(kepler_elements.mu_earth/p)*math.sin(kepler_elements.ta)
    perifocal_vel_y = math.sqrt(kepler_elements.mu_earth/p)*(kepler_elements.ecc + math.cos(kepler_elements.ta))
    perifocal_pos = Vector3(perifocal_pos_x, perifocal_pos_y, 0)
    perifocal_vel = Vector3(perifocal_vel_x, perifocal_vel_y, 0)
    return perifocal_pos, perifocal_vel

def compute_f_g_f_dot_g_dot(pos_0, vel_0, pos, vel:Vector3) -> tuple[float, float, float, float]:
    h = compute_angular_momentum_vector(pos_0, vel_0)
    f = (pos.x*vel_0.y - vel_0.x*pos.y)/h.magnitude()
    g = (pos_0.x*pos.y - pos.x*pos_0.y)/h.magnitude()
    f_dot = (vel.x*vel_0.y - vel_0.x*vel.y)/h.magnitude()
    g_dot = (pos_0.x*vel.y - vel.x*pos_0.y)/h.magnitude()
    return f, g, f_dot, g_dot

def compute_matrix_eci_to_uvw(pos_, vel_: Vector3):
    U_hat = pos_/abs(pos_.magnitude())
    W_hat = (pos_.cross(vel_))/abs((pos_.cross(vel_)).magnitude())
    V_hat = W_hat.cross(U_hat)
    # rotation matrix:
    R = np.array([
        [U_hat.x, U_hat.y, U_hat.z],
        [V_hat.x, V_hat.y, V_hat.z],
        [W_hat.x, W_hat.y, W_hat.z]
    ])
    # note - to translate backwards from uvw to eci use the inverse of this matrix
    return R

def compute_f_g_f_dot_g_dot_no_final_perifocal(ecc, original_ta, new_ta, a, mu_earth, delta_E, delta_t: float) -> tuple[float, float, float, float]:
    p = a*(1-math.pow(ecc,2))
    r_0 = p*(1/(1+ecc*math.cos(original_ta)))
    r = p*(1/(1+ecc*math.cos(new_ta)))
    f = 1-(a/r_0)*(1-math.cos(delta_E))
    g = (delta_t)-math.sqrt((a**3)/mu_earth)*(delta_E-math.sin(delta_E))
    f_dot = (-math.sin(delta_E)*math.sqrt(mu_earth*a))/(r_0*r)
    g_dot = 1-(a/r)*(1-math.cos(delta_E))
    return f, g, f_dot, g_dot

def compute_perifocal_via_f_and_g(pos_0: Vector3, vel_0: Vector3, f: float, g: float, f_dot: float, g_dot: float) -> tuple[Vector3, Vector3]:
    pos_ = f*pos_0 + g*vel_0
    vel_ = f_dot*pos_0 + g_dot*vel_0
    return pos_, vel_

def compute_rotaton_perifocal_to_eci(raan: float, i: float, argp: float, perifocal_pos: Vector3, perifocal_vel: Vector3) -> tuple[np.array, Vector3, Vector3]:
    # rotation matrix:
    R = np.array([
        [math.cos(raan)*math.cos(argp)-math.sin(raan)*math.sin(argp)*math.cos(i), -math.cos(raan)*math.sin(argp)-math.sin(raan)*math.cos(argp)*math.cos(i), math.sin(raan)*math.sin(i)],
        [math.sin(raan)*math.cos(argp)+math.cos(raan)*math.sin(argp)*math.cos(i), -math.sin(raan)*math.sin(argp)+math.cos(raan)*math.cos(argp)*math.cos(i), -math.cos(raan)*math.sin(i)],
        [math.sin(i)*math.sin(argp), math.sin(i)*math.cos(argp), math.cos(i)]
    ])
    pos_np = ([
        [perifocal_pos.x],
        [perifocal_pos.y],
        [perifocal_pos.z]
    ])
    vel_np = ([
        [perifocal_vel.x],
        [perifocal_vel.y],
        [perifocal_vel.z]
    ])
    eci_pos_np = R @ pos_np
    eci_vel_np = R @ vel_np
    return R, Vector3(float(eci_pos_np[0][0]), float(eci_pos_np[1][0]), float(eci_pos_np[2][0])) ,Vector3(float(eci_vel_np[0][0]), float(eci_vel_np[1][0]), float(eci_vel_np[2][0]))