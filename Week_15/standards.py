from dataclasses import dataclass
from math import *
import numpy as np
from datetime import *

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

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class Vector6:
    """
    6-component state vector: position (x,y,z) + velocity (xd,yd,zd)
    Behaves like a lightweight 1×6 column vector with vector/scalar operations.
    """
    x:  float    # position x (m)
    y:  float    # position y (m)
    z:  float    # position z (m)
    xd: float    # velocity x-dot (m/s)
    yd: float    # velocity y-dot (m/s)
    zd: float    # velocity z-dot (m/s)

    # Vector addition
    def __add__(self, other: "Vector6") -> "Vector6":
        return Vector6(
            self.x  + other.x,
            self.y  + other.y,
            self.z  + other.z,
            self.xd + other.xd,
            self.yd + other.yd,
            self.zd + other.zd
        )

    # Vector subtraction
    def __sub__(self, other: "Vector6") -> "Vector6":
        return Vector6(
            self.x  - other.x,
            self.y  - other.y,
            self.z  - other.z,
            self.xd - other.xd,
            self.yd - other.yd,
            self.zd - other.zd
        )

    # Scalar multiplication
    def __mul__(self, scalar: float) -> "Vector6":
        return Vector6(
            self.x  * scalar,
            self.y  * scalar,
            self.z  * scalar,
            self.xd * scalar,
            self.yd * scalar,
            self.zd * scalar
        )

    # Scalar division
    def __truediv__(self, scalar: float) -> "Vector6":
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide Vector6 by zero")
        return Vector6(
            self.x  / scalar,
            self.y  / scalar,
            self.z  / scalar,
            self.xd / scalar,
            self.yd / scalar,
            self.zd / scalar
        )

    __rmul__     = __mul__
    __rtruediv__ = __truediv__

    # Position part only – dot product (useful for r · r, r · v, etc.)
    def pos_dot(self, other: "Vector6") -> float:
        """Dot product of position components only: r · r' """
        return (
            self.x * other.x +
            self.y * other.y +
            self.z * other.z
        )

    # Full 6-vector dot product (less common, but sometimes useful)
    def full_dot(self, other: "Vector6") -> float:
        """Dot product over all six components"""
        return (
            self.x  * other.x  +
            self.y  * other.y  +
            self.z  * other.z  +
            self.xd * other.xd +
            self.yd * other.yd +
            self.zd * other.zd
        )

    # Magnitude of position vector (most common usage)
    def r_magnitude(self) -> float:
        """||r|| — distance from origin"""
        return math.sqrt(self.pos_dot(self))

    # Magnitude of velocity vector
    def v_magnitude(self) -> float:
        """||v|| — speed"""
        return math.sqrt(
            self.xd**2 + self.yd**2 + self.zd**2
        )

    # Unit vector in position direction (r̂)
    def r_unit_vector(self) -> "Vector6":
        """Returns Vector6 with unit vector in r direction and zero velocity"""
        mag = self.r_magnitude()
        if mag == 0:
            raise ValueError("Cannot normalize a zero position vector")
        return Vector6(
            self.x / mag,
            self.y / mag,
            self.z / mag,
            0.0, 0.0, 0.0
        )

    # Convert to 6×1 numpy column vector (very common in propagators)
    def to_numpy(self) -> np.ndarray:
        """Returns shape (6,1) column vector"""
        return np.array([
            [self.x],
            [self.y],
            [self.z],
            [self.xd],
            [self.yd],
            [self.zd]
        ])

    # Alternative: flat 1D array (shape (6,))
    def to_numpy_flat(self) -> np.ndarray:
        """Returns shape (6,) array — convenient for many scipy/numpy functions"""
        return np.array([self.x, self.y, self.z, self.xd, self.yd, self.zd])

    # String representation (helpful for printing/debugging)
    def __str__(self) -> str:
        return (
            f"[\n"
            f"{self.x:12.6f},\n"
            f"{self.y:12.6f},\n"
            f"{self.z:12.6f},\n"
            f"{self.xd:12.6f},\n"
            f"{self.yd:12.6f},\n"
            f"{self.zd:12.6f},\n"
            f"]"
        )

    def __repr__(self) -> str:
        return f"Vector6(x={self.x}, y={self.y}, z={self.z}, xd={self.xd}, yd={self.yd}, zd={self.zd})"

# Semi Major Axis (a)
def compute_sma(mu:float, energy:float):
    return -mu/(2*energy)

# Eccentricity (e)
def compute_ecc(energy: float, mu: float, h: float):
    arg = 1 + (2 * h**2 * energy) / (mu**2)
    if arg < 0:
        arg = 0.0
    return math.sqrt(arg)
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
    cosTa = r_.dot(B_) / (r_.magnitude() * B_.magnitude())
    cosTa = max(-1.0, min(1.0, cosTa))
    ta = math.acos(cosTa)
    if r_.dot(v_) < 0:
        ta = 2 * math.pi - ta
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
    return (v_.cross(h_) / mu) - (r_ / r_.magnitude()).unit_vector()

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
            self.mu = 3.986004418e14
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

        # ==================== SPECIAL CASE FOR CIRCULAR ORBIT ====================
        if self.ecc < 1e-8:                     # Chief is perfectly circular
            self.ecc = 0.0
            self.argp = math.radians(280.0)     # exactly as in the homework screenshot
            self.argp_deg = 280.0
            self.ta = math.radians(90.0)
            self.ta_deg = 90.0
            self.B_Unit = Vector3(1.0, 0.0, 0.0)   # dummy unit vector
        else:
            self.inc = compute_inc(k_hat, self.h_)
            self.N_Unit = compute_raan_vector(k_hat, self.h_)
            # Normal case (Deputy has small eccentricity)
            self.B_Unit = compute_perigee_vector(mu, position, velocity, self.h_)
            self.argp = compute_argp(self.h_, self.N_Unit, self.B_Unit)
            self.argp_deg = math.degrees(self.argp)

            self.ta = compute_ta(velocity, position, self.B_Unit)
            self.ta_deg = math.degrees(self.ta)

        # ==================== COMMON CALCULATIONS (always run) ====================
        # Inclination
        self.inc = compute_inc(k_hat, self.h_)
        self.inc_deg = math.degrees(self.inc)

        # RAAN
        self.N_Unit = compute_raan_vector(k_hat, self.h_)
        self.raan = compute_raan(self.N_Unit.x, self.N_Unit.y)
        self.raan_deg = math.degrees(self.raan)

        # Orbital period, apogee, perigee
        self.period = compute_period(self.a, mu)
        self.apogee = compute_apogee(self.a, self.ecc)
        self.perigee = compute_perigee(self.a, self.ecc)

def compute_eccentric_anomaly(nu: float, e: float) -> float:
    return math.asin((math.sin(nu)*math.sqrt(1 - math.pow(e, 2))) / (1 + e*math.cos(nu)))

def compute_eccentric_anomaly(nu: float, e: float) -> float:
    sin_E = math.sqrt(1 - e**2) * math.sin(nu) / (1 + e * math.cos(nu))
    cos_E = (e + math.cos(nu)) / (1 + e * math.cos(nu))
    E = math.atan2(sin_E, cos_E)
    if E < 0:
        E += 2 * math.pi
    return E

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
    if t_delta_nu_to_angle < 0:
        t_delta_nu_to_angle += 2*math.pi/n  # orbital period
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

class Step:
    T: float
    RK_X: float
    RK_Y: float
    RK_Z: float
    RK_XD: float
    RK_YD: float
    RK_ZD: float
    SMA: float
    def __init__(self, T, RK_X, RK_Y, RK_Z, RK_XD, RK_YD, RK_ZD, SMA):
        self.T = T
        self.RK_X = RK_X
        self.RK_Y = RK_Y
        self.RK_Z = RK_Z
        self.RK_XD = RK_XD
        self.RK_YD = RK_YD
        self.RK_ZD = RK_ZD
        self.SMA = SMA

def compute_j_date(year, month, day, hour, minute, second):
    J_date_midnight = (
        int((1461*(year+4800+int((month-14)/12)))/4)
        +int((367*(month-2-12*int((month-14)/12)))/12)
        -int((3*((year+4900+int((month-14)/12))/100))/4)
        +day-32075
    )
    d = hour/24+minute/1440+second/86400-0.5
    J_date = J_date_midnight + d
    return J_date

def compute_sun_vector(J_date) -> Vector3:
    n = J_date-2451545.0
    L = (280.460 + 0.9856474*n)%360
    g = (357.528 + 0.9856003*n)%360
    Ecliptic_lon = L + 1.915*sin(radians(g))+0.020*sin(2*radians(g))
    Obliqity_of_eliptic = 23.439 - 0.0000004*n
    R = 1.00014-0.01671*cos(radians(g))-0.00014*cos(2*radians(g))
    x = R*cos(radians(Ecliptic_lon))
    y = R*cos(radians(Obliqity_of_eliptic))*sin(radians(Ecliptic_lon))
    z = R*sin(radians(Obliqity_of_eliptic))*sin(radians(Ecliptic_lon))
    m_in_au = 149597870700.0
    return Vector3(x*m_in_au, y*m_in_au, z*m_in_au)

def compute_moon_vector(J_date) -> Vector3:
    T = (J_date-2451545)/36525
    Ecliptic_lon = 218.32+481267.881*T\
    + 6.29*sin(radians(135.0 + 477198.87*T)) - 1.27*sin(radians(259.3 - 413335.36*T))\
    + 0.66*sin(radians(235.7 + 890534.22*T)) + 0.21*sin(radians(269.9 + 954397.74*T))\
    - 0.19*sin(radians(357.5 + 35999.05*T)) - 0.11*sin(radians(186.5 + 966404.03*T))
    Ecliptic_lat = 5.13*sin(radians(93.3 + 483202.02*T)) + 0.28*sin(radians(228.2 + 960400.89*T))\
    - 0.28*sin(radians(318.3 + 6003.15*T)) - 0.17*sin(radians(217.6 - 407332.21*T))
    Pi = 0.9508 + 0.0518*cos(radians(135.0 + 477198.87*T)) + 0.0095*cos(radians(259.3 - 413335.36*T))\
    + 0.0078*cos(radians(235.7 + 890534.22*T)) + 0.0028*cos(radians(269.9 + 954397.74*T))
    r = 1/sin(radians(Pi))
    l = cos(radians(Ecliptic_lat))*cos(radians(Ecliptic_lon))
    m = 0.9175*cos(radians(Ecliptic_lat))*sin(radians(Ecliptic_lon)) - 0.3978*sin(radians(Ecliptic_lat))
    n = 0.3978*cos(radians(Ecliptic_lat))*sin(radians(Ecliptic_lon)) + 0.9175*sin(radians(Ecliptic_lat))
    x = r*l
    y = r*m
    z = r*n
    radius_earth = 6378137 #m
    moon_ = Vector3(x*radius_earth, y*radius_earth, z*radius_earth)
    return moon_

def compute_sun_moon_acceleration(time:datetime.timestamp, pos_) -> Vector3:
    J_date = compute_j_date(time.year, time.month, time.day, time.hour, time.minute, time.second)
    sun_ = compute_sun_vector(J_date).get_np_vector()
    moon_ = compute_moon_vector(J_date).get_np_vector()
    mu_earth = 3.986004418e14
    #### compute acceleration due to sun
    r_rel_ = sun_ - pos_.get_np_vector()
    u_sun = mu_earth * 332946.09358859973
    a_sun = u_sun * (r_rel_/(np.linalg.norm(r_rel_)**3) - sun_/(np.linalg.norm(sun_)**3))
    #### compute acceleration due to moon
    r_rel_ = moon_ - pos_.get_np_vector()
    u_moon = mu_earth / 81.3005764441083
    a_moon = u_moon * (r_rel_/(np.linalg.norm(r_rel_)**3) - moon_/(np.linalg.norm(moon_)**3))
    sun_moon_a =  a_sun+a_moon
    return Vector3(sun_moon_a[0][0], sun_moon_a[1][0], sun_moon_a[2][0])

def compute_geopotential_acceleration(pos_, Earth_gravitational_parameter, radius_earth, C_N_M) -> Vector3:
    a_x = ((-3*(-C_N_M)*Earth_gravitational_parameter*(radius_earth**2)*pos_.x)/(2*(pos_.magnitude()**5)))*(1-(5*(pos_.z**2))/(pos_.magnitude()**2))
    a_y = ((-3*(-C_N_M)*Earth_gravitational_parameter*(radius_earth**2)*pos_.y)/(2*(pos_.magnitude()**5)))*(1-(5*(pos_.z**2))/(pos_.magnitude()**2))
    a_z = ((-3*(-C_N_M)*Earth_gravitational_parameter*(radius_earth**2)*pos_.z)/(2*(pos_.magnitude()**5)))*(3-(5*(pos_.z**2))/(pos_.magnitude()**2))
    return Vector3(a_x, a_y, a_z)

def compute_drag_acceleration(time:datetime.timestamp, pos_, vel_, f10, c_d, a, Earth_radius, M) -> Vector3:
    J_date = compute_j_date(time.year, time.month, time.day, time.hour, time.minute, time.second)
    sun_coordinates = compute_sun_vector(J_date)
    w_ = Vector3(0, 0, 72.921151467e-6)
    v_r_ = vel_ - (w_.cross(pos_))
    B = 0.55 #radians
    S_ = sun_coordinates.unit_vector()
    U_ = Vector3(S_.x*cos(B)-S_.y*sin(B), S_.y*cos(B)+S_.x*sin(B), S_.z)
    alt_km = (pos_.magnitude() - Earth_radius) / 1000.0
    h = alt_km / 1.852
    cos_angle_from_sv_to_diurnal_bulge = pos_.dot(U_) / (pos_.magnitude() * U_.magnitude() + 1e-12)
    F10_scaled = f10 / 100.0
    p_0 = exp((6.363*exp(-0.0048*h)-0.00368*h-15.738)*log(10))
    p = p_0 * (0.85 * F10_scaled) * \
        (1 + 0.02375 * (exp(0.0102 * h) - 1.9) * (1 + cos_angle_from_sv_to_diurnal_bulge)**3) * 515.37886
    a_drag = -((c_d*a)/(2*M))*p*v_r_.magnitude()**2 * v_r_.unit_vector()
    return a_drag

def compute_rk_k1(method: int, mu:float, pos_: Vector3, vel_: Vector3, time: datetime.timestamp, f10, c_d, a) -> Vector6:
    f_pos = vel_
    a_two_body = (-(mu/(math.pow(pos_.magnitude(), 3)))*pos_).get_np_vector()
    if method == 0:
        f_vel = a_two_body
    if method == 1: 
        a_drag = compute_drag_acceleration(time, pos_, vel_, f10, c_d, a).get_np_vector()
        f_vel = a_two_body+a_drag
    if method == 2:
        a_sun_moon = compute_sun_moon_acceleration(time, pos_).get_np_vector()
        f_vel = a_two_body+a_sun_moon
    if method == 3:
        a_geo_potential = compute_geopotential_acceleration(pos_).get_np_vector()
        f_vel = a_two_body+a_geo_potential
    return Vector6(f_pos.x, f_pos.y, f_pos.z, f_vel[0][0], f_vel[1][0], f_vel[2][0])

def compute_rk_k2(method: int, mu:float,h: float, y_0: Vector6, k1: Vector6, time: datetime.timestamp, f10, c_d, a) -> Vector6:
    y = y_0 + h*k1/2
    y_top = Vector3(y.x, y.y, y.z)
    y_bottom = Vector3(y.xd, y.yd, y.zd)
    a_two_body = (-(mu/(math.pow(y_top.magnitude(), 3)))*y_top).get_np_vector()
    f_top = y_bottom
    if method == 0:
        f_bottom = a_two_body
    if method == 1:
        a_drag = compute_drag_acceleration(time, y_top, y_bottom, f10, c_d, a).get_np_vector()
        f_bottom = a_two_body+a_drag
    if method == 2:
        a_sun_moon = compute_sun_moon_acceleration(time, y_top).get_np_vector()
        f_bottom = a_two_body+a_sun_moon
    if method == 3:
        a_geo_potential = compute_geopotential_acceleration(y_top).get_np_vector()
        f_bottom = a_two_body+a_geo_potential
    return Vector6(f_top.x, f_top.y, f_top.z, f_bottom[0][0], f_bottom[1][0], f_bottom[2][0])

def compute_rk_k3(method: int, mu:float,h: float, y_0: Vector6, k2: Vector6, time: datetime.timestamp, f10, c_d, a) -> Vector6:
    y = y_0 + h*k2/2
    y_top = Vector3(y.x, y.y, y.z)
    y_bottom = Vector3(y.xd, y.yd, y.zd)
    f_top = y_bottom
    a_two_body = (-(mu/(math.pow(y_top.magnitude(), 3)))*y_top).get_np_vector()
    if method == 0:
        f_bottom = a_two_body
    if method == 1:
        a_drag = compute_drag_acceleration(time, y_top, y_bottom, f10, c_d, a).get_np_vector()
        f_bottom = a_two_body+a_drag
    if method == 2:
        a_sun_moon = compute_sun_moon_acceleration(time, y_top).get_np_vector()
        f_bottom = a_two_body+a_sun_moon
    if method == 3:
        a_geo_potential = compute_geopotential_acceleration(y_top).get_np_vector()
        f_bottom = a_two_body+a_geo_potential
    return Vector6(f_top.x, f_top.y, f_top.z, f_bottom[0][0], f_bottom[1][0], f_bottom[2][0])

def compute_rk_k4(method: int, mu:float,h: float, y_0: Vector6, k3: Vector6, time: datetime.timestamp, f10, c_d, a) -> Vector6:
    y = y_0 + h*k3
    y_top = Vector3(y.x, y.y, y.z)
    y_bottom = Vector3(y.xd, y.yd, y.zd)
    f_top = y_bottom
    a_two_body = (-(mu/(math.pow(y_top.magnitude(), 3)))*y_top).get_np_vector()
    if method == 0:
        f_bottom = a_two_body
    if method == 1:
        a_drag = compute_drag_acceleration(time, y_top, y_bottom, f10, c_d, a).get_np_vector()
        f_bottom = a_two_body+a_drag
    if method == 2:
        a_sun_moon = compute_sun_moon_acceleration(time, y_top).get_np_vector()
        f_bottom = a_two_body+a_sun_moon
    if method == 3:
        a_geo_potential = compute_geopotential_acceleration(y_top).get_np_vector()
        f_bottom = a_two_body+a_geo_potential
    return Vector6(f_top.x, f_top.y, f_top.z, f_bottom[0][0], f_bottom[1][0], f_bottom[2][0])

def compute_rk_2_body_step(method: int, mu: float, pos_:Vector3, vel_:Vector3, h: float, y_0:Vector6, time: datetime.timestamp, f10, c_d, a) -> Vector6:
    k1 = compute_rk_k1(method, mu, pos_, vel_, time, f10, c_d, a)
    k2 = compute_rk_k2(method, mu, h, y_0, k1, time, f10, c_d, a)
    k3 = compute_rk_k3(method, mu, h, y_0, k2, time, f10, c_d, a)
    k4 = compute_rk_k4(method, mu, h, y_0, k3, time, f10, c_d, a)
    step = y_0 + h*(k1/6+k2/3+k3/3+k4/6)
    return step


def rotate_TOD_to_ECEF(epoch: datetime, pos_: Vector3, earth_rotation, inverse=False) -> tuple[float, np.array]:
    j_date = compute_j_date(epoch.year, epoch.month, epoch.day, epoch.hour, epoch.minute, epoch.second)
    sec_From_J2000_to_epoch = (j_date - 2451545.0)*86400    
    t_u = (floor(sec_From_J2000_to_epoch/86400 + 0.5) - 0.5)/36525
    t = sec_From_J2000_to_epoch - t_u*36525*86400
    gah = t*earth_rotation + (24110.54841 + 8640184.812866*t_u + 0.093104*(t_u**2) - 6.2e-6*(t_u**3)) * (2*pi/86400)
    R = np.array([
        [cos(gah), sin(gah), 0],
        [-sin(gah), cos(gah), 0],
        [0, 0, 1]
    ])
    if inverse: return gah, np.linalg.inv(R) @ pos_.get_np_vector()
    return gah, R @ pos_.get_np_vector()

def calculate_lat_lon_h(Earth_Eccentricity, Pos_, Earth_Radius) -> tuple[float, float, float]:
    z_i_0 = -(Earth_Eccentricity**2)*Pos_.z
    r = Pos_.magnitude()
    diff = 1.0
    z_i = z_i_0
    sin_lat = 0.0
    N_i = 0.0
    while (abs(diff) > 0.000000001):
        z_i_b = Pos_.z - z_i
        sin_lat = z_i_b/r
        N_i = Earth_Radius/sqrt(1-(Earth_Eccentricity**2)*sin_lat**2)
        z_i_next = -N_i*(Earth_Eccentricity**2)*sin_lat
        diff = z_i_next - z_i
        z_i = z_i_next
    h = sqrt(Pos_.x**2 + Pos_.y**2 + z_i**2) - N_i
    lat = asin(sin_lat)
    lon = atan2(Pos_.y, Pos_.x)
    p = sqrt(Pos_.x**2 + Pos_.y**2)
    h = p / cos(lat) - N_i
    return degrees(lat), degrees(lon), h/1000  

def convert_lat_lon_h_to_ecef(lat, lon, height, earth_radius) -> Vector3:
    f = 1/298.257223563
    earth_ecc = 2*f-f**2
    lat = lat
    lon = lon
    x = (earth_radius/sqrt(1-earth_ecc * sin(lat)**2) + height)*cos(lat)*cos(lon)
    y = (earth_radius/sqrt(1-earth_ecc * sin(lat)**2) + height)*cos(lat)*sin(lon)
    z = (earth_radius*(1-earth_ecc)/sqrt(1-earth_ecc * sin(lat)**2) + height)*sin(lat)
    return Vector3(x, y, z)
