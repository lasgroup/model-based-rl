import jax
from brax.envs.base import PipelineEnv, State, Env
import chex
from flax import struct
import jax.numpy as jnp
from jaxtyping import Float, Array
from functools import partial

from mbrl.utils.tolerance_reward import ToleranceReward


@chex.dataclass
class GreenHouseParams:
    beta: chex.Array = struct.field(default_factory=lambda: jnp.array(0.01))  # Heat absorption efficiency
    gamma: chex.Array = struct.field(default_factory=lambda: jnp.array(0.067))  # apparent psychometric constant
    epsilon: chex.Array = struct.field(default_factory=lambda: jnp.array(3.0))  # Cover heat resistance ratio
    zeta: chex.Array = struct.field(default_factory=lambda: jnp.array(2.7060 * (10 ** (-5))))
    # ventilation rate parameter
    eta: chex.Array = struct.field(default_factory=lambda: jnp.array(0.7))  # radiation conversion factor
    theta: chex.Array = struct.field(default_factory=lambda: jnp.array(4.02 * (10 ** (-5))))  # ventilation rate
    # parameter 1
    kappa: chex.Array = struct.field(default_factory=lambda: jnp.array(5.03 * (10 ** (-5))))  # ventilation rate
    # parameter 2
    lam: chex.Array = struct.field(default_factory=lambda: jnp.array(0.46152))  # pressure constant
    mu: chex.Array = struct.field(default_factory=lambda: jnp.array(1.4667))  # Molar weight fraction C02 CH20
    nu_2: chex.Array = struct.field(
        default_factory=lambda: jnp.array(3.68 * (10 ** (-5))))  # ventilation rate parameter 1
    xi: chex.Array = struct.field(
        default_factory=lambda: jnp.array(6.3233 * (10 ** (-5))))  # ventilation rate parameter 2
    rho_w: chex.Array = struct.field(default_factory=lambda: jnp.array(998))  # Density of water
    rho_a: chex.Array = struct.field(default_factory=lambda: jnp.array(1.29))  # Density of air
    sigma: chex.Array = struct.field(
        default_factory=lambda: jnp.array(7.1708 * (10 ** (-5))))  # ventilation rate parameter 3
    tau: chex.Array = struct.field(default_factory=lambda: jnp.array(3.0))  # pipe heat transfer coefficient 1
    nu: chex.Array = struct.field(default_factory=lambda: jnp.array(0.74783))  # pipe heat transfer coefficient 2
    chi: chex.Array = struct.field(default_factory=lambda: jnp.array(0.0156))  # ventilation rate parameter 4
    psi: chex.Array = struct.field(default_factory=lambda: jnp.array(7.4 * (10 ** (-5))))
    omega: chex.Array = struct.field(default_factory=lambda: jnp.array(0.622))  # Humidity ratio
    a1: chex.Array = struct.field(default_factory=lambda: jnp.array(0.611))  # Saturated vapour pressure parameter 1
    a2: chex.Array = struct.field(default_factory=lambda: jnp.array(17.27))  # Saturated vapour pressure parameter 2
    a3: chex.Array = struct.field(default_factory=lambda: jnp.array(239.0))  # Saturated vapour pressure parameter 3
    ap: chex.Array = struct.field(default_factory=lambda: jnp.array(314.16))  # Heating pipe outer surface area
    b1: chex.Array = struct.field(default_factory=lambda: jnp.array(2.7))  # buffer_coefficient
    cg: chex.Array = struct.field(default_factory=lambda: jnp.array(32 * (10 ** 3)))  # green_house_heat_capacity
    cp_w: chex.Array = struct.field(default_factory=lambda: jnp.array(4180.0))  # specific_heat_water
    cs: chex.Array = struct.field(default_factory=lambda: jnp.array(120 * (10 ** 3)))  # green_house_soil_heat_capacity
    cp_a: chex.Array = struct.field(default_factory=lambda: jnp.array(1010))  # air_specific_heat_water
    d1: chex.Array = struct.field(default_factory=lambda: jnp.array(2.1332 * (10 ** (-7))))  # plant development rate 1
    d2: chex.Array = struct.field(default_factory=lambda: jnp.array(2.4664 * (10 ** (-1))))  # plant development rate 2
    d3: chex.Array = struct.field(default_factory=lambda: jnp.array(20))  # plant development rate 3
    d4: chex.Array = struct.field(default_factory=lambda: jnp.array(7.4966 * (10 ** (-11))))  # plant development rate 4
    f: chex.Array = struct.field(default_factory=lambda: jnp.array(1.2))  # fruit assimilate requirement
    f1: chex.Array = struct.field(default_factory=lambda: jnp.array(8.1019 * (10 ** (-7))))  # fruit growth rate
    f2: chex.Array = struct.field(default_factory=lambda: jnp.array(4.6296 * (10 ** (-6))))  # fruit growth rate
    g1: chex.Array = struct.field(
        default_factory=lambda: jnp.array(20.3 * (10 ** (-3))))  # Leaf conductance parameter 1
    g2: chex.Array = struct.field(default_factory=lambda: jnp.array(0.44))  # Leaf conductance parameter 2
    g3: chex.Array = struct.field(default_factory=lambda: jnp.array(2.5 * (10 ** (-3))))  # Leaf conductance parameter 3
    g4: chex.Array = struct.field(default_factory=lambda: jnp.array(3.1 * (10 ** (-4))))  # Leaf conductance parameter 4
    gb: chex.Array = struct.field(default_factory=lambda: jnp.array(10 ** (-2)))  # Boundary layer conductance
    kd: chex.Array = struct.field(default_factory=lambda: jnp.array(2.0))  # Soil to soil heat transfer coefficient
    kr: chex.Array = struct.field(default_factory=lambda: jnp.array(7.9))  # Roof heat transfer coefficient
    ks: chex.Array = struct.field(default_factory=lambda: jnp.array(5.75))  # Soil to air heat transfer coefficient
    l1: chex.Array = struct.field(
        default_factory=lambda: jnp.array(2.501 * (10 ** 6)))  # Vaporisation energy coefficient 1
    l2: chex.Array = struct.field(
        default_factory=lambda: jnp.array(2.381 * (10 ** 3)))  # Vaporisation energy coefficient 2
    m1: chex.Array = struct.field(default_factory=lambda: jnp.array(1.0183 * (10 ** (-3))))  # mass transfer parameter
    m2: chex.Array = struct.field(default_factory=lambda: jnp.array(0.33))  # Mass transfer parameter 2
    Mco2: chex.Array = struct.field(default_factory=lambda: jnp.array(0.0044))  # Molar mass CO2
    MF: chex.Array = struct.field(
        default_factory=lambda: jnp.array(1.157 * (10 ** (-7))))  # Fruit maintenance respiration coefficient
    ML: chex.Array = struct.field(
        default_factory=lambda: jnp.array(2.894 * (10 ** (-7))))  # Vegetative maintenance respiration coefficient
    mp: chex.Array = struct.field(default_factory=lambda: jnp.array(4.57))  # Watt to micromol conversion constant
    p1: chex.Array = struct.field(
        default_factory=lambda: jnp.array(-2.17 * (10 ** (-4))))  # Net photosynthesis parameter 1 -> check these
    p2: chex.Array = struct.field(
        default_factory=lambda: jnp.array(3.31 * (10 ** (-3))))  # Net photosynthesis parameter 2 -> check these
    p3: chex.Array = struct.field(default_factory=lambda: jnp.array(577.0))  # Net photosynthesis parameter 3
    p4: chex.Array = struct.field(default_factory=lambda: jnp.array(221.0))  # Net photosynthesis parameter 4
    p5: chex.Array = struct.field(
        default_factory=lambda: jnp.array(5 * (10 ** (-5))))  # Net photosynthesis parameter 5 -> check these
    patm: chex.Array = struct.field(default_factory=lambda: jnp.array(101.0))  # atmospheric pressure
    pm: chex.Array = struct.field(
        default_factory=lambda: jnp.array(2.2538 * (10 ** (-3))))  # Maximum photosynthesis rate
    qg: chex.Array = struct.field(default_factory=lambda: jnp.array(2.0))  # fruit growth rate parameter
    qr: chex.Array = struct.field(default_factory=lambda: jnp.array(2.0))  # maintenance respiration
    rg: chex.Array = struct.field(default_factory=lambda: jnp.array(8.3144))  # Gas constant
    s1: chex.Array = struct.field(
        default_factory=lambda: jnp.array(1.8407 ** (-4)))  # saturated water vapour pressure curve slope parameter 1
    s2: chex.Array = struct.field(default_factory=lambda: jnp.array(
        9.7838 ** (10 ** (-4))))  # saturated water vapour pressure curve slope parameter 2
    s3: chex.Array = struct.field(
        default_factory=lambda: jnp.array(0.051492))  # saturated water vapour pressure curve slope parameter 3
    T0: chex.Array = struct.field(default_factory=lambda: jnp.array(273.15))  # conversion from Celsius to K
    Tg: chex.Array = struct.field(default_factory=lambda: jnp.array(20.0))  # growth rate temperature reference
    Td: chex.Array = struct.field(default_factory=lambda: jnp.array(10.0))  # Deep soil temperature
    Tr: chex.Array = struct.field(
        default_factory=lambda: jnp.array(25.0))  # Maintenance respiration reference temperature
    v: chex.Array = struct.field(
        default_factory=lambda: jnp.array(1.23))  # Vegetative assimilate requirement coefficient
    v1: chex.Array = struct.field(
        default_factory=lambda: jnp.array(1.3774))  # Vegetative fruit growth ratio parameter 1
    v2: chex.Array = struct.field(
        default_factory=lambda: jnp.array(-0.168))  # Vegetative fruit growth ratio parameter 2
    v3: chex.Array = struct.field(default_factory=lambda: jnp.array(19.0))  # Vegetative fruit growth ratio parameter 3
    vp: chex.Array = struct.field(default_factory=lambda: jnp.array(7.85))  # Heating pipe volume
    vg_ag: chex.Array = struct.field(default_factory=lambda: jnp.array(10.0))  # Average greenhouse height
    wr: chex.Array = struct.field(default_factory=lambda: jnp.array(32.23))  # LAI correction function parameter
    laim: chex.Array = struct.field(default_factory=lambda: jnp.array(2.511))  # LAI correction function parameter
    yf: chex.Array = struct.field(default_factory=lambda: jnp.array(0.5983))  # Fruit harvest coefficient parameter 1
    yl: chex.Array = struct.field(default_factory=lambda: jnp.array(0.5983))  # Fruit harvest coefficient parameter 2
    z: chex.Array = struct.field(default_factory=lambda: jnp.array(0.6081))  # Leaf fraction of vegetative dry weight
    phi: chex.Array = struct.field(default_factory=lambda: jnp.array(4 * (10 ** (-3))))  # heat valve opening
    rh: chex.Array = struct.field(default_factory=lambda: jnp.array(0.3))  # relative valve opening
    pg: chex.Array = struct.field(default_factory=lambda: jnp.array(0.475))  # PAR to global radiation ratio
    inj_scale: chex.Array = struct.field(default_factory=lambda: jnp.array(10 ** (-3)))
    dt: chex.Array = struct.field(default_factory=lambda: jnp.array(300))  # seconds for forward euler integration


@chex.dataclass
class GreenHouseRewardParams:
    control_cost: chex.Array = struct.field(default_factory=lambda: jnp.array(0.02))
    tg_min: chex.Array = struct.field(default_factory=lambda: jnp.array(15))
    tg_max: chex.Array = struct.field(default_factory=lambda: jnp.array(22))
    tg_des: chex.Array = struct.field(default_factory=lambda: jnp.array(20))
    violation_penalty: chex.Array = struct.field(default_factory=lambda: jnp.array(0.02))

class GreenHouseEnv(Env):
    init_state = jnp.array(
        [
            # t_g, t_p, t_s, c_i, v_i, mb, mf, ml, d_p, t_o, t_d, c_o, v_o, w, G, t
            18, 30, 10, 4, 10 ** (-3), 5, 50, 5, 0.0, 20.0, 5.0, 0.2, 10 ** (-3), 4, 150, 0,
        ])

    noise_std = jnp.array(
        [
            # t_g, t_p, t_s, c_i, v_i, mb, mf, ml, d_p, t_o, t_d, c_o, v_o, w, G, t
            0.05, 0.1, 0.05, 0.01, 0.05, 2, 1, 0.0, 0.00, 2,
            0.1, 1 * (10 ** (-3)), 1 * (10 ** (-4)), 0.2, 20, 0.0,
        ])

    constraint_lb = jnp.array(
        [
            -273.15, -273.15, -273.15, 0, 0, 0, 0, 0, 0, -273.15, -273.15, 0, 0, 0, 0, 0
        ]
    )

    constraint_ub = jnp.array(
        [
            200, 200, 200, jnp.inf, jnp.inf, jnp.inf, jnp.inf, jnp.inf, 1, 200, 200, jnp.inf, jnp.inf, 10, jnp.inf,
            jnp.inf
        ]
    )

    input_ub = jnp.array([80.0, 1.0, 1.0, 2.1])
    input_lb = jnp.array([10.0, 0.0, 0.0, 0.0])

    def __init__(self, dt_integration: float = 60, reward_source: str = 'yield_maximization'):
        self.dynamics_params = GreenHouseParams()
        self.reward_params = GreenHouseRewardParams()
        self.reward_source = reward_source
        self.greenhouse_state_dim = 5
        self.crop_states = 4
        self.exogenous_states = 6
        self.state_dim = self.greenhouse_state_dim + self.crop_states \
                         + self.exogenous_states + 1  # 1 additional state for time
        self.greenhouse_input_dim = 4
        self.eps = 1e-8
        self.dt_integration = dt_integration
        assert self.dt >= dt_integration
        assert (self.dt / dt_integration - int(
            self.dt / dt_integration)) < 1e-4, 'dt must be multiple of dt_integration'
        self._num_steps_integrate = int(self.dt / dt_integration)
        bound = 0.1
        value_at_margin = 0.1
        margin_factor = 10.0
        self.tolerance_reward = ToleranceReward(bounds=(0.0, bound),
                                                margin=margin_factor * bound,
                                                value_at_margin=value_at_margin,
                                                sigmoid='long_tail')

    def reset(self, rng: jax.Array) -> State:
        init_state = self.init_state + jax.random.normal(rng) * self.noise_std
        init_state = jnp.clip(init_state, self.constraint_lb, self.constraint_ub)
        return State(pipeline_state=None,
                     obs=init_state,
                     reward=jnp.array(0.0),
                     done=jnp.array(0.0), )

    def reward(self,
               obs: Float[Array, 'observation_dim'],
               action: Float[Array, 'action_dim']) -> Float[Array, 'None']:

        action_penalty = - self.reward_params.control_cost * jnp.square(action).sum()
        tg = obs[0]
        if self.reward_source == 'yield_maximization':
            mf = obs[self.greenhouse_state_dim + 1]
            # encourage tg_min <= tg <= tg_max
            violation_cost = self.reward_params.violation_penalty * (jax.nn.relu(tg - self.reward_params.tg_max)
                                                                     + jax.nn.relu(self.reward_params.tg_min - tg))
            reward = (mf + action_penalty + violation_cost).squeeze()
        elif self.reward_source == 'temperature_tracking':
            tracking_reward = self.tolerance_reward(jnp.sqrt((tg - self.reward_params.tg_des) ** 2))
            reward = (tracking_reward + action_penalty).squeeze()
        else:
            raise NotImplementedError

        return reward

    @staticmethod
    def buffer_switching_func(mb: jax.Array, b1: jax.Array):
        return 1 - jnp.exp(-b1 * mb)

    def get_respiration_param(self, obs: jax.Array, action: jax.Array, params: GreenHouseParams):
        # ml = x[self.greenhouse_state_dim + 2]
        # l_lai = (ml / params.wr) ** (params.laim) / (1 + (ml / params.wr) ** (params.laim))
        R = - params.p1 - params.p5
        return R

    def get_crop_photosynthesis(self, obs: jax.Array, action: jax.Array, params: GreenHouseParams):
        t_g, c_i = obs[0], obs[3]
        ml = obs[self.greenhouse_state_dim + 2]
        G = obs[-2]
        i_par = params.eta * G * params.mp * params.pg
        # note patm is in kPa. c_i is in g/m^3
        # c_ppm units: m^3 Pa/(mol K) * K * g/m^3 /(kPa * kg/mol)
        # c_ppm: units 10^-3 m^3 kPa/mol * 10^-3 kg/m^3 /(kPa * kg/mol)
        # c_ppm: units 10^-6 [] -> need to multiply with 10^-6 to get right units
        # c_ppm = 1/mol * g/kg
        c_ppm = params.rg / (params.patm * params.Mco2) * (t_g + params.T0) * c_i
        l_lai = (ml / params.wr) ** (params.laim) / (1 + (ml / params.wr) ** (params.laim))
        p_g = params.pm * l_lai * i_par / (params.p3 + i_par) * c_ppm / (params.p4 + c_ppm)
        return p_g

    def get_harvest_coefficient(self, obs: jax.Array, action: jax.Array, params: GreenHouseParams):
        d_p = obs[self.greenhouse_state_dim + 3]
        t = obs[-1]
        t_g = obs[0]
        temp_ratio = jnp.clip(t_g / params.d3, a_min=1e-6)
        h = (d_p >= 1) * (params.d1 + params.d2 * jnp.log(temp_ratio) - params.d4 * t)
        return h

    def ode(self, obs: jax.Array, action: jax.Array, params: GreenHouseParams) -> jax.Array:
        # C, C, C, m, g/m^3, kg/m^-3
        t_g, t_p, t_s, c_i, v_i = obs[0], obs[1], obs[2], obs[3], obs[4]
        # g/m^-2, g/m^-2, g/m^-2, []
        mb, mf, ml, d_p = obs[self.greenhouse_state_dim], obs[self.greenhouse_state_dim + 1], \
            obs[self.greenhouse_state_dim + 2], obs[self.greenhouse_state_dim + 3]
        t = obs[-1]
        egs_start_idx = self.greenhouse_state_dim + self.crop_states
        # C, C, g/m^3, kg/m^-3, m/s, w/m^-2
        t_o, t_d, c_o, v_o, w, G = obs[egs_start_idx], obs[egs_start_idx + 1], \
            obs[egs_start_idx + 2], obs[egs_start_idx + 3], obs[egs_start_idx + 4], obs[egs_start_idx + 5]
        t_h, rwl, rww, phi_c = action[0], action[1], action[2], action[3]
        phi_v = (params.sigma * rwl / (1 + params.chi * rwl) + params.zeta + params.xi * rww) * w \
                + params.psi
        k_v = params.rho_a * params.cp_a * phi_v
        alpha = params.nu * jnp.sqrt(params.tau + jnp.sqrt(jnp.abs(t_g - t_p)))
        s = params.s1 * jnp.power(t_g, 2) + params.s2 * t_g + params.s3
        p_g_star = params.a1 * jnp.exp((params.a2 * t_g) / (params.a3 + t_g + self.eps))
        # convert temp to Kelvin and then from pascal to kpa. v_i is in g/m^3
        p_g = (params.lam * (t_g + params.T0) * v_i)
        Dg = p_g_star - p_g
        # g1: mm/s, g2: [], g4: m^3/g, g3: s m ^2 /micromol
        g = params.g1 * (1 - params.g2 * jnp.exp(-params.g3 * G)) * jnp.exp(-params.g4 * c_i)
        l = params.l1 - params.l2 * t_g
        # gb: mm/s^-1, rho_a = kg/m^-3, cp_a: J/(C kg), Dg: kPa, E:g/(sm^2)
        # G: W/m^2, s: KPa/C
        E = (s * params.eta * G + params.rho_a * params.cp_a * Dg * params.gb) / \
            (l * (s + params.gamma * (1 + params.gb / g)))
        Wg = params.omega * p_g / (params.patm - p_g + self.eps)
        Wc = params.omega * p_g_star / (params.patm - p_g_star + self.eps)

        t_c = params.epsilon / (params.epsilon + 1) * t_o + 1 / params.epsilon * t_g
        Mc = jax.nn.relu(Wg - Wc) * params.m1 * (jnp.abs(t_g - t_c) ** params.m2)
        dt_g_dt = (k_v + params.kr) * (t_o - t_g) + alpha * (t_p - t_g) + params.ks * (t_s - t_g) \
                  + G * params.eta - l * E + l / (1 + params.epsilon) * Mc
        dt_g_dt = dt_g_dt / params.cg

        phi = params.phi
        rh = params.rh
        phi = 2 * phi * rh / (2 - rh)
        dt_p_dt = params.ap / (params.rho_w * params.cp_w) * (params.beta * G - alpha * (t_p - t_g)) + phi * (
                t_h - t_p)
        dt_p_dt = dt_p_dt / params.vp

        dt_s_dt = 1 / params.cs * (params.ks * (t_g - t_s) + params.kd * (params.Td - t_s))

        dc_i_dt = phi_v * (c_o - c_i) + phi_c * params.inj_scale \
                  + self.get_respiration_param(obs, action, params) \
                  - params.mu * self.get_crop_photosynthesis(obs, action, params)
        dc_i_dt = dc_i_dt / params.vg_ag

        dv_i_dt = (E / 1000 - phi_v * (v_i - v_o) - Mc / 1000) / params.vg_ag

        # Tomato model

        g_f = (params.f1 - params.f2 * d_p) * (params.qg ** ((t_g - params.Tg) / 10.0))
        g_l = g_f * params.v1 * jnp.exp(params.v2 * (t_g - params.v3))
        b = self.buffer_switching_func(mb, params.b1)
        buff_1 = b * (params.f * g_f * mf + params.v * g_l * ml / params.z)
        factor = (params.qr ** ((t_g - params.Tg) / 10.0))
        rf = params.MF * factor
        rl = params.ML * factor
        buff_2 = b * (rf * mf + rl * ml / params.z)
        dmb_dt = self.get_crop_photosynthesis(obs, action, params) - buff_1 - buff_2
        h = self.get_harvest_coefficient(obs, action, params)
        hf, hl = h * params.yf, h * params.yl
        dmf_dt = (b * g_f - (1 - b) * rf - hf) * mf
        dml_dt = (b * g_l - (1 - b) * rl - hl) * ml
        temp_ratio = jnp.clip(t_g / params.d3, a_min=1e-6)
        dd_p_dt = params.d1 + params.d2 * jnp.log(temp_ratio) - params.d4 * t - h
        # Exogenous effects

        dt_o_dt = jnp.zeros_like(dt_g_dt)
        dt_d_dt = jnp.zeros_like(dt_g_dt)
        dc_o_dt = jnp.zeros_like(dt_g_dt)
        dv_o_dt = jnp.zeros_like(dt_g_dt)
        dw_dt = jnp.zeros_like(dt_g_dt)
        dG_dt = jnp.zeros_like(dt_g_dt)

        # Time
        dt_dt = jnp.ones_like(dt_g_dt)

        dx_dt = jnp.stack([
            dt_g_dt, dt_p_dt, dt_s_dt, dc_i_dt, dv_i_dt,
            dmb_dt, dmf_dt, dml_dt, dd_p_dt,
            dt_o_dt, dt_d_dt, dc_o_dt, dv_o_dt, dw_dt, dG_dt, dt_dt,
        ])
        return dx_dt

    def integrate(self, obs: jax.Array, action: jax.Array, params: GreenHouseParams):
        def body(carry, _):
            q = carry + self.dt_integration * self.ode(carry, action, params)
            q = jnp.clip(q, a_min=self.constraint_lb, a_max=self.constraint_ub)
            return q, None

        x_next, _ = jax.lax.scan(body, obs, xs=None, length=self._num_steps_integrate)
        return x_next

    def scale_action(self, action: jax.Array) -> jax.Array:
        action = 0.5 * (action * (self.input_ub - self.input_lb) + (self.input_ub + self.input_lb))
        return action

    @partial(jax.jit, static_argnums=0)
    def step(self,
             state: State,
             action: jax.Array) -> State:
        obs = state.obs
        chex.assert_shape(obs, (self.observation_size,))
        chex.assert_shape(action, (self.action_size,))
        next_reward = self.reward(obs, action)
        action = self.scale_action(action)
        next_obs = self.integrate(obs, action, self.dynamics_params)
        next_state = State(pipeline_state=state.pipeline_state,
                           obs=next_obs,
                           reward=next_reward,
                           done=state.done,
                           metrics=state.metrics,
                           info=state.info)
        return next_state

    @property
    def dt(self):
        return self.dynamics_params.dt

    @property
    def observation_size(self) -> int:
        return self.state_dim

    @property
    def action_size(self) -> int:
        return self.greenhouse_input_dim

    def backend(self) -> str:
        return 'positional'


if __name__ == '__main__':
    env = GreenHouseEnv()
    key = jax.random.PRNGKey(12345)
    key1, key2 = jax.random.split(key, 2)
    init_state = env.reset(key1)
    action = jax.random.uniform(key=key2, shape=(4,), minval=-1, maxval=1)
    next_state = env.step(init_state, action)
