import re
import numpy as np
import pandas as pd
import gymnasium
from gymnasium import spaces
from dataclasses import dataclass, field, asdict
import traceback
import logging as log
from load_profile_data_loader import LoadProfileDataLoader

log.basicConfig(format="Line:%(lineno)d-%(funcName)s-%(levelname)s:  %(message)s")
log.getLogger().setLevel(log.INFO)

@dataclass
class EnvState:
    """
    Represents the state of the microgrid environment at a single timestep.
    Holds both scaled values for agent observation and unscaled values for internal environment logic.
    """
    
    # Scaled features for agent observation
    index: int
    ts_hour_sin: float
    ts_hour_cos: float
    tou_offpeak: int
    tou_standard: int
    tou_peak: int
    day_week: int
    day_saturday: int
    day_sunday: int
    site_load_energy: float    
    solar_prod_energy: float   
    solar_ctlr_setpoint: float
    grid_import_energy: float

    # Unscaled values for internal environment logic & reward calculation
    site_load_energy_unscaled: float
    solar_prod_energy_unscaled: float
    solar_ctlr_setpoint_unscaled: float
    grid_import_energy_unscaled: float

    # Other parameters mostly unscaled
    grid_notified_maximum_demand: float
    bess_avail_discharge_energy: float
    bess_capacity: float
    bess_cycle_efficiency:float
    bess_charge_from_solar_energy: float
    bess_charge_from_grid_energy: float  
    bess_discharge_energy: float         
    bess_prev_action_name: str
    bess_cycle_counter: int
    tou_peak_tariff: float
    tou_standard_tariff: float
    tou_offpeak_tariff:float
    solar_ppa_tariff: float
    debug_flag: bool

    # Calculated fields which use unscaled values for their logic
    grid_import_energy_with_bess: float = field(init=False)
    grid_surplus_energy: float = field(init=False)
    solar_surplus_energy: float = field(init=False)  
    solar_vs_load_ratio: float = field(init=False)
    bess_soc: float = field(init=False)              

    action_name: str = field(init=False)
    reward_earned: float = field(init=False)


    def __post_init__(self):
        """
        Post-initialization processing for calculated fields.
        """
        self.grid_import_energy_with_bess = 0.0
        self.grid_surplus_energy = self.get_grid_surplus_energy()
        self.solar_surplus_energy = self.get_solar_surplus_energy()
        self.solar_vs_load_ratio = self.get_solar_vs_load_ratio()
        self.bess_soc = self.get_bess_soc()


    def get_gym_state(self):
        """
        Returns the environment state formatted as a NumPy array for the Gym environment.
        Uses SCALED features for the observation. BESS SOC is calculated and then scaled.
        """

        # Scale BESS SOC from [0, 100] to [-1, 1]
        scaled_bess_soc = (self.bess_soc / 50.0) - 1.0

        return np.array([self.ts_hour_sin,
                         self.ts_hour_cos,
                         self.tou_offpeak,
                         self.tou_standard,
                         self.tou_peak,
                         self.day_week,
                         self.day_saturday,
                         self.day_sunday,
                         self.grid_import_energy,   
                         self.solar_prod_energy,   
                         self.solar_ctlr_setpoint, 
                         scaled_bess_soc], dtype=np.float32)


    def get_grid_surplus_energy(self) -> float:
        """
        Calculates the surplus energy available from the grid using UNSCALED values.
        This is the difference between the notified maximum demand and the current grid import.

        Returns:
            float: The grid surplus energy in kWh, rounded to 2 decimal places.
        """
        return np.round(self.grid_notified_maximum_demand - self.grid_import_energy_unscaled, 2)


    def get_solar_surplus_energy(self) -> float:
        """
        Calculates the surplus solar energy that is not being used or stored using UNSCALED values.
        This is the difference between potential full solar production and actual solar production
        based on the controller setpoint. solar_ctlr_setpoint_unscaled is 0-100.

        Returns:
            float: The solar surplus energy in kWh, rounded to 2 decimal places.
        """

        ctl_setpoint_ratio = self.solar_ctlr_setpoint_unscaled / 100.0
        
        solar_prod_full = self.solar_prod_energy_unscaled / (ctl_setpoint_ratio + 1e-6)
        
        solar_prod_surplus = solar_prod_full - self.solar_prod_energy_unscaled

        if self.debug_flag:
            log.info(f"""
            [{self.index}] Solar Surplus Energy Calculation ->
                    Control Setpoint: {self.solar_ctlr_setpoint_unscaled: .2f}%
                    Control Setpoint Ratio: {ctl_setpoint_ratio: .2f}
                    Solar Production Energy: {self.solar_prod_energy_unscaled: .2f} kWh
                    Solar Full Production Energy: {solar_prod_full: .2f} kWh
                    Solar Surplus Production Energy: {solar_prod_surplus: .2f} kWh
             """)
        
        return np.round(solar_prod_surplus, 2)


    def get_solar_vs_load_ratio(self) -> float:
        """
        Calculates the ratio of current solar production to the current site load.

        Returns:
            float: The solar production vs. site load ratio as a percentage, rounded to 2 decimal places.
        """
        return np.round((self.solar_prod_energy_unscaled / (self.site_load_energy_unscaled + 1e-6)) * 100.0, 2)


    def get_bess_soc(self) -> float:
        """
        Calculates the State of Charge (SOC) of the BESS.

        Returns:
            float: The BESS SOC as a percentage (0-100), rounded to 2 decimal places.
        """
        return np.round((self.bess_avail_discharge_energy / self.bess_capacity) * 100.0, 2)


    def calculate_bess_charge_cost(self) -> float:
        """
        Calculates the cost of charging the BESS from the grid and solar.
        
        Returns:
            float: The total cost of charging the BESS, rounded to 2 decimal places.
        """

        grid_cost = 0.0

        if self.tou_peak:
            
            grid_cost = self.tou_peak_tariff * self.bess_charge_from_grid_energy
            
        elif self.tou_standard:
            
            grid_cost = self.tou_standard_tariff * self.bess_charge_from_grid_energy
            
        elif self.tou_offpeak:
            
            grid_cost = self.tou_offpeak_tariff * self.bess_charge_from_grid_energy

        solar_cost = self.solar_ppa_tariff * self.bess_charge_from_solar_energy

        total_cost = np.round((grid_cost + solar_cost), 2)

        if self.debug_flag:

            log.info(f"""
            [{self.index}] Current BESS Charge Cost: {total_cost: .2f}
            """)

        return total_cost

    
    def calculate_bess_discharge_saving(self) -> float:
        """
        Calculates the savings achieved by discharging the BESS instead of importing from the grid.
       
        Returns:
            float: The total savings from BESS discharge, rounded to 2 decimal places.
        """

        grid_saving = 0.0

        if self.tou_peak:
            
            grid_saving = self.tou_peak_tariff * self.bess_discharge_energy
            
        elif self.tou_standard:
            
            grid_saving = self.tou_standard_tariff * self.bess_discharge_energy
            
        elif self.tou_offpeak:
            
            grid_saving = self.tou_offpeak_tariff * self.bess_discharge_energy
            
        if self.debug_flag:

            log.info(f"""
            [{self.index}] Current BESS Discharge Savings: {grid_saving: .2f}
            """)

        return np.round(grid_saving, 2)

    
    def calculate_potential_future_discharge_savings_when_charging(self) -> float:
        """
        Estimates potential future savings if the energy currently being charged
        were to be discharged later during higher tariff periods.

        Returns:
            float: Estimated potential future savings, rounded to 2 decimal places.
        """

        avg_lower_cost_grid_tariffs = 1.0
        
        if self.tou_peak:

            avg_lower_cost_grid_tariffs = (self.tou_offpeak_tariff + self.tou_standard_tariff) / 2
            
        elif self.tou_standard:
            
            avg_lower_cost_grid_tariffs = (self.tou_offpeak_tariff + self.tou_peak_tariff) / 2
            
        elif self.tou_offpeak:
            
            avg_lower_cost_grid_tariffs = (self.tou_standard_tariff + self.tou_peak_tariff) / 2
        
        future_discharge_saving = np.round((avg_lower_cost_grid_tariffs * self.bess_charge_from_grid_energy + self.solar_ppa_tariff * self.bess_charge_from_solar_energy) * self.bess_cycle_efficiency, 2)

        if self.debug_flag:

            log.info(f"""
            [{self.index}] Future BESS Discharge Savings (Charging): {future_discharge_saving: .2f}
                    BESS SoC: {self.bess_soc: .2f}
                    TOU Timeslot: Peak={self.tou_peak} , Standard={self.tou_standard}, Off-peak={self.tou_offpeak}
            """)

        return future_discharge_saving

    
    def calculate_potential_future_charge_cost_when_discharging(self) -> float:
        """
        Estimates the potential future cost of charging the BESS if the energy
        currently being discharged had to be replenished from the grid later.

        Returns:
            float: Estimated potential future charging cost.
        """
        avg_lower_cost_grid_tariffs = 1.0
        
        if self.tou_peak:

            avg_lower_cost_grid_tariffs = (self.tou_offpeak_tariff + self.tou_standard_tariff) / 2
            
        elif self.tou_standard:
            
            avg_lower_cost_grid_tariffs = (self.tou_offpeak_tariff + self.tou_peak_tariff) / 2
            
        elif self.tou_offpeak:
            
            avg_lower_cost_grid_tariffs = (self.tou_standard_tariff + self.tou_peak_tariff) / 2

        grid_charge_energy = (self.bess_discharge_energy / self.bess_cycle_efficiency) - self.solar_surplus_energy
        
        if grid_charge_energy < 0:
            grid_charge_energy = 0
            solar_charge_energy = (self.bess_discharge_energy / self.bess_cycle_efficiency)
        else:
            solar_charge_energy = self.solar_surplus_energy

        future_charge_cost = np.round((avg_lower_cost_grid_tariffs * grid_charge_energy + self.solar_ppa_tariff * solar_charge_energy), 2)

        if self.debug_flag:

            log.info(f"""
            [{self.index}] Future BESS Charge Cost (Discharging) -> {future_charge_cost: .2f}
                    BESS Grid Charge Energy: {grid_charge_energy: .2f} kWh
                    BESS Solar Charge Energy: {solar_charge_energy: .2f} kWh
                    Solar Surplus Energy Available: {self.solar_surplus_energy: .2f} kWh
                    BESS SoC: {self.bess_soc: .2f}%
                    TOU Timeslot: Peak={self.tou_peak} , Standard={self.tou_standard}, Off-peak={self.tou_offpeak}
            """)
        return future_charge_cost


    def calculate_potential_current_discharge_savings_when_do_nothing(self) -> float:
        """
        Calculates the potential savings if the BESS were to discharge now
        instead of doing nothing, considering the current grid import.

        Returns:
            float: Potential current discharge savings.
        """
        current_tariff = 1.0
        
        if self.tou_peak:

            current_tariff = self.tou_peak_tariff
            
        elif self.tou_standard:
            
            current_tariff = self.tou_standard_tariff
            
        elif self.tou_offpeak:
            
            current_tariff = self.tou_offpeak_tariff
        
        potential_current_discharge_saving = np.round((current_tariff * self.grid_import_energy_unscaled * self.bess_cycle_efficiency), 2)

        if self.debug_flag:
            log.info(f"""
            [{self.index}] Potential Current BESS Discharge Savings (Do-Nothing) -> {potential_current_discharge_saving: .2f}
                    Grid Import Energy Unscaled: {self.grid_import_energy_unscaled: .2f} kWh
                    BESS SoC: {self.bess_soc: .2f}%
                    TOU Timeslot: Peak={self.tou_peak} , Standard={self.tou_standard}, Off-peak={self.tou_offpeak}
            """)
        return potential_current_discharge_saving


    def calculate_potential_current_charge_cost_when_do_nothing(self) -> float:
        """
        Calculates the potential cost if the BESS were to charge now
        instead of doing nothing, considering available solar and grid surplus.
        
        Returns:
            float: Potential current charging cost.
        """

        current_tariff = 1.0
        
        if self.tou_peak:

            current_tariff = self.tou_peak_tariff
            
        elif self.tou_standard:
            
            current_tariff = self.tou_standard_tariff
            
        elif self.tou_offpeak:
            
            current_tariff = self.tou_offpeak_tariff

        potential_current_charge_cost = np.round((current_tariff * self.bess_charge_from_grid_energy - self.solar_ppa_tariff * self.solar_surplus_energy), 2)

        if self.debug_flag:

            log.info(f"""
            [{self.index}] Potential Current BESS Charge Cost (Do-Nothing): {potential_current_charge_cost: .2f}
                    BESS SoC: {self.bess_soc: .2f}
                    TOU Timeslot: Peak={self.tou_peak} , Standard={self.tou_standard}, Off-peak={self.tou_offpeak}
            """)

        return potential_current_charge_cost

    
    def calculate_without_bess_cost(self) -> float:
        """
        Calculates the cost of operating the microgrid without the BESS.

        Returns:
            float: The operational cost without BESS.
        """
        grid_cost = 0.0

        if self.tou_peak:
            
            grid_cost = self.tou_peak_tariff * self.grid_import_energy_unscaled
            
        elif self.tou_standard:
            
            grid_cost = self.tou_standard_tariff * self.grid_import_energy_unscaled
            
        elif self.tou_offpeak:
            
            grid_cost = self.tou_offpeak_tariff * self.grid_import_energy_unscaled

        solar_cost = self.solar_ppa_tariff * self.solar_prod_energy_unscaled

        total_cost = np.round((grid_cost + solar_cost), 2)

        if self.debug_flag:

            log.info(f"""
            [{self.index}] Total Cost Without BESS: {grid_cost: .2f} + {solar_cost: .2f} = {total_cost: .2f}
            """)

        return total_cost
        

    def calculate_bess_soc_reward(self):
        """
        Calculates a continuous reward component based on the BESS State of Charge (SOC).
        This reward function is designed to keep the SOC between 20% and 90%, with
        optimal operation in the 40%-70% range.

        Returns:
            float: The reward/penalty related to BESS SOC, rounded to 2 decimal places.
        """

        action_is_charge = re.match(r"^charge.*", self.action_name) is not None
        action_is_discharge = re.match(r"^discharge.*", self.action_name) is not None
        
        # Define the ideal SOC range
        min_soc = 20.0
        max_soc = 90.0
        optimal_low = 40.0
        optimal_high = 70.0
        
        # Base reward calculation using a piecewise function
        if min_soc <= self.bess_soc <= max_soc:
            # Within the acceptable range
            if optimal_low <= self.bess_soc <= optimal_high:
                # In the optimal range - highest reward
                base_reward = 500.0
            elif self.bess_soc < optimal_low:
                # Between min and optimal low - scale reward linearly
                base_reward = 100.0 + 400.0 * ((self.bess_soc - min_soc) / (optimal_low - min_soc))
            else:  # self.bess_soc > optimal_high
                # Between optimal high and max - scale reward linearly
                base_reward = 100.0 + 400.0 * ((max_soc - self.bess_soc) / (max_soc - optimal_high))
        else:
            # Outside acceptable range - apply penalties
            if self.bess_soc < min_soc:
                # Below minimum - significant negative reward that grows as SOC approaches 0
                penalty_factor = 1.0 + 2.0 * ((min_soc - self.bess_soc) / min_soc)
                base_reward = -1000.0 * penalty_factor
            else:  # self.bess_soc > max_soc
                # Above maximum - increasing penalty as SOC approaches 100%
                penalty_factor = 1.0 + 1.5 * ((self.bess_soc - max_soc) / (100.0 - max_soc))
                base_reward = -800.0 * penalty_factor
        
        # Additional logic for action-specific incentives
        action_modifier = 0.0
        if action_is_charge and self.bess_soc >= max_soc:
            # Extra penalty for charging when already near capacity
            action_modifier = -500.0 * ((self.bess_soc - max_soc) / (100.0 - max_soc))
        elif action_is_discharge and self.bess_soc <= min_soc:
            # Extra penalty for discharging when already near empty
            action_modifier = -500.0 * ((min_soc - self.bess_soc) / min_soc)
        
        # Apply the action-specific modifier
        bess_soc_reward = base_reward + action_modifier
            
        if self.debug_flag:
            log.info(f"""
            [{self.index}] BESS SoC Reward: {bess_soc_reward: .2f}
                    BESS SoC: {self.bess_soc: .2f}%
                    Action is charge: {action_is_charge}
                    Action is discharge: {action_is_discharge}
                    Base reward: {base_reward: .2f}
                    Action modifier: {action_modifier: .2f}
            """)

        return np.round(bess_soc_reward, 2)
    

    def calculate_bess_cycle_penalty(self):
        """
        Calculates a penalty based on BESS cycling to account for degradation.

        Returns:
            float: The penalty for BESS cycling.
        """

        bess_cycle_penalty = 0.0

        cycle_life = 5000  # Typical Li-ion battery cycle life
        battery_replacement_cost = 500_000 # Typical cost in USD of replacing a BESS
        base_cycle_cost = battery_replacement_cost / cycle_life

        cycle_penalty_factor = 1.0 + 0.1 * (self.bess_cycle_counter - 1) if self.bess_cycle_counter > 1 else 1.0

        bess_cycle_penalty = base_cycle_cost * self.bess_cycle_counter * cycle_penalty_factor

        self.bess_prev_action_name = self.action_name

        if self.debug_flag:

            log.info(f"""
            [{self.index}] BESS Cycle Penalty: {bess_cycle_penalty: .2f}
                    BESS Cycle Counter: {self.bess_cycle_counter}
            """)

        return np.round(bess_cycle_penalty, 2)


    def get_reward(self):
        """
        Calculates the total reward for the current state and action.
        This typically involves costs/savings from BESS operations, grid interaction,
        and potentially penalties or bonuses for SOC levels or cycling.

        Returns:
            float: The total calculated reward.
        """

        reward = 0.0

        if re.match(r"^charge.*", self.action_name) is not None: # charge

            reward = (
                - self.calculate_bess_charge_cost() # current cost of charging the bess
                + self.calculate_potential_future_discharge_savings_when_charging() # future savings when discharging the bess
                + self.calculate_bess_soc_reward() # rewards/penalize the agent for keeping the bess soc in the optimal range
                - self.calculate_bess_cycle_penalty() # penalize the agent for cycling the bess too much
                )

        elif re.match(r"^discharge.*", self.action_name) is not None: # discharge

            reward = (
                + self.calculate_bess_discharge_saving() # current saving from discharging the bess
                - self.calculate_potential_future_charge_cost_when_discharging() # future cost for charging the bess
                + self.calculate_bess_soc_reward() # rewards/penalize the agent for keeping the bess soc in the optimal range
                - self.calculate_bess_cycle_penalty() # penalize the agent for cycling the bess too much
                )

        else: # do-nothing

            reward = (
                + self.calculate_bess_soc_reward() # penalize the agent if the bess soc falls below 20%
                + self.calculate_potential_current_charge_cost_when_do_nothing() # saved cost for not charging the bess now
                - self.calculate_potential_current_discharge_savings_when_do_nothing() # missed savings for not discharging the bess now
                )

        self.reward_earned = reward

        if self.debug_flag:

          log.info(f"""
          [{self.index}] Calculated Reward -> {reward: .3f}
          """)
        
        return reward


class MicrogridEnv(gymnasium.Env):
  metadata = {"render_modes": ["human"]}
  """
  A custom Gymnasium environment for simulating a microgrid with a BESS.

  This environment models the interaction between solar generation, site load,
  grid electricity, and a Battery Energy Storage System (BESS). The agent\'s goal
  is to learn a policy for BESS charging and discharging to minimize operational
  costs or maximize savings, considering time-of-use tariffs and other factors.
  """

  def __init__(self, data: pd.DataFrame, loader: LoadProfileDataLoader,
               grid_notified_maximum_demand: float, bess_capacity:float, bess_cycle_efficiency: float, bess_step_sizes: list,
               tou_peak_tariff: float, tou_standard_tariff: float, tou_offpeak_tariff: float, solar_ppa_tariff: float, debug_flag: bool):
      """
      Initializes the MicrogridEnv.

      Args:
          data (pd.DataFrame): DataFrame containing the load profile and solar generation data.
          loader (LoadProfileDataLoader): Instance of the data loader for scaling information.
          grid_notified_maximum_demand (float): The notified maximum demand from the grid (kVA).
          bess_capacity (float): The total capacity of the BESS (kWh).
          bess_cycle_efficiency (float): The round-trip efficiency of the BESS (0.0 to 1.0).
          bess_step_sizes (list): A list of possible energy amounts (kWh) for BESS charge/discharge actions.
                                   The order corresponds to action indices (e.g., charge-large, charge-small, idle, discharge-small, discharge-large).
          tou_peak_tariff (float): Tariff rate during peak hours (cost per kWh).
          tou_standard_tariff (float): Tariff rate during standard hours (cost per kWh).
          tou_offpeak_tariff (float): Tariff rate during off-peak hours (cost per kWh).
          solar_ppa_tariff (float): Tariff for solar Power Purchase Agreement (cost per kWh).
          debug_flag (bool): If True, enables detailed debug logging.
      """

      super(MicrogridEnv).__init__()

      self.env_data = data
      self.loader = loader

      self.action_space = spaces.Discrete(5)
      self.action_space_dict = {0: 'charge-1000', 1: 'charge-250', 2: 'do-nothing', 3: 'discharge-250', 4: 'discharge-1000'}
      self.observation_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
      
      self.state = None
      self.state_idx = 0
      self.reward = 0.0
      self.reward_scale_factor = 1000.0
      self.done = False
      self.truncated = False
      self.debug_flag = debug_flag
      self.bess_cycle_counter = 0
      self.bess_prev_action_name = 'do-nothing'
      self.bess_capacity = bess_capacity
      self.bess_avail_discharge_energy = self.bess_capacity
      self.bess_cycle_efficiency = bess_cycle_efficiency
      self.bess_step_sizes = bess_step_sizes
      self.grid_notified_maximum_demand = grid_notified_maximum_demand
      self.tou_peak_tariff = tou_peak_tariff
      self.tou_standard_tariff = tou_standard_tariff
      self.tou_offpeak_tariff = tou_offpeak_tariff
      self.solar_ppa_tariff = solar_ppa_tariff
      self.monitoring_metrics = {}
      self.display_info()


  def display_info(self):
    """
    Displays information about the environment setup and data summary.
    """

    log.info("Environment Setup: ")
    log.info(f"""
    Grid Notified Maximum Demand: {self.grid_notified_maximum_demand} kVA
    BESS Capacity: {self.bess_capacity} kWh
    BESS Actions: {', '.join( list( self.action_space_dict.values() ) )}
    """)

    log.info("\nData Summary: ")
    log.info(self.env_data.info())
    log.info("\n")


  def get_data(self):
      """
      Returns the environment's historical load profile and solar data.

      Returns:
          pd.DataFrame: The DataFrame containing environment data.
      """

      return self.env_data


  def load_new_state(self):
      """
      Loads the data for the current timestep.
      Fetches SCALED data for observation, then INVERSE-TRANSFORMS necessary
      features to their UNSCALED (physical) values for EnvState internal logic.

      Returns:
          EnvState: An object representing the current state of the environment.
      """
      load_profile_data_scaled = self.get_data()
      obs_scaled = load_profile_data_scaled.iloc[self.state_idx, :]

      # Contains scaled site_load_energy, solar_prod_energy etc.
      state_obj = obs_scaled.to_dict() 

      # Inverse-transform features needed for internal logic
      columns_to_unscale = {
          'site_load_energy': 'site_load_energy_unscaled',
          'solar_prod_energy': 'solar_prod_energy_unscaled',
          'solar_ctlr_setpoint': 'solar_ctlr_setpoint_unscaled',
          'grid_import_energy': 'grid_import_energy_unscaled'
      }

      for scaled_col, unscaled_col_name in columns_to_unscale.items():
          
          if scaled_col in self.loader.scalers:
              
              scaler = self.loader.scalers[scaled_col]
              scaled_value = obs_scaled[scaled_col]
              unscaled_value = scaler.inverse_transform(np.array([[scaled_value]]))[0, 0]
              state_obj[unscaled_col_name] = unscaled_value

          else:
             
              state_obj[unscaled_col_name] = obs_scaled[scaled_col]

              if self.debug_flag:
                  log.warning(f"Scaler for '{scaled_col}' not found. Using its scaled value for '{unscaled_col_name}'.")


      state_obj['index'] = self.state_idx
      state_obj['grid_notified_maximum_demand'] = self.grid_notified_maximum_demand
      state_obj['bess_capacity'] = self.bess_capacity
      state_obj['bess_cycle_efficiency'] = self.bess_cycle_efficiency
      state_obj['bess_avail_discharge_energy'] = self.bess_avail_discharge_energy
      state_obj['bess_discharge_energy'] = 0.0
      state_obj['bess_charge_from_grid_energy'] = 0.0
      state_obj['bess_charge_from_solar_energy'] = 0.0
      state_obj['bess_cycle_counter'] = self.bess_cycle_counter
      state_obj['bess_prev_action_name'] = self.bess_prev_action_name
      state_obj['tou_peak_tariff'] = self.tou_peak_tariff
      state_obj['tou_standard_tariff'] = self.tou_standard_tariff
      state_obj['tou_offpeak_tariff'] = self.tou_offpeak_tariff
      state_obj['solar_ppa_tariff'] = self.solar_ppa_tariff
      state_obj['debug_flag'] = self.debug_flag
      
      return EnvState(**state_obj)
  

  def render(self, mode="human"):
      """
      Renders the current state of the environment.
      For "human" mode, it prints key state variables, including unscaled for clarity.
      """
      
      print(f"""
        [{self.state.index}] Environment State ->
                hour_sin: {self.state.ts_hour_sin: .4f} (scaled)
                hour_cos: {self.state.ts_hour_cos: .4f} (scaled)
                tou_offpeak: {self.state.tou_offpeak}
                tou_standard: {self.state.tou_standard}
                tou_peak: {self.state.tou_peak}
                day_week: {self.state.day_week}
                day_saturday: {self.state.day_saturday}
                day_sunday: {self.state.day_sunday}
                site_load_energy: {self.state.site_load_energy: .2f} (scaled), {self.state.site_load_energy_unscaled: .2f} (kWh)
                solar_prod_energy: {self.state.solar_prod_energy: .2f} (scaled), {self.state.solar_prod_energy_unscaled: .2f} (kWh)
                solar_ctlr_setpoint: {self.state.solar_ctlr_setpoint: .2f} (scaled), {self.state.solar_ctlr_setpoint_unscaled: .2f} (%)
                solar_vs_load_ratio: {self.state.solar_vs_load_ratio: .2f} (%) (uses unscaled)
                grid_import_energy: {self.state.grid_import_energy: .2f} (scaled), {self.state.grid_import_energy_unscaled: .2f} (kWh)
                grid_surplus_energy: {self.state.grid_surplus_energy: .2f} (kWh) (uses unscaled)
                solar_surplus_energy: {self.state.solar_surplus_energy: .2f} (kWh) (uses unscaled)
                bess_capacity: {self.state.bess_capacity: .2f} (kWh)
                bess_cycle_efficiency: {self.state.bess_cycle_efficiency: .2f}
                bess_avail_discharge_energy: {self.state.bess_avail_discharge_energy: .2f} (kWh)
                bess_soc: {self.state.bess_soc: .2f} (%)
                action_name: {self.state.action_name}
                reward_earned (raw): {self.state.reward_earned: .3f} 
                done: {self.done}
        """)

    
  def get_number_of_actions(self):
      """
      Returns the total number of discrete actions available to the agent.

      Returns:
          int: The number of actions.
      """

      return len(self.action_space_dict)

    
  def sample_action(self):
      """
      Samples a random action from the action space.

      Returns:
          int: A randomly selected action index.
      """

      return np.random.choice( self.get_number_of_actions() )

    
  def rule_based_policy(self):
      """
      Implements a simple rule-based policy for BESS control.
      This can be used as a baseline or for comparison.

      The policy attempts to:
      - Discharge during peak hours.
      - Charge from solar surplus during standard hours.
      - Charge from grid during off-peak hours if BESS is not full.
      - Otherwise, do nothing.

      Returns:
          int: The action index determined by the rule-based policy.
      """

      # Action space indices
      charge_1000_idx = 0
      charge_250_idx = 1
      do_nothing_idx = 2
      discharge_250_idx = 3
      discharge_1000_idx = 4

      action_idx = None

      if self.state.tou_peak == 1:
          # Discharge 1000 kWh when it's TOU Peak
          action_idx = discharge_1000_idx

      elif (self.state.tou_standard == 1) and (self.state.solar_surplus_energy == 0.0) and (self.state.bess_avail_discharge_energy < self.state.bess_capacity):
          # Charge 250 kWh when it's TOU Standard and when there is no Surplus Solar PV and the BESS available energy is less than it's full capacity
          action_idx = charge_250_idx
          
      elif (self.state.tou_standard == 1) and (self.state.solar_surplus_energy > 0.0) and (self.state.bess_avail_discharge_energy < self.state.bess_capacity):
          # Charge 1000 kWh when it's TOU Standard and when there is Surplus Solar PV and the BESS available energy is less than it's full capacity
          action_idx = charge_1000_idx
          
      elif (self.state.tou_offpeak == 1) and (self.state.bess_avail_discharge_energy < self.state.bess_capacity):
          # Charge 1000 kWh when it's TOU Off-peak and the BESS available energy is less than it's full capacity
          action_idx = charge_1000_idx
          
      else:
          # Default action is to do nothing
          action_idx = do_nothing_idx

      if self.debug_flag:
          
          log.info(f"""
            [{self.state.index}] Rule-Based Policy Selected Action -> {self.action_space_dict[action_idx]}:
                Solar Surplus Energy: {self.state.solar_surplus_energy: .2f} kWh
                BESS Available Discharge Energy: {self.state.bess_avail_discharge_energy: .2f} kWh
                TOU Timeslot: Peak={self.state.tou_peak}, Standard={self.state.tou_standard}, Off-peak={self.state.tou_offpeak}
            """)
      
      return action_idx

    
  def calculate_bess_charge_energy(self, action: int):
      """
      Calculates the actual BESS charge energy based on the selected action,
      solar surplus, UNSCALED grid surplus, and BESS available capacity.
      Updates self.state.bess_charge_from_solar_energy and self.state.bess_charge_from_grid_energy (in kWh).
      """
      charge_step_size = self.bess_step_sizes[action]
      
      required_charge_from_grid_import_energy = charge_step_size - self.state.solar_surplus_energy

      charge_from_solar_energy = 0.0
      charge_from_grid_import_energy = 0.0

      # Determine the possible bess charge energy split between solar and grid
      if (required_charge_from_grid_import_energy > 0) and (self.state.grid_surplus_energy > required_charge_from_grid_import_energy):
          
          charge_from_grid_import_energy = required_charge_from_grid_import_energy
          charge_from_solar_energy = self.state.solar_surplus_energy
          self.state.grid_import_energy_with_bess = self.state.grid_import_energy_unscaled + charge_from_grid_import_energy

      elif (required_charge_from_grid_import_energy > 0) and (self.state.grid_surplus_energy < required_charge_from_grid_import_energy):

          charge_from_grid_import_energy = self.state.grid_surplus_energy
          charge_from_solar_energy = self.state.solar_surplus_energy
          self.state.grid_import_energy_with_bess = self.state.grid_import_energy_unscaled + charge_from_grid_import_energy

      else:

          charge_from_grid_import_energy = 0.0
          charge_from_solar_energy = charge_step_size
          self.state.grid_import_energy_with_bess = self.state.grid_import_energy_unscaled + charge_from_grid_import_energy

      adjusted_charge_step_size = charge_from_solar_energy + charge_from_grid_import_energy
      charge_from_solar_prop = 1.0 if charge_from_grid_import_energy == 0.0 else (charge_from_solar_energy / adjusted_charge_step_size)
      charge_from_grid_import_prop = 1.0 - charge_from_solar_prop

      # Apply the bess charge energy to the battery storage, and adjust for overcharging condition
      if (self.state.bess_avail_discharge_energy + adjusted_charge_step_size) > self.state.bess_capacity:

          surplus_charge_energy = (self.state.bess_avail_discharge_energy + adjusted_charge_step_size) - self.state.bess_capacity

          final_charge_step_size = adjusted_charge_step_size - surplus_charge_energy
          self.state.bess_charge_from_solar_energy = final_charge_step_size * charge_from_solar_prop
          self.state.bess_charge_from_grid_energy = final_charge_step_size * charge_from_grid_import_prop

          # BESS fully charged
          self.bess_avail_discharge_energy = self.state.bess_capacity
          self.state.bess_avail_discharge_energy = self.bess_avail_discharge_energy
          self.state.bess_soc = self.state.get_bess_soc()

      else:

          self.state.bess_charge_from_solar_energy = charge_from_solar_energy
          self.state.bess_charge_from_grid_energy = charge_from_grid_import_energy

          # BESS busy charging
          self.bess_avail_discharge_energy += adjusted_charge_step_size
          self.state.bess_avail_discharge_energy = self.bess_avail_discharge_energy
          self.state.bess_soc = self.state.get_bess_soc()

      
  def calculate_bess_discharge_energy(self, action: int):
      """
      Calculates the actual BESS discharge energy based on the selected action,
      grid import energy (to offset), and BESS available discharge energy.
      Updates self.state.bess_discharge_energy (in kWh).
      """
      discharge_step_size = self.bess_step_sizes[action] # This is in kWh

      adjusted_discharge_step_size = 0.0

      # Check if the discharge step size can be applied without exceeding grid import energy
      if (self.state.grid_import_energy_unscaled - discharge_step_size) < 0.0:
          
          adjusted_discharge_step_size = self.state.grid_import_energy_unscaled
         
      else:
          
          adjusted_discharge_step_size = discharge_step_size 
          
      
      # Apply the bess discharge energy to the battery storage, and adjust for overdischarging condition
      if (self.state.bess_avail_discharge_energy - adjusted_discharge_step_size) < 0.0:

          self.state.bess_discharge_energy = self.state.bess_avail_discharge_energy
          self.state.grid_import_energy_with_bess = self.state.grid_import_energy_unscaled - self.state.bess_discharge_energy

          # BESS fully discharged
          self.bess_avail_discharge_energy = 0.0
          self.state.bess_avail_discharge_energy = self.bess_avail_discharge_energy
          self.state.bess_soc = self.state.get_bess_soc()

      else:

          self.state.bess_discharge_energy = adjusted_discharge_step_size
          self.state.grid_import_energy_with_bess = self.state.grid_import_energy_unscaled - self.state.bess_discharge_energy

          # BESS busy discharging
          self.bess_avail_discharge_energy -= adjusted_discharge_step_size
          self.state.bess_avail_discharge_energy = self.bess_avail_discharge_energy
          self.state.bess_soc = self.state.get_bess_soc()


  def update_bess_cycle_counter(self, action: int):
      """
      Updates the BESS cycle counter based on the current and previous actions.
      The BESS cycle counter is incremented when the BESS transitions between charging and discharging states.
      The counter is reset every 24 steps.
            
      Args:
          action (int): The action index chosen by the agent.
      """
      
      self.state.action_name = self.action_space_dict.get(action, None)

      # Reset BESS cycle counter every 24 steps
      if (self.state.index % 24) == 0:
            self.bess_cycle_counter = 0
            self.state.bess_cycle_counter = self.bess_cycle_counter

      if ( (re.match(r"^charge.*", self.bess_prev_action_name) is not None) and (re.match(r"^discharge.*", self.state.action_name) is not None) ) or \
           ((re.match(r"^discharge.*", self.bess_prev_action_name) is not None) and (re.match(r"^charge.*", self.state.action_name) is not None) ):
            self.bess_cycle_counter += 1
            self.state.bess_cycle_counter = self.bess_cycle_counter
      
      self.bess_prev_action_name = self.state.action_name
      self.state.bess_prev_action_name = self.bess_prev_action_name

    
  def apply_bess_action(self, action: int):
      """
      Applies the chosen BESS action to the environment state.
      This involves updating BESS energy levels, grid import/export,
      and tracking BESS cycles.

      Args:
          action (int): The action index chosen by the agent.
      """
      
      self.state.action_name = self.action_space_dict.get(action, None)

      if self.debug_flag:

          log.info(f"""
          [{self.state.index}] Selected BESS Action Name -> {self.state.action_name}
          """)

      self.update_bess_cycle_counter(action=action)
      
      if re.match(r"^charge.*", self.state.action_name) is not None: # charge

          self.calculate_bess_charge_energy(action=action)
          
      elif re.match(r"^discharge.*", self.state.action_name) is not None: # discharge

          self.calculate_bess_discharge_energy(action=action)

      else: # do-nothing

          self.state.bess_avail_discharge_energy = self.bess_avail_discharge_energy


  def update_monitoring_metrics(self, action: int, raw_reward: float):
        """
        Updates the monitoring metrics for the current state.
        This includes energy values and other relevant statistics.
    
        Returns:
            None
        """

        bess_step_sizes = [-1*s if i <= 1 else s for i, s in enumerate(self.bess_step_sizes)]

        self.monitoring_metrics = {
            "grid_import_energy": self.state.grid_import_energy_unscaled,
            "grid_import_energy_with_bess": self.state.grid_import_energy_with_bess,
            "solar_prod_energy": self.state.solar_prod_energy_unscaled,
            "solar_controller_setpoint": self.state.solar_ctlr_setpoint_unscaled,
            "bess_soc": self.state.bess_soc,
            "bess_avail_discharge": self.state.bess_avail_discharge_energy,
            "bess_discharge_energy": self.state.bess_discharge_energy,
            "bess_charge_from_grid_energy": self.state.bess_charge_from_grid_energy,
            "bess_charge_from_solar_energy": self.state.bess_charge_from_solar_energy,
            "raw_reward_earned": raw_reward,
            "scaled_reward_earned": self.reward,
            "action_energy": bess_step_sizes[action]
        }


  def terminal_state(self) -> bool:
      """
      Checks if the current state is a terminal state (end of the dataset).

      Returns:
          bool: True if the end of the dataset has been reached, False otherwise.
      """

      nr_records = self.env_data.shape[0]

      self.done = (self.state.index + 1) > (nr_records - 1)

      if self.done and self.debug_flag:
          
          log.info("Episode terminal state reached !")

      return self.done
    

  def step(self, action: int):
      """
      Executes one time step in the environment based on the given action.
      Reward calculation uses unscaled values internally via EnvState.
      The final reward is then scaled using tanh.

      Args:
          action: The action taken by the agent.

      Returns:
          tuple: A tuple containing:
              - observation (np.array): The agent's observation of the current environment.
              - reward (float): The amount of reward returned after previous action.
              - terminated (bool): Whether the episode has ended (e.g., reached a terminal state).
              - truncated (bool): Whether the episode was truncated (e.g., due to time limit).
              - info (dict): Contains auxiliary diagnostic information.
      """

      self.reward = 0.0

      self.apply_bess_action(action=action)
      
      raw_reward = self.state.get_reward()
      self.reward = np.tanh(raw_reward / self.reward_scale_factor) * 10
      self.update_monitoring_metrics(action=action, raw_reward=raw_reward)

      if self.debug_flag:
          log.info(f"""
          [{self.state.index}] Raw Reward -> {raw_reward: .3f}
            Scaled Reward -> {self.reward: .3f}
          """)

      self.terminal_state()

      if self.done is False:
          self.state_idx += 1
    
      self.state = self.load_new_state()

      return self.state.get_gym_state(), float(self.reward), self.done, self.truncated, self.monitoring_metrics

    
  def reset(self, seed=None):
      """
      Resets the environment to an initial state.
      
      Args:
          seed (int, optional): The seed that is used to initialize the environment's RNG.
                                Defaults to None.

      Returns:
          tuple: A tuple containing:
              - observation (np.array): The initial observation of the space.
              - info (dict): Contains auxiliary diagnostic information.
      """

      if seed is None:
          np.random.seed(42)
      else:
          np.random.seed(seed)

      self.reward = 0.0
      self.state_idx = 0
      self.done = False
      self.truncated = False
      self.bess_cycle_counter = 0
      self.bess_avail_discharge_energy = self.bess_capacity
      self.bess_prev_action_name = 'do-nothing'
      
      # Load the initial state
      self.state = self.load_new_state()

      # Reset the monitoring metrics
      self.update_monitoring_metrics(action=2, raw_reward=0.0)
      
      return self.state.get_gym_state(), self.monitoring_metrics
  

  def close(self):
        """
        Performs any necessary cleanup.
        Currently, this method does nothing.
        """