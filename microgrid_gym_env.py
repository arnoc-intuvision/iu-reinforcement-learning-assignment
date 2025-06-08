import re
import numpy as np
import pandas as pd
import gymnasium
from gymnasium import spaces
from dataclasses import dataclass, field, asdict
import traceback
import logging as log

log.basicConfig(format="Line:%(lineno)d-%(funcName)s-%(levelname)s:  %(message)s")
log.getLogger().setLevel(log.INFO)

@dataclass
class EnvState:
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
    grid_surplus_energy: float = field(init=False)
    solar_surplus_energy: float = field(init=False)
    solar_vs_load_ratio: float = field(init=False)
    bess_soc: float = field(init=False)
    action_name: str = field(init=False)
    reward_earned: float = field(init=False)

    def __post_init__(self):

        self.grid_surplus_energy = self.get_grid_surplus_energy()
        self.solar_surplus_energy = self.get_solar_surplus_energy()
        self.solar_vs_load_ratio = self.get_solar_vs_load_ratio()
        self.bess_soc = self.get_bess_soc()

    def get_gym_state(self):

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
                         self.bess_soc]) 

    
    def get_grid_surplus_energy(self) -> float:

        return np.round(self.grid_notified_maximum_demand - self.grid_import_energy, 2)

    
    def get_solar_surplus_energy(self) -> float:

        ctl_setpoint_ratio = self.solar_ctlr_setpoint / 100.0
        
        solar_prod_full = self.solar_prod_energy / (ctl_setpoint_ratio + 1e-6)
        
        solar_prod_surplus = solar_prod_full - self.solar_prod_energy

        if self.debug_flag:

            log.info(f"""
            [{self.index}] Solar Surplus Energy Calculation ->
                    Control Setpoint Ratio: {ctl_setpoint_ratio: .2f}
                    Solar Production Energy: {self.solar_prod_energy: .2f}
                    Solar Full Production Energy: {solar_prod_full: .2f}
                    Solar Surplus Production Energy: {solar_prod_surplus: .2f}
             """)
        
        return np.round(solar_prod_surplus, 2)

    
    def get_solar_vs_load_ratio(self) -> float:

        return np.round((self.solar_prod_energy / (self.site_load_energy + 1e-6)) * 100.0, 2)

    
    def get_bess_soc(self) -> float: 
        
        return np.round((self.bess_avail_discharge_energy / self.bess_capacity) * 100.0, 2)

    
    def calculate_bess_charge_cost(self) -> float:

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
            [{self.index}] Future BESS Charge Cost (Discharging): {future_charge_cost: .2f}
                    BESS Grid Charge Energy: {grid_charge_energy: .2f}
                    BESS Solar Charge Energy: {solar_charge_energy: .2f}
                    BESS SoC: {self.bess_soc: .2f}
                    TOU Timeslot: Peak={self.tou_peak} , Standard={self.tou_standard}, Off-peak={self.tou_offpeak}
            """)

        return future_charge_cost


    def calculate_potential_current_discharge_savings_when_do_nothing(self) -> float:

        current_tariff = 1.0
        
        if self.tou_peak:

            current_tariff = self.tou_peak_tariff
            
        elif self.tou_standard:
            
            current_tariff = self.tou_standard_tariff
            
        elif self.tou_offpeak:
            
            current_tariff = self.tou_offpeak_tariff
        
        potential_current_discharge_saving = np.round((current_tariff * self.grid_import_energy * self.bess_cycle_efficiency), 2)

        if self.debug_flag:

            log.info(f"""
            [{self.index}] Potential Current BESS Discharge Savings (Do-Nothing): {potential_current_discharge_saving: .2f}
                    BESS SoC: {self.bess_soc: .2f}
                    TOU Timeslot: Peak={self.tou_peak} , Standard={self.tou_standard}, Off-peak={self.tou_offpeak}
            """)

        return potential_current_discharge_saving


    def calculate_potential_current_charge_cost_when_do_nothing(self) -> float:

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
            """)

        return potential_current_charge_cost

    
    def calculate_without_bess_cost(self) -> float:

        grid_cost = 0.0

        if self.tou_peak:
            
            grid_cost = self.tou_peak_tariff * self.grid_import_energy
            
        elif self.tou_standard:
            
            grid_cost = self.tou_standard_tariff * self.grid_import_energy
            
        elif self.tou_offpeak:
            
            grid_cost = self.tou_offpeak_tariff * self.grid_import_energy

        solar_cost = self.solar_ppa_tariff * self.solar_prod_energy

        total_cost = np.round((grid_cost + solar_cost), 2)

        if self.debug_flag:

            log.info(f"""
            [{self.index}] Total Cost Without BESS: {grid_cost: .2f} + {solar_cost: .2f} = {total_cost: .2f}
            """)

        return total_cost
        

    def calculate_bess_soc_reward(self):

        bess_soc_reward = 0.0

        action_is_charge = re.match(r"^charge.*", self.action_name) is not None

        charged_soc_ratio = self.bess_soc / 100.0
        discharged_soc_ratio = (100.0 - self.bess_soc) / 100.0
            
        if (action_is_charge is True) and (self.bess_soc >= 90.0):
            
            bess_soc_reward = -1 * charged_soc_ratio * 100.0

        elif self.bess_soc >= 80.0:

            bess_soc_reward = charged_soc_ratio * 1000.0

        elif self.bess_soc < 20.0:
            
            bess_soc_reward = -1 * discharged_soc_ratio * 1000.0

        if self.debug_flag:

            log.info(f"""
            [{self.index}] BESS SoC Reward: {bess_soc_reward: .2f}
                    BESS SoC: {self.bess_soc: .2f}
            """)

        return np.round(bess_soc_reward, 2)
    

    def calculate_bess_cycle_penalty(self):

        bess_cycle_penalty = 0.0

        cycle_life = 5000  # Typical Li-ion battery cycle life
        battery_replacement_cost = 100_000
        cycle_cost = battery_replacement_cost / cycle_life
        bess_cycle_penalty = -1 * self.bess_cycle_counter * cycle_cost

        self.bess_prev_action_name = self.action_name

        if self.debug_flag:

            log.info(f"""
            [{self.index}] BESS Cycle Penalty: {bess_cycle_penalty: .2f}
                    BESS Cycle Counter: {self.bess_cycle_counter}
            """)

        return np.round(bess_cycle_penalty, 2)


    def get_reward(self):

        reward = 0.0

        if re.match(r"^charge.*", self.action_name) is not None: # charge

            reward = (- self.calculate_bess_charge_cost() # current cost of charging the bess
                      + self.calculate_potential_future_discharge_savings_when_charging() # future savings when discharging the bess
                      + self.calculate_bess_soc_reward() # penalize the agent if the bess soc falls below 20%
                      + self.calculate_bess_cycle_penalty() # penalize the agent for cycling the bess too much
                     )

        elif re.match(r"^discharge.*", self.action_name) is not None: # discharge

            reward = (+ self.calculate_bess_discharge_saving() # current saving from discharging the bess
                      - self.calculate_potential_future_charge_cost_when_discharging() # future cost for charging the bess
                      + self.calculate_bess_soc_reward() # penalize the agent if the bess soc falls below 20%
                      + self.calculate_bess_cycle_penalty() # penalize the agent for cycling the bess too much
                     )

        else: # do-nothing

            reward = (+ self.calculate_potential_current_charge_cost_when_do_nothing() # saved cost for not charging the bess now
                      - self.calculate_potential_current_discharge_savings_when_do_nothing() # missed savings for not discharging the bess now
                      + self.calculate_bess_soc_reward() # penalize the agent if the bess soc falls below 20%
                     )

        self.reward_earned = reward

        if self.debug_flag:

          log.info(f"""
          [{self.index}] Calculated Reward -> {reward: .3f}
          """)
        
        return reward


class MicrogridEnv(gymnasium.Env):
  metadata = {"render_modes": ["human"]}

  def __init__(self, data: pd.DataFrame, grid_notified_maximum_demand: float, bess_capacity:float, bess_cycle_efficiency: float, bess_step_sizes: list, 
                                         tou_peak_tariff: float, tou_standard_tariff: float, tou_offpeak_tariff: float, solar_ppa_tariff: float, debug_flag: bool):
      
      super(MicrogridEnv).__init__()

      self.env_data = data
      
      self.action_space = spaces.Discrete(5)
      self.action_space_dict = {0: 'charge-1000', 1: 'charge-250', 2: 'do-nothing', 3: 'discharge-250', 4: 'discharge-1000'}

      self.observation_space = np.array([ float(0), float(0), int(0), int(0), int(0), int(0), int(0), int(0), float(0), float(0), float(0), float(0) ])
      
      self.state = None
      self.state_idx = 0
      self.reward = 0.0
      self.reward_scale_factor = self.env_data['grid_import_energy'].max()
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
      self.display_info()


  def display_info(self):
    
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

      return self.env_data


  def load_new_state(self):

      load_profile_data = self.get_data()

      obs = load_profile_data.iloc[self.state_idx, : ]

      state_obj = obs.to_dict()
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
      
      print(f"""
        [{self.state.index}] Environment State ->
                hour_sin: {self.state.ts_hour_sin: .4f}
                hour_cos: {self.state.ts_hour_cos: .4f}
                tou_offpeak: {self.state.tou_offpeak: .0f}
                tou_standard: {self.state.tou_standard: .0f}
                tou_peak: {self.state.tou_peak: .0f}
                day_week: {self.state.day_week: .0f}
                day_saturday: {self.state.day_saturday: .0f}
                day_sunday: {self.state.day_sunday: .0f}
                site_load_energy: {self.state.site_load_energy: .2f} (kWh)
                solar_prod_energy: {self.state.solar_prod_energy: .2f} (kWh)
                solar_ctlr_setpoint: {self.state.solar_ctlr_setpoint: .2f} (%)
                solar_vs_load_ratio: {self.state.get_solar_vs_load_ratio(): .2f} (%)
                grid_import_energy: {self.state.grid_import_energy: .2f} (kWh)
                bess_capacity: {self.state.bess_capacity: .2f} (kWh)
                bess_cycle_efficiency: {self.state.bess_cycle_efficiency: .2f} (kWh)
                bess_avail_discharge_energy: {self.state.bess_avail_discharge_energy: .2f} (kWh)
                bess_soc: {self.state.bess_soc: .2f}
                done: {self.done}
        """)

    
  def get_number_of_actions(self):

      return len(self.action_space_dict)

    
  def sample_action(self):

      return np.random.choice( self.get_number_of_actions() )

    
  def rule_based_policy(self):

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
      
      return action_idx

    
  def calculate_bess_charge_energy(self, action: int):

      charge_step_size = self.bess_step_sizes[action]
      
      required_charge_from_grid_import_energy = charge_step_size - self.state.solar_surplus_energy

      charge_from_solar_energy = 0.0
      charge_from_grid_import_energy = 0.0

      # Determine the possible bess charge energy split between solar and grid
      if (required_charge_from_grid_import_energy > 0) and (self.state.grid_surplus_energy > required_charge_from_grid_import_energy):
          
          charge_from_grid_import_energy = required_charge_from_grid_import_energy
          charge_from_solar_energy = self.state.solar_surplus_energy

      elif (required_charge_from_grid_import_energy > 0) and (self.state.grid_surplus_energy < required_charge_from_grid_import_energy):

          charge_from_grid_import_energy = self.state.grid_surplus_energy
          charge_from_solar_energy = self.state.solar_surplus_energy

      else:

          charge_from_grid_import_energy = 0.0
          charge_from_solar_energy = charge_step_size

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

      discharge_step_size = self.bess_step_sizes[action]

      adjusted_discharge_step_size = 0.0

      # Determine the possible discharge energy from the grid import energy
      if (self.state.grid_import_energy - discharge_step_size) < 0.0:
          
          adjusted_discharge_step_size = self.state.grid_import_energy
          
      else:
          
          adjusted_discharge_step_size = discharge_step_size 
          
      
      # Apply the bess discharge energy to the battery storage, and adjust for overdischarging condition
      if (self.state.bess_avail_discharge_energy - adjusted_discharge_step_size) < 0.0:

          self.state.bess_discharge_energy = self.state.bess_avail_discharge_energy

          # BESS fully discharged
          self.bess_avail_discharge_energy = 0.0
          self.state.bess_avail_discharge_energy = self.bess_avail_discharge_energy
          self.state.bess_soc = self.state.get_bess_soc()

      else:

          self.state.bess_discharge_energy = adjusted_discharge_step_size

          # BESS busy discharging
          self.bess_avail_discharge_energy -= adjusted_discharge_step_size
          self.state.bess_avail_discharge_energy = self.bess_avail_discharge_energy
          self.state.bess_soc = self.state.get_bess_soc()

    
  def apply_bess_action(self, action: int):

      self.state.action_name = self.action_space_dict.get(action, None)

      if self.debug_flag:

          log.info(f"""
          [{self.state.index}] Selected BESS Action Name -> {self.state.action_name}
          """)

      if ( (re.match(r"^charge.*", self.bess_prev_action_name) is not None) and (re.match(r"^discharge.*", self.state.action_name) is not None) ) or \
           ((re.match(r"^discharge.*", self.bess_prev_action_name) is not None) and (re.match(r"^charge.*", self.state.action_name) is not None) ):
            self.bess_cycle_counter += 1
            self.state.bess_cycle_counter = self.bess_cycle_counter
      
      self.bess_prev_action_name = self.state.action_name
      self.state.bess_prev_action_name = self.bess_prev_action_name
      
      if re.match(r"^charge.*", self.state.action_name) is not None: # charge

          self.calculate_bess_charge_energy(action=action)
          
      elif re.match(r"^discharge.*", self.state.action_name) is not None: # discharge

          self.calculate_bess_discharge_energy(action=action)

      else: # do-nothing

          self.state.bess_avail_discharge_energy = self.bess_avail_discharge_energy

    
  def terminal_state(self) -> bool:

      nr_records = self.env_data.shape[0]

      self.done = (self.state.index + 1) > (nr_records - 1)

      if self.done and self.debug_flag:
          
          log.info("Episode terminal state reached !")

      return self.done

    
  def step(self, action):

      self.reward = 0.0

      self.apply_bess_action(action=action)
      
      self.reward = self.state.get_reward() / self.reward_scale_factor

      self.terminal_state()

      if self.done is False:
          self.state_idx += 1
    
      self.state = self.load_new_state()

      return self.state.get_gym_state(), float(self.reward), self.done, self.truncated, {}

    
  def reset(self, seed=None):
      
      if seed is None:
          np.random.seed(42)
      else:
          np.random.seed(seed)

      self.reward = 0.0
      self.reward_scale_factor = self.env_data['grid_import_energy'].max()
      self.bess_avail_discharge_energy = self.bess_capacity
      self.done = False
      self.truncated = False
      self.state_idx = 0
      self.bess_cycle_counter = 0
      self.bess_prev_action_name = 'do-nothing'
      self.state = self.load_new_state()

      # log.info(f"First State: {self.state.get_gym_state()}")
      
      return self.state.get_gym_state(), {}
  

  def close(self):
        pass